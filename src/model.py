import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionLayer(nn.Module):
    """
    Feature-space reflection operator on the latent dimension M.

    Input:
        x: [B, H, W, M]

    At each spatial location (h, w), the latent feature vector x[h, w, :]
    is reflected with respect to an adaptive direction v_hat:

        R(x) = x - 2 <x, v_hat> v_hat

    This operation:
        - acts pointwise in space,
        - only mixes the feature dimension M,
        - preserves spatial locality.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.v_proj = nn.Linear(num_features, num_features)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(x)  # [B, H, W, M]
        v_hat = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)
        proj = torch.sum(x * v_hat, dim=-1, keepdim=True)
        return x - 2.0 * proj * v_hat


class RefractionLayer(nn.Module):
    """
    Feature-space refraction operator on the latent dimension M.

    At each spatial location, the latent feature vector is decomposed into
    parallel and orthogonal components with respect to an adaptive direction:

        x = x_parallel + x_perp

    The refraction operator rescales the orthogonal component:

        T(x) = x_parallel + eta * x_perp

    where:
        - eta is a learnable scalar constrained near 1,
        - the transformation acts only on the feature dimension M,
        - no spatial interaction is introduced.
    """
    def __init__(self, num_features: int, rate_range: float = 0.25):
        super().__init__()
        self.v_proj = nn.Linear(num_features, num_features)
        self.refractive_param = nn.Parameter(torch.tensor(0.0))
        self.rate_range = rate_range
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.v_proj(x)
        v_hat = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)

        x_perp = torch.sum(x * v_hat, dim=-1, keepdim=True) * v_hat
        x_parallel = x - x_perp

        eta = 1.0 + self.rate_range * torch.tanh(self.refractive_param)
        return x_parallel + eta * x_perp


# TODO： O(N^2) -> O(N)
class ScatteringLayer(nn.Module):
    """
    Spatial scattering operator with relative positional bias.

    Input:
        x: [B, H, W, M]

    Interpretation:
        - M represents the latent optical feature dimension,
        - scattering operates over the physical spatial domain (H, W),
        - each spatial location is treated as a node in a fully-connected graph.

    The operator constructs an input-adaptive kernel:

        K_ij = softmax_j( q_i^T k_j / sqrt(d) + b(p_i - p_j) )

    where:
        - i, j index spatial locations,
        - p_i, p_j are normalized coordinates,
        - b(p_i - p_j) is a learnable distance-based bias.

    The update rule is:

        S(x)_i = sigma * ( sum_j K_ij v_j - x_i )

    This layer captures:
        - non-local spatial interactions,
        - content-dependent propagation,
        - distance-aware attenuation via relative positional bias.
    """
    def __init__(self, num_features: int, hidden_dim: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim or num_features

        self.q_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(num_features, num_features, bias=False)

        # Learnable distance-based relative positional bias:
        #   bias_ij = - softplus(tau) * ||p_i - p_j||^2
        # Larger tau implies a stronger locality prior.
        self.log_tau = nn.Parameter(torch.tensor(0.0))

        # Scattering strength / dissipation factor.
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.scale = self.hidden_dim ** -0.5

    @staticmethod
    def _build_coords(H: int, W: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
        coords = coords.reshape(H * W, 2)               # [N, 2]
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, M]
        """
        B, H, W, M = x.shape
        N = H * W

        # Flatten physical space while keeping the latent feature dimension.
        x_flat = x.reshape(B, N, M)  # [B, N, M]

        # Content-based projections.
        q = self.q_proj(x_flat)  # [B, N, d]
        k = self.k_proj(x_flat)  # [B, N, d]
        v = self.v_proj(x_flat)  # [B, N, M]

        # Content similarity logits.
        logits = torch.einsum("bid,bjd->bij", q, k) * self.scale  # [B, N, N]

        # Relative positional bias based on squared Euclidean distance.
        coords = self._build_coords(H, W, x.device, x.dtype)      # [N, 2]
        rel = coords[:, None, :] - coords[None, :, :]             # [N, N, 2]
        dist2 = (rel ** 2).sum(dim=-1)                            # [N, N]

        tau = F.softplus(self.log_tau)
        pos_bias = -tau * dist2                                   # [N, N]

        logits = logits + pos_bias.unsqueeze(0)                   # [B, N, N]

        # Stabilize softmax.
        logits = logits - logits.max(dim=-1, keepdim=True).values
        K = F.softmax(logits, dim=-1)                             # [B, N, N]

        # Spatial propagation.
        y = torch.einsum("bij,bjm->bim", K, v)                    # [B, N, M]

        sigma = self.log_sigma.exp()
        y = sigma * (y - x_flat)

        return y.reshape(B, H, W, M)


class FeatureMLP(nn.Module):
    """
    Pointwise nonlinear transformation on the latent feature dimension M.

    Input / Output:
        x: [B, H, W, M]

    This module:
        - operates independently at each spatial location,
        - enhances feature expressiveness,
        - does not introduce spatial coupling.
    """
    def __init__(self, num_features: int, expansion: int = 2):
        super().__init__()
        hidden = expansion * num_features
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LightEvolutionBlock(nn.Module):
    """
    Core evolution block combining feature-space optics and spatial propagation.

    Input:
        x: [B, H, W, M]

    Modeling decomposition:
        - Reflection and refraction act in feature space (M),
        - Scattering acts in physical space (H, W).

    The update rule is:

        x -> alpha_r * R(x)
           + alpha_t * T(x)
           + alpha_s * S(x)

    where:
        - alpha_r, alpha_t, alpha_s are learned global mixing weights.

    The block further includes:
        - a residual linear projection,
        - a pointwise nonlinear feature transformation (MLP).

    This design separates:
        - feature transformation (optical analogy),
        - spatial interaction (kernel propagation).
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)

        self.reflection = ReflectionLayer(num_features)
        self.refraction = RefractionLayer(num_features)
        self.scattering = ScatteringLayer(num_features)

        self.gate = nn.Sequential(
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, 3),
        )

        self.out_proj = nn.Linear(num_features, num_features)
        self.ffn = FeatureMLP(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, M]
        """
        h = self.norm1(x)

        xr = self.reflection(h)   # feature-space transform
        xt = self.refraction(h)   # feature-space transform
        xs = self.scattering(h)   # spatial propagation

        pooled = h.mean(dim=(1, 2))                    # [B, M]
        alpha = F.softmax(self.gate(pooled), dim=-1)  # [B, 3]

        a_r = alpha[:, 0].view(-1, 1, 1, 1)
        a_t = alpha[:, 1].view(-1, 1, 1, 1)
        a_s = alpha[:, 2].view(-1, 1, 1, 1)

        mix = a_r * xr + a_t * xt + a_s * xs

        x = x + self.out_proj(mix)
        x = x + self.ffn(self.norm2(x))
        return x


class LiftingLayer(nn.Module):
    """
    Linear lifting from input field to latent feature space.

    Input:
        x: [B, H, W, in_channels]

    Output:
        y: [B, H, W, M]

    This layer:
        - embeds raw inputs into a latent feature space,
        - does not introduce spatial coupling.
    """
    def __init__(self, in_channels: int, num_features: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProjectionLayer(nn.Module):
    """
    Linear projection from latent feature space to output field.

    Input:
        x: [B, H, W, M]

    Output:
        y: [B, H, W, out_channels]

    This layer maps the learned latent representation back to the target space.
    """
    def __init__(self, num_features: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LightNeuralOperator2D(nn.Module):
    """
    Light-inspired neural operator for 2D problems.

    Architecture overview:
        - H, W represent physical spatial coordinates,
        - M represents latent optical feature channels,
        - reflection and refraction operate in feature space,
        - scattering models spatial propagation via a learned kernel.

    Pipeline:
        input -> lifting -> stacked evolution blocks -> projection -> output

    This model can be interpreted as:
        a hybrid operator combining feature-space transformations and
        non-local spatial kernel propagation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int = 16,
        depth: int = 4,
    ):
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        self.lifting = LiftingLayer(in_channels, num_features)

        self.blocks = nn.ModuleList([
            LightEvolutionBlock(num_features)
            for _ in range(depth)
        ])

        self.projection = ProjectionLayer(num_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, in_channels]
        """
        x = self.lifting(x)  # [B, H, W, M]

        for blk in self.blocks:
            x = blk(x)

        y = self.projection(x)  # [B, H, W, out_channels]
        return y


if __name__ == "__main__":
    model = LightNeuralOperator2D(
        in_channels=3,
        out_channels=1,
        num_features=16,
        depth=2,
    )

    x = torch.randn(2, 32, 32, 3)
    y = model(x)
    print(y.shape)  # torch.Size([2, 32, 32, 1])