#############################################################################################################################################
#  _     _       _     _        _                 _              _   _   _                      _    ___                       _
# | |   (_) __ _| |__ | |_     (_)_ __  ___ _ __ (_)_ __ ___  __| | | \ | | ___ _   _ _ __ __ _| |  / _ \ _ __   ___ _ __ __ _| |_ ___  _ __
# | |   | |/ _` | '_ \| __|____| | '_ \/ __| '_ \| | '__/ _ \/ _` | |  \| |/ _ \ | | | '__/ _` | | | | | | '_ \ / _ \ '__/ _` | __/ _ \| '__|
# | |___| | (_| | | | | ||_____| | | | \__ \ |_) | | | |  __/ (_| | | |\  |  __/ |_| | | | (_| | | | |_| | |_) |  __/ | | (_| | || (_) | |
# |_____|_|\__, |_| |_|\__|    |_|_| |_|___/ .__/|_|_|  \___|\__,_| |_| \_|\___|\__,_|_|  \__,_|_|  \___/| .__/ \___|_|  \__,_|\__\___/|_|
#         |___/                           |_|                                                           |_|

"""
Light-inspired neural operator building blocks for 1D and 2D PDE problems.

Layout convention: the field / latent channel dimension is **last** in the
tensor (channels-last), for example ``[B, L, C]`` in 1D and ``[B, H, W, C]`` in
2D. Inside the trunk, ``C`` becomes the latent width ``M`` after lifting.
"""
#############################################################################################################################################

from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionLayer(nn.Module):
    """
    Feature-space reflection on the latent dimension ``M``.

    Input:
        x: [B, *spatial, M]

    The reflection operator is implemented in a matrix-free Householder form:
        R(x) = x - 2 <x, v_hat> v_hat,

    which is equivalent to applying
        H = I - 2 v_hat v_hat^T

    without explicitly constructing the dense M x M Householder matrix.
    The transform is pointwise over the spatial domain and acts only along
    the latent feature dimension.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.v_proj = nn.Linear(num_features, num_features)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply reflection in feature space; shape is preserved."""
        v = self.v_proj(x)
        v_hat = F.normalize(v, p=2, dim=-1, eps=self.eps)
        # proj = <x, v_hat>
        proj = torch.sum(x * v_hat, dim=-1, keepdim=True)
        # Matrix-free Householder reflection: R(x) = x - 2 <x, v_hat> v_hat
        return x - 2.0 * proj * v_hat


class RefractionLayer(nn.Module):
    """
    Feature-space refraction on the latent dimension ``M``.

    Input:
        x: [B, *spatial, M]

    The refraction operator is implemented in a matrix-free form:
        T(x) = x + (eta - 1) <x, v_hat> v_hat,

    which is equivalent to
        T(x) = x_parallel + eta * x_perp,

    but avoids explicitly constructing x_parallel. The transform is pointwise
    over the spatial domain and acts only along the latent feature dimension.
    """

    def __init__(self, num_features: int, rate_range: float = 0.25):
        super().__init__()
        self.v_proj = nn.Linear(num_features, num_features)
        self.refractive_param = nn.Parameter(torch.tensor(0.0))
        self.rate_range = rate_range
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply refraction in feature space; shape is preserved."""
        # # The original formulation computes the parallel and perpendicular components
        # v = self.v_proj(x)
        # v_hat = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)
        # x_perp = torch.sum(x * v_hat, dim=-1, keepdim=True) * v_hat
        # x_parallel = x - x_perp
        # eta = 1.0 + self.rate_range * torch.tanh(self.refractive_param)
        # return x_parallel + eta * x_perp

        # The refactored matrix-free implementation avoids explicitly computing x_parallel:
        v = self.v_proj(x)
        v_hat = F.normalize(v, p=2, dim=-1, eps=self.eps)
        # x_perp = <x, v_hat> v_hat
        proj = torch.sum(x * v_hat, dim=-1, keepdim=True)
        x_perp = proj * v_hat
        eta = 1.0 + self.rate_range * torch.tanh(self.refractive_param)
        # Matrix-free rank-one update: T(x) = x + (eta - 1) x_perp
        return x + (eta - 1.0) * x_perp


class ScatteringLayer(nn.Module):
    """
    Spatial scattering with full attention and relative positional bias (``O(N^2)``).

    Input:
        ``x``: ``[B, *spatial, M]`` where ``*spatial`` is length ``L`` in 1D or
        ``(H, W)`` in 2D.

    Interpretation:
        - M represents the latent optical feature dimension,
        - scattering operates over the physical spatial domain ``L`` in 1D or ``(H, W)`` in 2D,
        - each spatial location is treated as a node in a fully-connected graph.

    The operator constructs an input-adaptive kernel:

        ``K_ij = softmax_j( q_i^T k_j / sqrt(d) + b(p_i - p_j) )``

    with normalized coordinates ``p`` in ``spatial_dims`` dimensions and a
    Gaussian-style bias ``b = -softplus(tau) * ||p_i - p_j||^2``. Larger
    ``tau`` encourages locality. The update rule is

        ``S(x)_i = sigma * ( sum_j K_ij v_j - x_i )``.

    This layer captures:
        - non-local spatial interactions,
        - content-dependent propagation,
        - distance-aware attenuation via relative positional bias.
    """

    def __init__(
        self, num_features: int, spatial_dims: int, hidden_dim: int | None = None
    ):
        super().__init__()
        if spatial_dims not in (1, 2):
            raise ValueError(
                f"Only 1D/2D spatial domains are supported, got {spatial_dims}"
            )

        self.num_features = num_features
        self.spatial_dims = spatial_dims
        self.hidden_dim = hidden_dim or num_features

        self.q_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(num_features, num_features, bias=False)

        # Learnable distance-based relative positional bias:
        #   bias_ij = - softplus(tau) * ||p_i - p_j||^2
        # Larger tau implies a stronger locality prior.
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        # Scattering strength / dissipation factor
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.scale = self.hidden_dim**-0.5

    def _build_coords(
        self, spatial_shape: Sequence[int], device, dtype
    ) -> torch.Tensor:
        """Normalized grid coordinates of shape ``[N, spatial_dims]`` for ``N = prod(spatial_shape)``."""
        axes = [
            torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            for n in spatial_shape
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(mesh, dim=-1)
        return coords.reshape(-1, self.spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, *spatial, M]`` latent features.

        Returns:
            Scattering residual of the same shape as ``x``.
        """
        spatial_shape = x.shape[1:-1]
        B, M = x.shape[0], x.shape[-1]
        N = 1
        for n in spatial_shape:
            N *= n

        x_flat = x.reshape(B, N, M)
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        logits = torch.einsum("bid,bjd->bij", q, k) * self.scale

        coords = self._build_coords(spatial_shape, x.device, x.dtype)
        rel = coords[:, None, :] - coords[None, :, :]
        dist2 = (rel**2).sum(dim=-1)

        tau = F.softplus(self.log_tau)
        logits = logits + (-tau * dist2).unsqueeze(0)

        # Stabilize softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values
        attn = F.softmax(logits, dim=-1)
        # Spatial propagation
        y = torch.einsum("bij,bjm->bim", attn, v)

        sigma = self.log_sigma.exp()
        y = sigma * (y - x_flat)
        return y.reshape(B, *spatial_shape, M)


class EfficientScatteringLayer(nn.Module):
    """
    Efficient spatial scattering: linear complexity in ``N = prod(*spatial)``.

    Input:
        ``x``: ``[B, *spatial, M]``

    Design rationale:
        The original implementation realizes the scattering kernel

            K_ij = softmax_j(q_i^T k_j + b_ij),

        which is faithful to the formulation in the method note, but requires
        O(N^2) memory and compute. Here we replace it with a linear-attention
        kernel approximation and add a lightweight local branch to preserve the
        locality prior induced by the Gaussian positional bias.

    The resulting update keeps the same operator-level interpretation:

        S(x)_i = sigma * ( weighted spatial average - x_i ),

    but the weighted average is computed in O(N) with respect to the number of
    spatial points.
    """

    def __init__(
        self,
        num_features: int,
        spatial_dims: int,
        hidden_dim: int | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if spatial_dims not in (1, 2):
            raise ValueError(
                f"Only 1D/2D spatial domains are supported, got {spatial_dims}"
            )

        self.num_features = num_features
        self.spatial_dims = spatial_dims
        self.hidden_dim = hidden_dim or num_features
        self.eps = eps

        self.q_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(num_features, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(num_features, num_features, bias=False)

        # A small coordinate encoder injects absolute spatial information into
        # the query/key construction without forming an O(N^2) bias matrix.
        self.coord_mlp = nn.Sequential(
            nn.Linear(spatial_dims, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Lightweight local propagation branch. This serves as a surrogate for
        # the Gaussian locality prior in the original relative positional bias.
        conv_cls = nn.Conv1d if spatial_dims == 1 else nn.Conv2d
        self.local_conv = conv_cls(
            num_features,
            num_features,
            kernel_size=3,
            padding=1,
            groups=num_features,
            bias=False,
        )
        self.local_proj = nn.Linear(num_features, num_features, bias=False)

        # Learnable mixing between global linear attention and local diffusion.
        self.local_gate = nn.Parameter(torch.tensor(-1.0))

        # Scattering strength / dissipation factor.
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.scale = self.hidden_dim**-0.5

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """Positive map ``phi`` used in linearized attention (ELU + 1)."""
        return F.elu(x) + 1.0

    def _build_coords(
        self, spatial_shape: Sequence[int], device, dtype
    ) -> torch.Tensor:
        """Same normalized grid as :meth:`ScatteringLayer._build_coords`."""
        axes = [
            torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            for n in spatial_shape
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        coords = torch.stack(mesh, dim=-1)
        return coords.reshape(-1, self.spatial_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, *spatial, M]``.

        Returns:
            Scattering residual of the same shape as ``x``.
        """
        spatial_shape = x.shape[1:-1]
        B, M = x.shape[0], x.shape[-1]
        N = 1
        for n in spatial_shape:
            N *= n

        x_flat = x.reshape(B, N, M)
        v = self.v_proj(x_flat)

        coords = self._build_coords(spatial_shape, x.device, x.dtype)
        coord_feat = self.coord_mlp(coords).unsqueeze(0)

        q = self.q_proj(x_flat) * self.scale + coord_feat
        k = self.k_proj(x_flat) * self.scale + coord_feat

        q_phi = self._feature_map(q)
        k_phi = self._feature_map(k)

        # Linear attention:
        #   y_i = (phi(q_i)^T sum_j phi(k_j) v_j) / (phi(q_i)^T sum_j phi(k_j))
        kv = torch.einsum("bnd,bnm->bdm", k_phi, v)
        k_sum = k_phi.sum(dim=1)

        global_num = torch.einsum("bnd,bdm->bnm", q_phi, kv)
        global_den = torch.einsum("bnd,bd->bn", q_phi, k_sum)
        global_y = global_num / (global_den.unsqueeze(-1) + self.eps)

        # Local branch: cheap spatial mixing to retain short-range propagation
        if self.spatial_dims == 1:
            x_local = x.permute(0, 2, 1)
            local_y = self.local_conv(x_local).permute(0, 2, 1)
        else:
            x_local = x.permute(0, 3, 1, 2)
            local_y = self.local_conv(x_local).permute(0, 2, 3, 1)

        local_y = self.local_proj(local_y.reshape(B, N, M))

        beta = torch.sigmoid(self.local_gate)
        y = (1.0 - beta) * global_y + beta * local_y

        sigma = self.log_sigma.exp()
        y = sigma * (y - x_flat)
        return y.reshape(B, *spatial_shape, M)


class FeatureMLP(nn.Module):
    """
    Pointwise MLP on the latent dimension ``M``.

    Input / output:
        ``x``: ``[B, *spatial, M]``

    Applied independently at each spatial location; introduces no explicit
    spatial mixing beyond what upstream layers already encoded in ``M``.
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
        ``x``: ``[B, *spatial, M]``

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

    def __init__(
        self, num_features: int, spatial_dims: int, scattering_type: str = "efficient"
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.reflection = ReflectionLayer(num_features)
        self.refraction = RefractionLayer(num_features)
        st = scattering_type.lower()
        if st == "standard":
            self.scattering = ScatteringLayer(num_features, spatial_dims=spatial_dims)
        elif st == "efficient":
            self.scattering = EfficientScatteringLayer(
                num_features, spatial_dims=spatial_dims
            )
        else:
            raise ValueError(
                f"scattering_type must be 'standard' or 'efficient', got {scattering_type!r}"
            )
        self.gate = nn.Sequential(
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, 3),
        )
        self.out_proj = nn.Linear(num_features, num_features)
        self.ffn = FeatureMLP(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual update; output shape matches ``x``."""
        h = self.norm1(x)
        xr = self.reflection(h)
        xt = self.refraction(h)
        xs = self.scattering(h)

        reduce_dims = tuple(range(1, h.ndim - 1))
        pooled = h.mean(dim=reduce_dims)
        alpha = F.softmax(self.gate(pooled), dim=-1)

        shape = [x.shape[0]] + [1] * (x.ndim - 2) + [1]
        a_r = alpha[:, 0].view(*shape)
        a_t = alpha[:, 1].view(*shape)
        a_s = alpha[:, 2].view(*shape)

        mix = a_r * xr + a_t * xt + a_s * xs
        x = x + self.out_proj(mix)
        x = x + self.ffn(self.norm2(x))
        return x


class LiftingLayer(nn.Module):
    """
    Linear lifting from input field to latent feature space.

    Input:
        ``x``: ``[B, *spatial, in_channels]``

    Output:
        ``[B, *spatial, M]`` — embedding only; no spatial coupling.

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
        ``x``: ``[B, *spatial, M]``

    Output:
        ``[B, *spatial, out_channels]``.

    This layer maps the learned latent representation back to the target space.
    """

    def __init__(self, num_features: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(num_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LightNeuralOperator(nn.Module):
    """
    Light-inspired neural operator for 1D/2D problems.

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
        spatial_dims: int,
        num_features: int = 16,
        depth: int = 4,
        scattering_type: str = "efficient",
    ):
        super().__init__()
        if spatial_dims not in (1, 2):
            raise ValueError(f"Only 1D/2D are supported, got {spatial_dims}")
        self.spatial_dims = spatial_dims
        self.num_features = num_features
        self.depth = depth
        self.scattering_type = scattering_type

        self.lifting = LiftingLayer(in_channels, num_features)
        self.blocks = nn.ModuleList(
            [
                LightEvolutionBlock(
                    num_features,
                    spatial_dims=spatial_dims,
                    scattering_type=scattering_type,
                )
                for _ in range(depth)
            ]
        )
        self.projection = ProjectionLayer(num_features, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Field tensor whose ndim is ``spatial_dims + 2``.

        Returns:
            Predicted field with ``out_channels`` on the last dimension.
        """
        expected_ndim = self.spatial_dims + 2
        if x.ndim != expected_ndim:
            raise ValueError(
                f"Expected input ndim={expected_ndim} for spatial_dims={self.spatial_dims}, got shape {tuple(x.shape)}"
            )
        x = self.lifting(x)
        for blk in self.blocks:
            x = blk(x)
        return self.projection(x)


class LightNeuralOperator1D(LightNeuralOperator):
    """Convenience wrapper with ``spatial_dims=1`` (sequence / 1D grid)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int = 16,
        depth: int = 4,
        scattering_type: str = "efficient",
    ):
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=1,
            num_features=num_features,
            depth=depth,
            scattering_type=scattering_type,
        )


class LightNeuralOperator2D(LightNeuralOperator):
    """Convenience wrapper with ``spatial_dims=2`` (image / 2D grid)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: int = 16,
        depth: int = 4,
        scattering_type: str = "efficient",
    ):
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=2,
            num_features=num_features,
            depth=depth,
            scattering_type=scattering_type,
        )
