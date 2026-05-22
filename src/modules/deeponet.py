from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple fully-connected network used by the DeepONet branch and trunk nets.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 128,
        depth: int = 4,
        activation: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be at least 2, got {depth}")

        layers: list[nn.Module] = [nn.Linear(in_dim, width), activation()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), activation()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepONet1D(nn.Module):
    """
    DeepONet baseline for 1D operator learning on a fixed grid.

    Input:
        x: [B, L, C]. The first ``branch_in_channels`` channels are flattened
        and passed to the branch network. If a coordinate channel is available,
        the last channel is used as trunk input; otherwise a normalized grid is
        built internally.

    Output:
        y: [B, L, out_channels].
    """

    def __init__(
        self,
        num_branch_inputs: int,
        branch_in_channels: int = 1,
        coord_dim: int = 1,
        out_channels: int = 1,
        basis_dim: int = 128,
        branch_width: int = 128,
        trunk_width: int = 128,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        use_bias: bool = True,
    ):
        super().__init__()
        self.num_branch_inputs = num_branch_inputs
        self.branch_in_channels = branch_in_channels
        self.coord_dim = coord_dim
        self.out_channels = out_channels
        self.basis_dim = basis_dim

        self.branch_net = MLP(
            in_dim=num_branch_inputs,
            out_dim=out_channels * basis_dim,
            width=branch_width,
            depth=branch_depth,
        )
        self.trunk_net = MLP(
            in_dim=coord_dim,
            out_dim=out_channels * basis_dim,
            width=trunk_width,
            depth=trunk_depth,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if use_bias else None

    def _default_coords(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[0], x.shape[1]
        coord = torch.linspace(0.0, 1.0, L, device=x.device, dtype=x.dtype)
        coord = coord.view(1, L, 1).expand(B, L, 1)
        return coord

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"DeepONet1D expects [B, L, C], got {tuple(x.shape)}")
        if x.shape[-1] < self.branch_in_channels:
            raise ValueError(
                f"Input has {x.shape[-1]} channels, but branch_in_channels="
                f"{self.branch_in_channels}."
            )

        branch_x = x[..., : self.branch_in_channels].reshape(x.shape[0], -1)
        if branch_x.shape[-1] != self.num_branch_inputs:
            raise ValueError(
                f"Branch input dimension mismatch: expected {self.num_branch_inputs}, "
                f"got {branch_x.shape[-1]}. Check --target-size and --branch-in-channels."
            )

        # In the default Burgers setting, x[..., -1:] is the normalized coordinate
        # channel produced by build_operator_split(use_coord=True).
        if x.shape[-1] >= self.branch_in_channels + self.coord_dim:
            trunk_x = x[..., -self.coord_dim :]
        else:
            trunk_x = self._default_coords(x)

        B, L = x.shape[0], x.shape[1]
        branch = self.branch_net(branch_x).view(B, self.out_channels, self.basis_dim)
        trunk = self.trunk_net(trunk_x).view(B, L, self.out_channels, self.basis_dim)
        y = torch.einsum("bop,blop->blo", branch, trunk)  # vanilla DeepONet
        if self.bias is not None:
            y = y + self.bias.view(1, 1, self.out_channels)
        return y


__all__ = ["DeepONet1D"]
