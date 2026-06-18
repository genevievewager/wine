"""Grape PG-AN: Physics-Guided Attention Network adapted for tabular vine-year data.

Adapted from PG-AN/PG-AN.py (AAAI 2023):
  - Original: GRU(64) over 365-day sequences + softmax attention over timesteps,
    then Dense head for yield (lines 288-315).
  - Here: group-wise attention over tabular feature groups (vigor, historical yield,
    weather, spatial/management) replaces temporal attention.
  - Original custom_loss with NaN mask (lines 311-313) → masked_mse in PyTorch.
  - Original energy conservation loss (GPP+Ra+Rh+NEE) → agronomy-guided penalties
    in GrapePhysicsLoss (negative yield, temporal jumps, vigor-yield inconsistency).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


def masked_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Adapted from PG-AN.py custom_loss (NaN-safe MSE)."""
    mask = torch.isfinite(y_true)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=y_true.device)
    return F.mse_loss(y_pred[mask], y_true[mask])


@dataclass
class PhysicsLossConfig:
    lambda_negative: float = 0.1
    lambda_temporal: float = 0.05
    lambda_vigor: float = 0.05
    use_negative_penalty: bool = True
    use_temporal_penalty: bool = True
    use_vigor_penalty: bool = True
    max_year_jump: float = 15.0
    vigor_yield_ratio_threshold: float = 3.0


class GrapePhysicsLoss(nn.Module):
    """Modular agronomy-guided loss terms."""

    def __init__(self, config: PhysicsLossConfig | None = None):
        super().__init__()
        self.config = config or PhysicsLossConfig()

    def negative_yield_penalty(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(F.relu(-y_pred) ** 2)

    def temporal_jump_penalty(
        self,
        y_pred: torch.Tensor,
        prev_yield: torch.Tensor | None,
    ) -> torch.Tensor:
        if prev_yield is None:
            return torch.tensor(0.0, device=y_pred.device)
        jump = torch.abs(y_pred.squeeze() - prev_yield.squeeze())
        excess = F.relu(jump - self.config.max_year_jump)
        return torch.mean(excess ** 2)

    def vigor_penalty(
        self,
        y_pred: torch.Tensor,
        shoots: torch.Tensor | None,
        canopy: torch.Tensor | None,
    ) -> torch.Tensor:
        if shoots is None or canopy is None:
            return torch.tensor(0.0, device=y_pred.device)
        vigor = (shoots + canopy) / 2.0
        low_vigor = F.relu(1.0 - vigor)
        high_yield = F.relu(y_pred.squeeze() - vigor * self.config.vigor_yield_ratio_threshold)
        return torch.mean(low_vigor * high_yield)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        prev_yield: torch.Tensor | None = None,
        shoots: torch.Tensor | None = None,
        canopy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        c = self.config
        loss = masked_mse(y_true, y_pred)
        parts = {"mse": float(loss.detach())}

        if c.use_negative_penalty:
            p = self.negative_yield_penalty(y_pred)
            loss = loss + c.lambda_negative * p
            parts["negative"] = float(p.detach())

        if c.use_temporal_penalty:
            p = self.temporal_jump_penalty(y_pred, prev_yield)
            loss = loss + c.lambda_temporal * p
            parts["temporal"] = float(p.detach())

        if c.use_vigor_penalty:
            p = self.vigor_penalty(y_pred, shoots, canopy)
            loss = loss + c.lambda_vigor * p
            parts["vigor"] = float(p.detach())

        return loss, parts


class GroupAttentionEncoder(nn.Module):
    """Feature-group attention inspired by PG-AN.py attn_yield block (lines 292-300).

    Original applies Dense layers on (365, 19) then softmax over time axis.
    Here we encode each feature group, compute attention weights, and aggregate.
    """

    def __init__(self, group_dims: dict[str, int], hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.group_names = list(group_dims.keys())
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            for name, dim in group_dims.items()
            if dim > 0
        })
        self.attn = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, group_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = []
        names = []
        for name in self.group_names:
            if name not in self.encoders or name not in group_tensors:
                continue
            h = self.encoders[name](group_tensors[name])
            encoded.append(h)
            names.append(name)
        if not encoded:
            raise ValueError("No feature groups with non-zero dimension.")
        stack = torch.stack(encoded, dim=1)  # [B, G, H]
        attn_logits = self.attn(stack).squeeze(-1)  # [B, G]
        weights = F.softmax(attn_logits, dim=1)
        context = torch.sum(stack * weights.unsqueeze(-1), dim=1)
        return self.dropout(context)


class GrapePGAN(nn.Module):
    """Tabular PG-AN for grape yield prediction."""

    def __init__(
        self,
        group_dims: dict[str, int],
        hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Adapted from PG-AN.py: GRU base → group encoder; yield head Dense stack (lines 301-305)
        self.encoder = GroupAttentionEncoder(group_dims, hidden=hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, group_tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(group_tensors)
        return self.head(h)


def split_features_by_group(
    X: torch.Tensor,
    group_indices: dict[str, list[int]],
) -> dict[str, torch.Tensor]:
    out = {}
    for name, idxs in group_indices.items():
        if idxs:
            out[name] = X[:, idxs]
    return out
