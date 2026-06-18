"""Grape PG-GNN: GraphSAGE adapted from PG-GNN.py for tabular vine graphs.

Adapted from PG-GNN/PG-GNN.py (AAAI 2023):
  - Original GraphMask layer (lines 37-47) → optional edge weighting in SAGEConv.
  - Original global_model (lines 50-92): GRU + temporal attention + neighbor mean
    aggregation over 199-county adjacency → GrapeGraphSAGE with PyG GraphSAGE layers.
  - Original gnn_train / gnn_rmse_testing_evaluation (lines 95-148) → train loop in train.py.
  - Spatial adjacency from shapefile (Rook, lines 200-212) → graph_builder.py strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .grape_pg_an import GroupAttentionEncoder


class GrapeGraphSAGE(nn.Module):
    """GraphSAGE model with group attention node encoder (PG-AN + PG-GNN hybrid)."""

    def __init__(
        self,
        group_dims: dict[str, int],
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GroupAttentionEncoder(group_dims, hidden=hidden, dropout=dropout)
        self.convs = nn.ModuleList()
        in_dim = hidden
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden))
            in_dim = hidden
        self.dropout = dropout
        # Adapted from PG-GNN global_model yield head (lines 83-88)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, group_tensors: dict[str, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(group_tensors)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h)
