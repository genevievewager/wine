"""Build PyTorch Geometric edge_index for vine/block graphs.

Graph strategies (adapted from PG-GNN.py spatial adjacency via libpysal Rook;
here we use vineyard row/block structure from tabular metadata):

  A. same_vineyard_row_adjacent — vines in same vineyard+row with adjacent vine_position
  B. same_block — all nodes in the same block
  C. similar_yield — k-nearest by historical yield within vineyard
  D. hybrid — spatial adjacency + same block edges
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist

from . import config as cfg


class GraphStrategy(str, Enum):
    SAME_VINEYARD_ROW_ADJACENT = "same_vineyard_row_adjacent"
    SAME_BLOCK = "same_block"
    SIMILAR_YIELD = "similar_yield"
    HYBRID = "hybrid"


def _undirected_edges(pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    edges = np.array(pairs, dtype=np.int64).T
    # add reverse
    rev = edges[::-1].copy()
    both = np.concatenate([edges, rev], axis=1)
    both = np.unique(both, axis=1)
    return both


def build_same_vineyard_row_adjacent(df: pd.DataFrame, node_index: pd.Series) -> np.ndarray:
    """Option A: connect adjacent vine_position within vineyard+row (or block proxy)."""
    pairs = []
    idx_map = node_index.to_dict()
    group_cols = [cfg.COL_VINEYARD, cfg.COL_ROW] if cfg.COL_ROW in df.columns else [cfg.COL_VINEYARD, cfg.COL_BLOCK]
    for _, g in df.groupby(group_cols):
        g = g.sort_values(cfg.COL_VINE_POSITION if cfg.COL_VINE_POSITION in g.columns else cfg.COL_BLOCK)
        nodes = [idx_map[i] for i in g.index]
        for a, b in zip(nodes[:-1], nodes[1:]):
            pairs.append((a, b))
    return _undirected_edges(pairs)


def build_same_block(df: pd.DataFrame, node_index: pd.Series) -> np.ndarray:
    """Option B: fully connect nodes sharing vineyard+block+year."""
    pairs = []
    idx_map = node_index.to_dict()
    for _, g in df.groupby([cfg.COL_VINEYARD, cfg.COL_BLOCK, cfg.COL_YEAR]):
        nodes = [idx_map[i] for i in g.index]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.append((nodes[i], nodes[j]))
    return _undirected_edges(pairs)


def build_similar_yield(
    df: pd.DataFrame,
    node_index: pd.Series,
    k: int = 3,
    yield_col: str = cfg.COL_PREV_ACTUAL,
) -> np.ndarray:
    """Option C: connect k most similar historical yields within vineyard."""
    pairs = []
    idx_map = node_index.to_dict()
    col = yield_col if yield_col in df.columns else cfg.COL_ACTUAL_YIELD
    for vineyard, g in df.groupby(cfg.COL_VINEYARD):
        yields = g[col].fillna(g[col].median()).values.reshape(-1, 1)
        dist = cdist(yields, yields, metric="euclidean")
        for i, row_idx in enumerate(g.index):
            nn = np.argsort(dist[i])[1 : k + 1]
            for j in nn:
                pairs.append((idx_map[row_idx], idx_map[g.index[j]]))
    return _undirected_edges(pairs)


def build_hybrid(df: pd.DataFrame, node_index: pd.Series, k: int = 2) -> np.ndarray:
    """Option D: union of row-adjacent and same-block edges."""
    e1 = build_same_vineyard_row_adjacent(df, node_index)
    e2 = build_same_block(df, node_index)
    if e1.size == 0:
        combined = e2
    elif e2.size == 0:
        combined = e1
    else:
        combined = np.concatenate([e1, e2], axis=1)
    combined = np.unique(combined, axis=1)
    return combined


def build_edge_index(
    df: pd.DataFrame,
    strategy: GraphStrategy | str = GraphStrategy.HYBRID,
    k_similar: int = 3,
) -> torch.Tensor:
    """Return edge_index tensor [2, num_edges] for PyG."""
    df = df.reset_index(drop=True)
    node_index = pd.Series(range(len(df)), index=df.index)
    strategy = GraphStrategy(strategy)

    if strategy == GraphStrategy.SAME_VINEYARD_ROW_ADJACENT:
        edges = build_same_vineyard_row_adjacent(df, node_index)
    elif strategy == GraphStrategy.SAME_BLOCK:
        edges = build_same_block(df, node_index)
    elif strategy == GraphStrategy.SIMILAR_YIELD:
        edges = build_similar_yield(df, node_index, k=k_similar)
    else:
        edges = build_hybrid(df, node_index, k=k_similar)

    if edges.size == 0:
        # self-loops fallback
        n = len(df)
        edges = np.stack([np.arange(n), np.arange(n)])
    return torch.tensor(edges, dtype=torch.long)


def build_pyg_data(
    features: np.ndarray,
    targets: np.ndarray,
    df: pd.DataFrame,
    strategy: GraphStrategy | str = GraphStrategy.HYBRID,
):
    """Build a torch_geometric.data.Data object."""
    from torch_geometric.data import Data

    edge_index = build_edge_index(df, strategy=strategy)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
    return Data(x=x, edge_index=edge_index, y=y)
