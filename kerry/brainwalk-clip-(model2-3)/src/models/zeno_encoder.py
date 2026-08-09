"""Zeno metric MLP encoder and video projection head for the shared embedding space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZenoEncoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 512, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class VideoProjection(nn.Module):
    """Small projection on top of frozen CLIP video features."""

    def __init__(self, in_dim: int = 512, embed_dim: int = 512, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(v), dim=-1)
