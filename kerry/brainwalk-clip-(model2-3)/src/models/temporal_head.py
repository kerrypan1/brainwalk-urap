"""Lightweight temporal heads over frozen per-frame CLIP features [B, T, D].

Kept tiny on purpose: only 89 labeled clips, so a heavy sequence model overfits.
Attention pooling learns which frames matter; we also expose cheap time-statistics
(mean/std/max over frames) that capture gait variability with zero learned params.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def time_stats(x: torch.Tensor) -> torch.Tensor:
    """[B,T,D] -> [B,3D] = concat(mean, std, max) over time. std ~ gait variability."""
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    mx = x.amax(dim=1)
    return torch.cat([mean, std, mx], dim=-1)


class AttnPoolHead(nn.Module):
    def __init__(self, in_dim=512, n_classes=4, hidden=128, dropout=0.3):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.drop = nn.Dropout(dropout)
        # classifier sees attention-pooled features + std-over-time (variability cue)
        self.cls = nn.Linear(in_dim * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,D]
        w = self.score(x).softmax(dim=1)          # [B,T,1]
        pooled = (w * x).sum(dim=1)               # [B,D]
        std = x.std(dim=1)                        # [B,D]
        feat = self.drop(torch.cat([pooled, std], dim=-1))
        return self.cls(feat)


class TemporalTransformer(nn.Module):
    """Small Transformer encoder over frozen per-frame features (Vita-CLIP surrogate).

    Models frame-to-frame temporal structure that mean-pool discards, while keeping
    CLIP frozen and cheap (works on cached [B,T,D] windows). A learnable [SUMMARY]
    token aggregates the sequence; classification uses it plus std-over-time.
    """

    def __init__(self, in_dim=512, n_classes=4, d_model=256, nhead=4, layers=2,
                 dropout=0.3, max_len=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.summary = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.summary, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(d_model + in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,D]
        B, T, _ = x.shape
        h = self.proj(x)                                   # [B,T,d]
        s = self.summary.expand(B, -1, -1)                 # [B,1,d]
        h = torch.cat([s, h], dim=1) + self.pos[:, : T + 1]
        h = self.encoder(h)
        summary = h[:, 0]                                  # [B,d]
        std = x.std(dim=1)                                 # [B,D] gait-variability cue
        return self.cls(self.drop(torch.cat([summary, std], dim=-1)))
