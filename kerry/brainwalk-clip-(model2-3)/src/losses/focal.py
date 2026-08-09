"""Multi-class focal loss (paper Eq. 5: alpha=0.25, gamma=2).

Down-weights easy examples and, via per-class alpha, the majority class — the
imbalance remedy the paper uses in place of plain cross-entropy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | float = 0.25,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[target]
        else:
            at = self.alpha
        loss = -at * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
