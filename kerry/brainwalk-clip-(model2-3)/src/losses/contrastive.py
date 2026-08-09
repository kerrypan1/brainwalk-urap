from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_infonce(v: torch.Tensor, z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE for L2-normalized paired embeddings v, z (both [B, D])."""
    logits = v @ z.t() / temperature
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def retrieval_metrics(v: torch.Tensor, z: torch.Tensor) -> dict:
    """Video->Zeno retrieval on a held-out set (diagonal = positive pair)."""
    sim = v @ z.t()                       # [N, N]
    n = sim.shape[0]
    ranks = []
    order = sim.argsort(dim=1, descending=True)
    for i in range(n):
        rank = (order[i] == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float32)
    return {
        "n": int(n),
        "recall@1": float((ranks < 1).float().mean()),
        "recall@5": float((ranks < 5).float().mean()),
        "recall@10": float((ranks < 10).float().mean()),
        "median_rank": float(ranks.median().item()) + 1.0,
        "random_recall@1": 1.0 / n,
    }
