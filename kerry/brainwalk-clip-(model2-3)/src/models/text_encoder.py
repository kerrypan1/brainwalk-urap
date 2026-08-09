"""Frozen CLIP text tower that encodes pre-embedded prompt sequences.

Mirrors open_clip's classic CLIP text forward but takes token *embeddings*
(so learnable context vectors can be spliced in) instead of token ids.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer("attn_mask", clip_model.attn_mask, persistent=False)

    def forward(self, prompt_embeds: torch.Tensor, tokenized: torch.Tensor) -> torch.Tensor:
        # prompt_embeds: [n_cls, L, d] (learnable ctx already spliced in).
        # open_clip's text transformer is batch_first, so no NLD<->LND permute.
        x = prompt_embeds + self.positional_embedding.to(prompt_embeds.dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        # EOT pooling: EOT token is the highest id in the CLIP vocab -> argmax
        eot = tokenized.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eot]
        x = x @ self.text_projection
        return x
