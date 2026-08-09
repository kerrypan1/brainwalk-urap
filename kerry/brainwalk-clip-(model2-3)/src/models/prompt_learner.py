"""CoOp-style learnable prompt learner over a frozen CLIP text tower.

Builds `[SOS] [ctx_1..ctx_n] [class description tokens] [EOS] ...` where the
context vectors are learnable and everything else is frozen. Supports unified
context (shared across classes) or class-specific context (CSC).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import CLIPTextEncoder


class PromptLearner(nn.Module):
    def __init__(self, clip_model, tokenizer, classnames, n_ctx=8, class_specific=False):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.token_embedding.weight.dtype
        d = clip_model.token_embedding.embedding_dim

        if class_specific:
            ctx = torch.empty(n_cls, n_ctx, d, dtype=dtype)
        else:
            ctx = torch.empty(n_ctx, d, dtype=dtype)
        nn.init.normal_(ctx, std=0.02)
        self.ctx = nn.Parameter(ctx)

        device = clip_model.token_embedding.weight.device
        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized = torch.cat([tokenizer(p) for p in prompts]).to(device)  # [n_cls, L]

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)  # [n_cls, L, d]

        # sanity: EOT must sit after SOS + n_ctx placeholders (i.e. class tokens exist)
        eot = tokenized.argmax(dim=-1)
        assert int(eot.min()) > 1 + n_ctx, "class tokens missing; check n_ctx/placeholder tokenization"

        self.register_buffer("token_prefix", embedding[:, :1, :])            # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])    # class + EOS + pad
        self.register_buffer("tokenized", tokenized)
        self.n_ctx = n_ctx
        self.n_cls = n_cls

    def forward(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)  # [n_cls, L, d]


class CoOpFGA(nn.Module):
    """Classify pre-computed (frozen) image features by similarity to learned prompts."""

    def __init__(self, clip_model, tokenizer, classnames, n_ctx=8, class_specific=False):
        super().__init__()
        self.prompt_learner = PromptLearner(clip_model, tokenizer, classnames, n_ctx, class_specific)
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.register_buffer("logit_scale", clip_model.logit_scale.exp().detach())

    def text_features(self) -> torch.Tensor:
        prompts = self.prompt_learner()
        tf = self.text_encoder(prompts, self.prompt_learner.tokenized)
        return F.normalize(tf, dim=-1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(image_features, dim=-1)
        return self.logit_scale * image_features @ self.text_features().t()
