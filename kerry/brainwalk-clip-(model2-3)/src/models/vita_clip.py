"""Phase 8: end-to-end temporal prompt-tuned CLIP (a tractable Vita-CLIP variant).

Unlike Phases 1-7 (which classify frozen, pre-cached features), here the CLIP
image tower is part of the trainable graph via **visual prompt tuning** (VPT):
CLIP weights stay frozen, but learnable prompt tokens are injected into every ViT
layer so the per-frame representation can adapt to gait. A small **temporal
transformer** then fuses the per-frame CLS embeddings of a contiguous clip into
one video embedding, which is classified by cosine similarity to **learnable text
prototypes** (CoOp + KAPT descriptions) -- the paper's text-aligned scheme.

Trainable: visual prompts + temporal transformer + text context (tiny; CLIP frozen).
Runs on 8 GB via grad-checkpointed ViT blocks + bf16 autocast.
"""
from __future__ import annotations

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .prompt_learner import PromptLearner
from .text_encoder import CLIPTextEncoder

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_frames(x: torch.Tensor) -> torch.Tensor:
    """x: [..., 3, H, W] float in [0,1] -> CLIP-normalized."""
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


class PromptedVisual(nn.Module):
    """Frozen open_clip ViT with VPT(-deep) prompts. Returns per-image CLS embed [B, embed_dim]."""

    def __init__(self, clip_model, n_prompt: int = 8, deep: bool = True, grad_ckpt: bool = True):
        super().__init__()
        v = clip_model.visual
        self.conv1 = v.conv1
        self.class_embedding = v.class_embedding
        self.positional_embedding = v.positional_embedding
        self.patch_dropout = v.patch_dropout
        self.ln_pre = v.ln_pre
        self.resblocks = v.transformer.resblocks
        self.ln_post = v.ln_post
        self.proj = v.proj

        width = v.class_embedding.shape[0]
        n_layers = len(self.resblocks)
        self.n_prompt = n_prompt
        self.deep = deep
        self.grad_ckpt = grad_ckpt
        n_sets = n_layers if deep else 1
        self.prompts = nn.Parameter(torch.empty(n_sets, n_prompt, width))
        nn.init.normal_(self.prompts, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 3, H, W] normalized
        x = self.conv1(x)                                       # [B, width, gh, gw]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, gh*gw, width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                          # [B, 1+P, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        B = x.shape[0]
        p = self.prompts[0].to(x.dtype).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)           # [B, 1+n_prompt+P, width]

        for i, blk in enumerate(self.resblocks):
            if self.deep and i > 0:
                p = self.prompts[i].to(x.dtype).unsqueeze(0).expand(B, -1, -1)
                x = torch.cat([x[:, :1], p, x[:, 1 + self.n_prompt:]], dim=1)
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        x = self.ln_post(x[:, 0])                               # CLS token
        if self.proj is not None:
            x = x @ self.proj
        return x                                                # [B, embed_dim]


class TemporalEncoder(nn.Module):
    """Fuse [B, T, in_dim] per-frame embeddings into one [B, in_dim] video embedding."""

    def __init__(self, in_dim=512, d_model=256, nhead=4, layers=2, dropout=0.3, max_len=128):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # [B, T, in_dim]
        B, T, _ = x.shape
        h = self.in_proj(x)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos[:, : T + 1]
        h = self.enc(h)
        h = self.norm(h[:, 0])
        return self.out(self.drop(h))                           # [B, in_dim]


class VitaCLIP(nn.Module):
    """Prompt-tuned CLIP visual + temporal encoder, with a text-cosine or linear head.

    head_type="text":  cosine(video_embed, learned text prototypes) * temperature
                       (the paper's text-aligned scheme). We use a *learnable* clamped
                       temperature initialized low (1/0.07) instead of CLIP's frozen
                       logit_scale (~100), which otherwise saturates the softmax and
                       stalls training for a freshly-initialized temporal encoder.
    head_type="linear": a plain linear classifier on the video embedding (avoids the
                       cosine-saturation issue; standard end-to-end fine-tuning).
    """

    def __init__(self, classnames, model_name="ViT-B-32-quickgelu", pretrained="openai",
                 n_ctx=8, n_prompt=8, deep=True, d_model=256, nhead=4, t_layers=2,
                 dropout=0.3, grad_ckpt=True, head_type="linear"):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        for p in clip_model.parameters():
            p.requires_grad_(False)

        self.head_type = head_type
        n_cls = len(classnames)
        self.visual = PromptedVisual(clip_model, n_prompt=n_prompt, deep=deep, grad_ckpt=grad_ckpt)
        embed_dim = clip_model.visual.proj.shape[1]
        self.temporal = TemporalEncoder(embed_dim, d_model, nhead, t_layers, dropout)

        if head_type == "text":
            self.prompt_learner = PromptLearner(clip_model, tokenizer, classnames, n_ctx=n_ctx)
            self.text_encoder = CLIPTextEncoder(clip_model)
            # learnable, clamped temperature (log space), init at 1/0.07
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        elif head_type == "linear":
            self.classifier = nn.Linear(embed_dim, n_cls)
        else:
            raise ValueError(head_type)

    def text_features(self) -> torch.Tensor:
        prompts = self.prompt_learner()
        tf = self.text_encoder(prompts, self.prompt_learner.tokenized)
        return F.normalize(tf, dim=-1)

    def encode_clip(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: [B, T, 3, H, W] normalized -> video embedding [B, embed_dim]."""
        B, T = frames.shape[:2]
        feat = self.visual(frames.reshape(B * T, *frames.shape[2:]))   # [B*T, embed_dim]
        feat = feat.reshape(B, T, -1)
        return self.temporal(feat)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        vid = self.encode_clip(frames)
        if self.head_type == "linear":
            return self.classifier(vid)
        scale = self.logit_scale.clamp(max=torch.log(torch.tensor(100.0))).exp()
        return scale * F.normalize(vid, dim=-1) @ self.text_features().t()  # [B, n_cls]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
