"""Faithful reproduction of the video model in arXiv:2403.13756 (gait scoring).

Video encoder = frozen CLIP ViT + **Vita-CLIP video prompt learner** (Wasim et al.,
arXiv:2304.03307), Eqs (5)-(8). At every layer, for a clip of T frames:
  * Summary token S^(l): project each frame's CLS token (P_sum), run a trainable
    cross-frame MHSA over the T projected CLS tokens (message passing between frames),
    residual -> one summary token per frame. (Eq 5)
  * Global prompts G^(l): M_v trainable video-level tokens (shared across frames). 
  * Local prompts L^(l): T trainable tokens, each conditioned on its frame's CLS:
    l_hat_t = l_t + z_{t,0}. (Eq 6)
  * Append [S, G, L] to every frame's token sequence, apply the FROZEN pretrained
    self-attention (Eq 7), drop the appended tokens, apply the FROZEN FFN on the frame
    tokens only (Eq 8).
Per-frame CLS of the last layer is projected and **average-pooled over frames** to get
the video feature F^V. Classification = cosine(F^V, text prototype)/tau, focal loss.

Everything in CLIP (conv/attn/mlp/ln/proj) is frozen; only VPL modules + text context
learn. bf16 autocast + grad checkpointing to fit 8 GB.
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
    mean = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


class VitaVPLVisual(nn.Module):
    """Frozen open_clip ViT with the Vita-CLIP video prompt learner injected per layer."""

    def __init__(self, clip_model, n_frames: int, n_global: int = 8, n_head_sum: int = 8,
                 max_frames: int = 70, grad_ckpt: bool = True):
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

        D = v.class_embedding.shape[0]
        n_layers = len(self.resblocks)
        self.D = D
        self.T = n_frames
        self.n_global = n_global
        self.grad_ckpt = grad_ckpt

        # temporal positional embedding e^tm (spatial e^sp already added in _embeds)
        self.temporal_embedding = nn.Parameter(torch.zeros(max_frames, D))
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

        # per-layer VPL modules (trainable; CLIP frozen)
        self.p_sum = nn.ModuleList([nn.Linear(D, D) for _ in range(n_layers)])
        self.ln_sum = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_layers)])
        self.attn_sum = nn.ModuleList(
            [nn.MultiheadAttention(D, n_head_sum, batch_first=True) for _ in range(n_layers)])
        self.global_prompt = nn.ParameterList(
            [nn.Parameter(torch.empty(n_global, D)) for _ in range(n_layers)])
        self.local_prompt = nn.ParameterList(
            [nn.Parameter(torch.empty(max_frames, D)) for _ in range(n_layers)])
        for g in self.global_prompt:
            nn.init.normal_(g, std=0.02)
        for l in self.local_prompt:
            nn.init.normal_(l, std=0.02)

    def _embeds(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)      # [BT, N, D]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                                  # [BT, 1+N, D]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        return x

    def _layer(self, i, z):
        """z: [B, T, P, D] (P = 1+N frame tokens). Returns next-layer [B, T, P, D]."""
        B, T, P, D = z.shape
        blk = self.resblocks[i]
        Z0 = z[:, :, 0, :]                                              # [B, T, D] CLS per frame
        Z0p = self.p_sum[i](Z0)                                         # [B, T, D]
        S = self.attn_sum[i](self.ln_sum[i](Z0p), self.ln_sum[i](Z0p),
                             self.ln_sum[i](Z0p), need_weights=False)[0] + Z0p   # [B, T, D] (Eq 5)
        G = self.global_prompt[i].to(z.dtype).unsqueeze(0).expand(B, -1, -1)     # [B, Mv, D]
        L = self.local_prompt[i][:T].to(z.dtype).unsqueeze(0) + Z0               # [B, T, D] (Eq 6)
        appended = torch.cat([S, G, L], dim=1)                          # [B, T+Mv+T, D]
        A = appended.shape[1]

        # append the SAME [S,G,L] set to every frame's sequence
        app = appended.unsqueeze(1).expand(B, T, A, D)                  # [B, T, A, D]
        seq = torch.cat([z, app], dim=2).reshape(B * T, P + A, D)       # [BT, P+A, D]

        # frozen attention sub-block over the full sequence (Eq 7)
        h = blk.ln_1(seq)
        seq = seq + blk.ls_1(blk.attention(q_x=h))
        # keep only frame tokens, frozen FFN (Eq 8)
        zf = seq[:, :P, :]
        zf = zf + blk.ls_2(blk.mlp(blk.ln_2(zf)))
        return zf.reshape(B, T, P, D)

    def forward(self, frames):                                         # frames: [B, T, 3, H, W]
        B, T = frames.shape[:2]
        z = self._embeds(frames.reshape(B * T, *frames.shape[2:]))     # [BT, P, D]
        P = z.shape[1]
        z = z.reshape(B, T, P, self.D)
        z = z + self.temporal_embedding[:T].to(z.dtype).view(1, T, 1, self.D)
        for i in range(len(self.resblocks)):
            if self.grad_ckpt and self.training:
                z = checkpoint(self._layer, i, z, use_reentrant=False)
            else:
                z = self._layer(i, z)
        cls = self.ln_post(z[:, :, 0, :])                              # [B, T, D]
        if self.proj is not None:
            cls = cls @ self.proj                                      # [B, T, D']
        v = cls.mean(dim=1)                                            # average-pool over frames
        return v                                                       # [B, D'] video feature F^V


class VitaCLIPFaithful(nn.Module):
    def __init__(self, classnames, n_frames=70, model_name="ViT-B-32-quickgelu",
                 pretrained="openai", n_ctx=8, n_global=8, tau=0.01, grad_ckpt=True):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tokenizer = open_clip.get_tokenizer(model_name)
        for p in clip_model.parameters():
            p.requires_grad_(False)

        self.visual = VitaVPLVisual(clip_model, n_frames=n_frames, n_global=n_global,
                                    max_frames=max(70, n_frames), grad_ckpt=grad_ckpt)
        self.prompt_learner = PromptLearner(clip_model, tokenizer, classnames, n_ctx=n_ctx)
        self.text_encoder = CLIPTextEncoder(clip_model)
        self.tau = tau
        # kept for auxiliary modules (e.g. NTE) that need the same frozen text tower
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def text_features(self):
        prompts = self.prompt_learner()
        tf = self.text_encoder(prompts, self.prompt_learner.tokenized)
        return F.normalize(tf, dim=-1)

    def forward(self, frames):
        v = F.normalize(self.visual(frames), dim=-1)                   # [B, D']
        return (v @ self.text_features().t()) / self.tau               # cosine / tau  (Eq 5 logits)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
