"""Faithful Numerical Text Embedding (NTE), arXiv:2403.13756 Sec 2.3.

Encodes a sentence of 4 gait parameters, e.g.
    "the mean stride length (cm) is [NUM] and the cadence (steps/min) is [NUM]
     and the mean stance time (s) is [NUM] and the asymmetry index of step
     length (cm) is [NUM]."
where each [NUM] slot's token embedding is **replaced** by `value * e_NUM`, a
single learnable vector shared across all numeric slots (the paper's
"dedicated embedding base ... orthogonal to the position encoding"; we do not
enforce hard orthogonality, just let it learn under the alignment loss, same
as most re-implementations of this style of numeric embedding).

F^num = FCLIP_T([frag_0, [NUM]*v0, frag_1, [NUM]*v1, ...])   (paper Eq. 6)

Cross-modal alignment (Sec 2.3): small projection heads map F^num and each
class's text prototype F^T_i into a shared space; a cross-entropy loss trains
F^num to be closest (by cosine) to the prototype of its own class:
    L_gp = CE(cosine(P^num, {P^T_i}) / tau_gp, class_label)
Total loss (paper): L = L_k + omega * L_gp, omega = 0.05.

Deviations from the paper (documented, forced by our data):
  - Paper's 29 GAITRite parameters -> our 83 Zeno-Walkway metric columns; we
    generate natural-language descriptions programmatically (base-name +
    stat-suffix dictionary), not from a clinician-reviewed table.
  - We select up to `max_combos` random 4-param combinations with all pairwise
    |Pearson r| < 0.4 (paper: 438 combos out of C(29,4)); combos are chosen
    once from the full corpus (not label-dependent, so no CV leakage risk).
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import CLIPTextEncoder

CONTEXT_LENGTH = 77

# longest-prefix-match dictionaries built from our Zeno column-naming convention
_BASE_UNITS = sorted([
    ("absolutesteplengthcm", "absolute step length (cm)"),
    ("ambulationtimesec", "ambulation time (s)"),
    ("cadencestepsmin", "cadence (steps per minute)"),
    ("singlesupportsec", "single support time (s)"),
    ("singlesupport", "single support (percent of cycle)"),
    ("stancetimesec", "stance time (s)"),
    ("steplengthcm", "step length (cm)"),
    ("steptimesec", "step time (s)"),
    ("stridelengthcm", "stride length (cm)"),
    ("stridetimesec", "stride time (s)"),
    ("stridevelocitycmsec", "stride velocity (cm per s)"),
    ("swingtimesec", "swing time (s)"),
    ("velocitycmsec", "velocity (cm per s)"),
], key=lambda kv: -len(kv[0]))

_SUFFIXES = sorted([
    ("cvleft", "coefficient of variation, left"),
    ("cvright", "coefficient of variation, right"),
    ("meanleft", "mean, left"),
    ("meanright", "mean, right"),
    ("ratiolr", "left-right ratio"),
    ("asi", "asymmetry index"),
    ("cv", "coefficient of variation"),
    ("mean", "mean"),
], key=lambda kv: -len(kv[0]))


def describe_metric(col: str) -> str:
    """Heuristic natural-language phrase for a Zeno metric column name."""
    for base, base_phrase in _BASE_UNITS:
        if col.startswith(base):
            rest = col[len(base):]
            for suf, suf_phrase in _SUFFIXES:
                if rest == suf:
                    if suf == "mean":
                        return f"mean {base_phrase}"
                    return f"{suf_phrase} of {base_phrase}"
            if rest == "":
                return base_phrase
    # fallback: split camel-ish name on digit/unit boundaries
    words = re.findall(r"[a-z]+", col)
    return " ".join(words) if words else col


def select_low_corr_combos(pairs_df, cols, k=4, thresh=0.4, max_combos=200, seed=0,
                           max_missing_frac=0.5):
    """Random search for k-column combos with all pairwise |corr| < thresh.

    Correlation computed on the whole corpus (label-independent structural
    choice, mirrors the paper's global 438-combo vocabulary; not CV leakage).
    """
    usable = [c for c in cols if pairs_df[c].isna().mean() <= max_missing_frac]
    corr = pairs_df[usable].corr().to_numpy()
    corr = np.nan_to_num(corr, nan=1.0)  # treat undefined corr as "correlated" -> excluded
    ok = np.abs(corr) < thresh
    np.fill_diagonal(ok, True)

    rng = np.random.default_rng(seed)
    n = len(usable)
    combos = set()
    attempts = 0
    while len(combos) < max_combos and attempts < max_combos * 200:
        attempts += 1
        idx = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
        if idx in combos:
            continue
        sub = ok[np.ix_(idx, idx)]
        if sub.all():
            combos.add(idx)
    return [[usable[i] for i in idx] for idx in combos]


class NumericTextEncoder(nn.Module):
    """Encodes 4-parameter numeric sentences and aligns them to class-text prototypes."""

    def __init__(self, clip_model, tokenizer, proj_dim=128, tau_gp=0.01):
        super().__init__()
        self.token_embedding = clip_model.token_embedding  # frozen
        self.text_encoder = CLIPTextEncoder(clip_model)     # frozen (shared with class prototypes)
        self.tokenizer = tokenizer
        self.tau_gp = tau_gp

        d = clip_model.token_embedding.embedding_dim
        out_dim = clip_model.text_projection.shape[1]
        self.num_embed = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.num_embed, std=0.02)
        self.proj_num = nn.Linear(out_dim, proj_dim)
        self.proj_txt = nn.Linear(out_dim, proj_dim)

    def _build_ids(self, descs):
        """descs: list of k description strings -> (ids[L], slot_positions[k])."""
        ids = [self.tokenizer.sot_token_id]
        slots = []
        for i, d in enumerate(descs):
            frag = f"the {d} is" if i == 0 else f" and the {d} is"
            ids += self.tokenizer.encode(frag)
            slots.append(len(ids))       # index of the (placeholder) NUM token
            ids += [0]                    # placeholder, embedding overwritten later
        ids += self.tokenizer.encode(".")
        ids += [self.tokenizer.eot_token_id]
        if len(ids) > CONTEXT_LENGTH:
            raise ValueError(f"NTE sentence too long ({len(ids)} tokens): {descs}")
        ids += [0] * (CONTEXT_LENGTH - len(ids))
        return ids, slots

    def forward(self, desc_batch, values: torch.Tensor) -> torch.Tensor:
        """desc_batch: list of B lists of k descriptions. values: [B, k]. -> F^num [B, D]."""
        device = values.device
        dtype = self.num_embed.dtype
        B = len(desc_batch)
        all_ids, all_slots = [], []
        for descs in desc_batch:
            ids, slots = self._build_ids(descs)
            all_ids.append(ids)
            all_slots.append(slots)
        ids_t = torch.tensor(all_ids, device=device, dtype=torch.long)     # [B, L]
        emb = self.token_embedding(ids_t).to(dtype)                        # [B, L, D]
        for b, slots in enumerate(all_slots):
            for j, pos in enumerate(slots):
                emb[b, pos] = values[b, j].to(dtype) * self.num_embed
        return self.text_encoder(emb, ids_t)                               # [B, D_out]

    def align_loss(self, f_num: torch.Tensor, text_feats: torch.Tensor,
                   class_labels: torch.Tensor) -> torch.Tensor:
        """text_feats: [n_cls, D_out] (already L2-normalized class prototypes)."""
        p_num = F.normalize(self.proj_num(f_num), dim=-1)                  # [B, proj_dim]
        p_txt = F.normalize(self.proj_txt(text_feats), dim=-1)             # [n_cls, proj_dim]
        logits = (p_num @ p_txt.t()) / self.tau_gp                          # [B, n_cls]
        return F.cross_entropy(logits, class_labels)
