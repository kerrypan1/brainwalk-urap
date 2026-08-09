"""Ordinal classifiers for small-n gait scoring (§26.2).

CORAL (Consistent Rank Logits): for K ordered classes, fit K-1 binary models
P(y > k | x), k = 0..K-2, and predict class = sum_k 1[P(y>k) > 0.5].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def coral_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Out-of-fold CORAL predictions. X: [N, D], y integer in 0..K-1."""
    y = np.asarray(y, dtype=int)
    n = len(y)
    K = int(y.max()) + 1
    if K < 3:
        raise ValueError("CORAL needs at least 3 ordinal classes")

    oof = np.full(n, -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        probas = []
        for k in range(K - 1):
            y_bin = (y[tr] > k).astype(int)
            if y_bin.min() == y_bin.max():
                # degenerate threshold on this fold — fall back to constant
                probas.append(np.full(te.sum(), float(y_bin[0])))
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
            clf.fit(Xtr, y_bin)
            probas.append(clf.predict_proba(Xte)[:, 1])
        probas = np.stack(probas, axis=1)          # [n_te, K-1]
        oof[te] = (probas > 0.5).sum(axis=1).astype(int)
    return oof


def coral_oof_continuous(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray, C: float = 1.0
) -> np.ndarray:
    """Out-of-fold CORAL expected scores before rounding.

    For an ordinal target encoded by cumulative thresholds,
    ``E[y] = sum_k P(y > k)``.
    """
    y = np.asarray(y, dtype=int)
    K = int(y.max()) + 1
    if K < 3:
        raise ValueError("CORAL needs at least 3 ordinal classes")

    oof = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        probas = []
        for k in range(K - 1):
            y_bin = (y[tr] > k).astype(int)
            if y_bin.min() == y_bin.max():
                probas.append(np.full(te.sum(), float(y_bin[0])))
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
            clf.fit(Xtr, y_bin)
            probas.append(clf.predict_proba(Xte)[:, 1])
        oof[te] = np.stack(probas, axis=1).sum(axis=1)
    return oof


def regress_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold continuous Ridge predictions without clipping or rounding."""
    from sklearn.linear_model import Ridge

    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan, dtype=float)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=1.0)
        reg.fit(sc.transform(X[tr]), y[tr])
        oof[te] = reg.predict(sc.transform(X[te]))
    return oof


def regress_round_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold linear regression + round + clip to observed class range."""
    from sklearn.linear_model import Ridge

    y = y.astype(np.float64)
    y_min, y_max = int(y.min()), int(y.max())
    oof = np.full(len(y), -1, dtype=int)
    for fold in sorted(pd.unique(folds)):
        te = folds == fold
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=1.0)
        reg.fit(sc.transform(X[tr]), y[tr])
        pred = reg.predict(sc.transform(X[te]))
        oof[te] = np.clip(np.rint(pred), y_min, y_max).astype(int)
    return oof
