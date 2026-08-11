"""
Per-fold validation bootstrap using precomputed participant-level resamples.

Teammate ``bootstrap_indices.npz`` format (StratifiedGroupKFold, seed 42):
  - Keys ``fold_0`` … ``fold_{K-1}``, each array shaped ``(n_bootstrap, n_participants_in_val_fold)``.
  - Entries are **participant indices** in ``sorted(np.unique(participant_id))`` order (0 … n_participants-1),
    not row ilocs into the feature matrix.
    - Each bootstrap draw resamples ``n_participants_in_val_fold`` participants with replacement; we then
        expand each drawn participant to all validation rows belonging to that participant (e.g. FW + PWS).
        If validation rows per participant are not fixed, expanded bootstrap sample sizes can vary by draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


PathLike = Union[str, Path]


def sorted_participant_ids(groups: np.ndarray) -> list:
    """Lexicographically sorted unique participant ids (stable teammate convention)."""
    return sorted(np.unique(np.asarray(groups)))


def per_fold_bootstrap_mae_rmse(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    val_idx: np.ndarray,
    groups: np.ndarray,
    boot_matrix: np.ndarray,
    *,
    ci_alpha: float = 0.05,
    strict_fold_size: bool = False,
) -> Dict[str, Any]:
    """
    MAE / RMSE bootstrap distribution on one validation fold using precomputed participant draws.

    Parameters
    ----------
    y_val, y_pred
        Validation targets and predictions **in fold order** (length ``len(val_idx)``).
    val_idx
        Global row indices for this fold's validation set (same convention as sklearn ``split``).
    groups
        Full-length ``participant_id`` per row (same object as passed to CV).
    boot_matrix
        Shape ``(n_replicates, n_participants_in_val)``; participant indices as documented above.
    strict_fold_size
        When True, require every expanded bootstrap draw to have exactly ``len(val_idx)`` rows.
        When False (default), allow variable draw sizes (robust to non-fixed rows per participant).
    """
    y_val = np.asarray(y_val, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_val) != len(y_pred):
        raise ValueError("y_val and y_pred must match in length")
    val_idx = np.asarray(val_idx, dtype=int)
    n_val = int(len(val_idx))
    n_part_val = len(np.unique(groups[val_idx]))
    if boot_matrix.ndim != 2:
        raise ValueError(f"boot_matrix must be 2D, got shape {boot_matrix.shape}")
    if boot_matrix.shape[1] != n_part_val:
        raise ValueError(
            f"Bootstrap width {boot_matrix.shape[1]} != # unique participants in val ({n_part_val}). "
            "Check CV matches the npz (StratifiedGroupKFold, same data rows)."
        )
    sorted_pids = sorted_participant_ids(groups)
    pid_from_boot = {i: p for i, p in enumerate(sorted_pids)}
    val_groups = groups[val_idx]
    boot_to_locals: Dict[int, np.ndarray] = {}
    for b in np.unique(boot_matrix):
        bi = int(b)
        pid = pid_from_boot[bi]
        locals_idx = np.flatnonzero(val_groups == pid)
        if locals_idx.size == 0:
            raise ValueError(
                f"Participant index {bi} ({pid!r}) has no rows in this val fold; npz/CV mismatch."
            )
        boot_to_locals[bi] = locals_idx.astype(np.int64, copy=False)

    n_boot = int(boot_matrix.shape[0])
    mae_boot = np.empty(n_boot, dtype=float)
    rmse_boot = np.empty(n_boot, dtype=float)
    mse_boot = np.empty(n_boot, dtype=float)
    boot_sample_sizes = np.empty(n_boot, dtype=np.int64)

    for r in range(n_boot):
        row = boot_matrix[r]
        parts = [boot_to_locals[int(b)] for b in row]
        locs = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if locs.size == 0:
            raise ValueError("Expanded bootstrap sample has zero rows; cannot compute metrics.")
        if strict_fold_size and locs.size != n_val:
            raise ValueError(
                f"Expanded bootstrap sample has {locs.size} rows, expected {n_val} (val fold size). "
                "Data may not be multiview-fixed K rows/participant."
            )
        boot_sample_sizes[r] = int(locs.size)
        yt = y_val[locs]
        yp = y_pred[locs]
        mae_boot[r] = float(mean_absolute_error(yt, yp))
        mse_boot[r] = float(mean_squared_error(yt, yp))
        rmse_boot[r] = float(np.sqrt(mse_boot[r]))

    lo = 100.0 * (ci_alpha / 2.0)
    hi = 100.0 * (1.0 - ci_alpha / 2.0)
    mse_point = float(mean_squared_error(y_val, y_pred))
    return {
        "n_bootstrap": n_boot,
        "n_val_samples": n_val,
        "n_val_participants": n_part_val,
        "strict_fold_size": bool(strict_fold_size),
        "bootstrap_sample_size_min": int(np.min(boot_sample_sizes)),
        "bootstrap_sample_size_max": int(np.max(boot_sample_sizes)),
        "bootstrap_sample_size_mean": float(np.mean(boot_sample_sizes)),
        "bootstrap_sample_size_matches_val_all": bool(np.all(boot_sample_sizes == n_val)),
        "ci_alpha": float(ci_alpha),
        "mae_point_on_fold": float(mean_absolute_error(y_val, y_pred)),
        "mse_point_on_fold": mse_point,
        "rmse_point_on_fold": float(np.sqrt(mse_point)),
        "mae_boot_mean": float(np.mean(mae_boot)),
        "mae_boot_std": float(np.std(mae_boot)),
        "mae_ci_lower": float(np.percentile(mae_boot, lo)),
        "mae_ci_upper": float(np.percentile(mae_boot, hi)),
        "mse_boot_mean": float(np.mean(mse_boot)),
        "mse_boot_std": float(np.std(mse_boot)),
        "mse_ci_lower": float(np.percentile(mse_boot, lo)),
        "mse_ci_upper": float(np.percentile(mse_boot, hi)),
        "rmse_boot_mean": float(np.mean(rmse_boot)),
        "rmse_boot_std": float(np.std(rmse_boot)),
        "rmse_ci_lower": float(np.percentile(rmse_boot, lo)),
        "rmse_ci_upper": float(np.percentile(rmse_boot, hi)),
    }


def load_teammate_bootstrap_npz(
    path: PathLike,
    *,
    n_splits: int = 5,
) -> Dict[str, np.ndarray]:
    """Load ``.npz`` and validate expected ``fold_*`` keys."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Bootstrap npz not found: {p}")
    z = np.load(p)
    expected = [f"fold_{k}" for k in range(n_splits)]
    missing = [k for k in expected if k not in z.files]
    if missing:
        raise KeyError(f"{p} missing keys {missing}; have {list(z.files)}")
    return {k: np.asarray(z[k]) for k in expected}


def maybe_load_per_fold_bootstrap(
    path: Optional[PathLike],
    *,
    n_splits: int = 5,
) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    return load_teammate_bootstrap_npz(path, n_splits=n_splits)
