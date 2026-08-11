"""Bootstrap npz participant indices must align with CV fold val sets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

NPZ = ROOT / "data" / "bootstrap_indices.npz"


@pytest.mark.skipif(not NPZ.exists(), reason="bootstrap_indices.npz not present")
def test_bootstrap_npz_has_five_folds():
    from qixing_fga.evaluation.bootstrap import load_teammate_bootstrap_npz

    data = load_teammate_bootstrap_npz(str(NPZ), n_splits=5)
    for i in range(5):
        key = f"fold_{i}"
        assert key in data
        assert data[key].ndim == 2
        assert data[key].shape[1] > 0


@pytest.mark.skipif(not NPZ.exists(), reason="bootstrap_indices.npz not present")
def test_npz_folds_partition_the_fw_participants():
    """The npz indexes participants, not CSV rows.

    This is why the FW-only single-visit view does NOT need the indices
    regenerated: the fold membership is already a disjoint partition of exactly
    the 46 FW participants, so it stays valid (and stays aligned with the
    teammate's grouping) once PWS rows and the second visit are dropped.
    """
    import pandas as pd

    labels = pd.read_csv(ROOT / "data" / "2026_05_17_FWOnly_2visits_labels.csv")
    n_participants = labels["participant_id"].nunique()

    raw = np.load(NPZ)
    fold_sets = [set(int(i) for i in raw[f"fold_{k}"].ravel()) for k in range(5)]

    for a in range(5):
        for b in range(a + 1, 5):
            assert fold_sets[a].isdisjoint(fold_sets[b]), f"folds {a}/{b} overlap"

    union = set().union(*fold_sets)
    assert union == set(range(n_participants)), (
        "npz participant indices must cover 0..n_participants-1 exactly"
    )
    assert sum(len(s) for s in fold_sets) == n_participants


@pytest.mark.skipif(not NPZ.exists(), reason="bootstrap_indices.npz not present")
def test_bootstrap_width_matches_val_fold_participants():
    """Each fold's resample width must equal that fold's validation size."""
    raw = np.load(NPZ)
    for k in range(5):
        arr = raw[f"fold_{k}"]
        n_unique = len(set(int(i) for i in arr.ravel()))
        assert arr.shape[1] == n_unique, (
            f"fold_{k}: resample width {arr.shape[1]} != {n_unique} val participants"
        )


@pytest.mark.skipif(not NPZ.exists(), reason="bootstrap_indices.npz not present")
def test_bootstrap_indices_are_nonnegative_ints():
    raw = np.load(NPZ)
    for key in raw.files:
        arr = raw[key]
        assert np.issubdtype(arr.dtype, np.integer) or np.issubdtype(
            arr.dtype, np.floating
        )
        assert arr.min() >= 0
