"""Run-directory artifact writers produce the full publication set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qixing_fga.reporting.artifacts import (
    make_run_dir,
    write_config,
    write_environment,
    write_feature_importance,
    write_metrics,
    write_predictions,
    write_provenance,
)


def test_run_dir_has_full_artifact_set(tmp_path: Path):
    run_dir = make_run_dir("unit_test", base=str(tmp_path), repo_dir=str(ROOT))
    assert run_dir.is_dir()

    cfg = {"experiment_name": "unit_test", "random_seed": 42, "task": "regression"}
    write_config(run_dir, cfg)
    write_metrics(run_dir, {"summary": [{"model": "ridge", "mae_mean": 0.5}]})
    write_predictions(
        run_dir,
        pd.DataFrame(
            [
                {
                    "model": "ridge",
                    "fold": 1,
                    "feature_variant": "full",
                    "sample_id": "s1",
                    "participant_id": "p1",
                    "y_true": 1.0,
                    "y_pred": 1.2,
                }
            ]
        ),
    )
    write_feature_importance(
        run_dir, pd.DataFrame([{"feature": "f0", "n_folds_selected": 3, "frequency": 0.6}])
    )
    # Use a tiny existing file for checksum.
    data_file = ROOT / "configs" / "fga_baseline.yaml"
    write_provenance(
        run_dir,
        random_seed=42,
        dataset_paths=[str(data_file)],
        repo_dir=str(ROOT),
    )
    write_environment(run_dir)

    required = [
        "config.yaml",
        "metrics.json",
        "predictions.csv",
        "feature_importance.csv",
        "provenance.json",
        "environment.txt",
    ]
    for name in required:
        path = run_dir / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"

    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["random_seed"] == 42
    assert "git_commit" in provenance
    assert str(data_file) in provenance["datasets"]
