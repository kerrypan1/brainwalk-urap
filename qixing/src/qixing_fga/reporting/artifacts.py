"""Reproducibility artifacts: run directories and self-describing outputs.

Every run writes to ``experiments/<experiment_name>/<run_id>/`` where
``run_id = YYYYMMDD_HHMMSS_<short_git_hash>``. Artifacts: config.yaml,
metrics.json, predictions.csv, feature_importance.csv, provenance.json,
environment.txt. Rule: a run is incomplete if a metric cannot be traced to
config + seed + commit + dataset checksum.

Also includes legacy CSV/JSON dumps from the batch training CLI
(``save_results_to_dir``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from ..features.columns import aggregate_top_features_by_frequency


def _progress(msg: str) -> None:
    print(msg, flush=True)


def git_commit_short(cwd: Optional[str] = None) -> str:
    """Short git commit hash, or 'nogit' when unavailable (e.g. no commits yet)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "nogit"


def file_sha256(path: str) -> str:
    """Streamed SHA-256 of a file, or 'missing' if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_dir(
    experiment_name: str,
    base: str = "experiments",
    *,
    repo_dir: Optional[str] = None,
) -> Path:
    """Create and return experiments/<name>/<YYYYMMDD_HHMMSS>_<git>/ ."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{git_commit_short(repo_dir)}"
    run_dir = Path(base) / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: dict) -> None:
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
    )


def write_predictions(run_dir: Path, predictions: Optional[pd.DataFrame]) -> None:
    if predictions is not None and not predictions.empty:
        predictions.to_csv(run_dir / "predictions.csv", index=False)


def write_feature_importance(
    run_dir: Path, importance: Optional[pd.DataFrame]
) -> None:
    if importance is not None and not importance.empty:
        importance.to_csv(run_dir / "feature_importance.csv", index=False)


def write_provenance(
    run_dir: Path,
    *,
    random_seed: int,
    dataset_paths: list[str],
    repo_dir: Optional[str] = None,
) -> None:
    payload = {
        "git_commit": git_commit_short(repo_dir),
        "python_version": sys.version.split()[0],
        "random_seed": random_seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {
            path: file_sha256(path) for path in dataset_paths if path
        },
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_environment(run_dir: Path) -> None:
    """Snapshot the current interpreter's installed packages (pip freeze)."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        content = out.stdout if out.returncode == 0 else out.stderr
    except Exception as exc:  # pragma: no cover
        content = f"pip freeze failed: {exc}\n"
    (run_dir / "environment.txt").write_text(content, encoding="utf-8")


def _json_default(obj):
    """Fallback JSON encoder for numpy/pandas scalars."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def save_results_to_dir(
    output_dir: Path,
    *,
    results: Optional[pd.DataFrame] = None,
    summary: Optional[pd.DataFrame] = None,
    inner_rfecv_ranking: Optional[pd.DataFrame] = None,
    fold_selected_features: Optional[dict[int, list[str]]] = None,
    top_k_by_frequency: int = 20,
    nested_compare: Optional[pd.DataFrame] = None,
    predictions: Optional[pd.DataFrame] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictions is not None and not predictions.empty:
        predictions.to_csv(output_dir / "predictions.csv", index=False)
    if results is not None:
        results.to_csv(output_dir / "cv_results_nested.csv", index=False)
    if summary is not None:
        summary.to_csv(output_dir / "nested_summary.csv", index=False)
    if inner_rfecv_ranking is not None:
        inner_rfecv_ranking.to_csv(
            output_dir / "inner_rfecv_ranking_by_fold.csv", index=False
        )
    if nested_compare is not None:
        nested_compare.to_csv(output_dir / "nested_compare.csv", index=False)
    if fold_selected_features is not None:
        payload = {
            f"fold_{k}": v for k, v in fold_selected_features.items()
        }
        (output_dir / "fold_selected_features.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        top_features, top_freqs = aggregate_top_features_by_frequency(
            fold_selected_features, top_k=top_k_by_frequency
        )
        n_folds = len(fold_selected_features)
        top_payload = {
            "method": "frequency_top_k",
            "k": top_k_by_frequency,
            "n_outer_folds": n_folds,
            "selected_features": top_features,
            "feature_frequencies": {
                name: f"{freq}/{n_folds}" for name, freq in top_freqs.items()
            },
            "feature_frequency_counts": top_freqs,
        }
        top_path = output_dir / f"selected_features_top{top_k_by_frequency}_by_frequency.json"
        top_path.write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
        _progress(
            f"[Save] Top-{top_k_by_frequency} by fold frequency → {top_path.name}"
        )

    _progress(f"\n[Save] Results written to {output_dir.resolve()}")
