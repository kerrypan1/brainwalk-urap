"""Join raw videos to Zeno metrics for the auxiliary/contrastive branch.

Restricts to walking protocols (FW/PWS/DTW). Primary join is at the SESSION level
`(patient_id, date, protocol)`: Zeno trial metrics are averaged within a session
and attached to every video of that session. This is the right granularity for
auxiliary supervision and recovers videos whose per-trial numbering differs from
Zeno's. Strict trial-level match rate is reported as a diagnostic.

Output: artifacts/pairs.csv  (walking videos with matched session-mean Zeno metrics)
"""
from __future__ import annotations

import pandas as pd

from common import ARTIFACTS_DIR, WALKING_PROTOCOLS, ensure_artifacts

SESSION_KEYS = ["patient_id", "date", "protocol"]


def main() -> None:
    out_dir = ensure_artifacts()
    vid = pd.read_csv(ARTIFACTS_DIR / "video_index.csv")
    zen = pd.read_csv(ARTIFACTS_DIR / "zeno_metrics.csv")
    vid = vid[vid["protocol"].isin(WALKING_PROTOCOLS)].copy()

    metric_cols = [c for c in zen.columns if c not in SESSION_KEYS + ["trial"]]

    # --- Diagnostic: strict trial-level match rate ---
    strict = vid.merge(zen, on=SESSION_KEYS + ["trial"], how="left", indicator=True)
    strict_matched = strict["_merge"] == "both"

    # --- Primary: session-level mean of Zeno metrics ---
    zen_session = zen.groupby(SESSION_KEYS, as_index=False)[metric_cols].mean()
    zen_session["zeno_n_trials"] = (
        zen.groupby(SESSION_KEYS).size().reset_index(name="zeno_n_trials")["zeno_n_trials"]
    )
    merged = vid.merge(zen_session, on=SESSION_KEYS, how="left", indicator=True)
    matched = merged["_merge"] == "both"
    pairs = merged[matched].drop(columns=["_merge"])

    pairs_path = out_dir / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    print(f"[join] walking videos: {len(vid)}")
    print(f"  strict trial-level matches:  {strict_matched.sum()} ({strict_matched.mean():.1%})")
    print(f"  session-level matches:       {matched.sum()} ({matched.mean():.1%})")
    print("  session match rate by protocol:")
    by = merged.assign(matched=matched).groupby("protocol")["matched"].agg(["sum", "count", "mean"])
    print(by.to_string())

    core = [c for c in ["velocitycmsecmean", "cadencestepsminmean", "steplengthcmmean"] if c in metric_cols]
    if core:
        usable = pairs[core].notna().all(axis=1)
        print(f"  paired videos with all core metrics present {core}: {usable.sum()} ({usable.mean():.1%})")
    print(f"  unique paired patients: {pairs['patient_id'].nunique()}")
    print(f"  wrote {len(pairs)} paired video rows -> {pairs_path}")


if __name__ == "__main__":
    main()
