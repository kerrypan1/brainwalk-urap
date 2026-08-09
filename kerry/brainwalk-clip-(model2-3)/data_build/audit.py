"""Phase-0 data audit. Summarizes all built artifacts and coverage, and writes
artifacts/audit_summary.md. Run after the four builders.
"""
from __future__ import annotations

import io

import pandas as pd

from common import ARTIFACTS_DIR, SPLIT_CSV, BATH_FW_DIR, FGA_FIELDS, participant_id


def _load():
    vid = pd.read_csv(ARTIFACTS_DIR / "video_index.csv")
    fga = pd.read_csv(ARTIFACTS_DIR / "fga_labels.csv")
    zen = pd.read_csv(ARTIFACTS_DIR / "zeno_metrics.csv")
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs.csv")
    split = pd.read_csv(SPLIT_CSV)
    return vid, fga, zen, pairs, split


def main() -> None:
    vid, fga, zen, pairs, split = _load()
    buf = io.StringIO()

    def p(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        buf.write(line + "\n")

    p("# BrainWalk Models 2–3 Phase-0 Data Audit\n")

    p("## Raw video archive")
    p(f"- total videos: {len(vid)}")
    p(f"- patients: {vid['patient_id'].nunique()}, dates: {vid['date'].nunique()}")
    p("- by protocol:")
    for proto, n in vid["protocol"].value_counts().items():
        p(f"    {proto}: {n}")

    p("\n## Auxiliary corpus (video<->Zeno session pairs, walking protocols)")
    p(f"- paired videos: {len(pairs)}  |  patients: {pairs['patient_id'].nunique()}")
    for proto, n in pairs["protocol"].value_counts().items():
        p(f"    {proto}: {n}")

    p("\n## Zeno metrics")
    p(f"- trials: {len(zen)}  |  curated gait metrics: {len([c for c in zen.columns if c not in ['patient_id','date','protocol','trial']])}")

    p("\n## FGA labels (supervised target, FW-only)")
    p(f"- labeled (patient,visit) rows: {len(fga)}  |  patients: {fga['patient_id'].nunique()}")
    p("- FGA class distribution:")
    for k, v in fga["fga_score"].value_counts().sort_index().items():
        p(f"    class {k}: {v}")

    # labeled visit -> FW clip availability (bath_fw stem = {PPPP}_{visit})
    fw_stems = {p_.stem for p_ in BATH_FW_DIR.glob("*.mp4")}
    def stem(r):
        return f"{int(r['patient_id'].split('-')[1]):04d}_{int(r['visit_index'])}"
    fga = fga.assign(fw_stem=fga.apply(stem, axis=1))
    have_clip = fga["fw_stem"].isin(fw_stems)
    p(f"- labeled visits with a matching bath_fw clip: {have_clip.sum()}/{len(fga)}")
    if (~have_clip).any():
        p(f"    missing clips: {fga.loc[~have_clip,'fw_stem'].tolist()}")

    p("\n## Cross-validation split coverage")
    split_ids = set(split["participant_id"].map(participant_id))
    lab_ids = set(fga["patient_id"])
    p(f"- split participants: {len(split_ids)}  |  labeled patients: {len(lab_ids)}")
    p(f"- labeled patients present in split: {len(lab_ids & split_ids)}/{len(lab_ids)}")
    p("- labeled rows per fold:")
    sp = split.assign(pid=split["participant_id"].map(participant_id))
    fold_of = dict(zip(sp["pid"], sp["split"]))
    fga = fga.assign(fold=fga["patient_id"].map(fold_of))
    for fold, n in fga["fold"].value_counts().sort_index().items():
        cls = fga[fga["fold"] == fold]["fga_score"].value_counts().sort_index().to_dict()
        p(f"    {fold}: {n} labeled visits, class counts {cls}")

    # labeled patients also having FW Zeno session (for joint labeled features)
    fw_pairs_pat = set(pairs[pairs["protocol"] == "FW"]["patient_id"])
    p(f"\n- labeled patients with >=1 FW Zeno session: {len(lab_ids & fw_pairs_pat)}/{len(lab_ids)}")

    (ARTIFACTS_DIR / "audit_summary.md").write_text(buf.getvalue(), encoding="utf-8")
    p(f"\n[written] {ARTIFACTS_DIR / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
