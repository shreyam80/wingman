"""Collect per-video results for one or more settings into one table.

Reads, for every outputs/<setting>/<video>/ :
  meta.json        -> model, imgsz, speed
  selection.json   -> tracked fraction, number of IDs for the climber, longest loss
  selected.csv     -> fraction of frames with all 4 torso joints usable
  hold_eval.json   -> hand/foot hit rate against the dataset's hold labels, L/R swaps

Usage:
    python -m wingman.compare --names s960,s1280,m960 --participants p1,p3,p5
Writes outputs/comparison_<names>.csv and prints a per-setting summary.
Videos missing a selection or hold evaluation are listed, not silently dropped.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
TORSO = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def one_run(run, min_conf=0.5):
    meta = json.loads((run / "meta.json").read_text())
    row = {"video_id": run.name, "participant": run.name.split("_")[0].rstrip("ab"),
           "model": meta["model"], "imgsz": meta["imgsz"],
           "duration_s": meta["video_info"]["duration_s"], "x_real_time": meta["realtime_factor"]}
    if (run / "selection.json").exists():
        s = json.loads((run / "selection.json").read_text())
        lost = s["lost_intervals_s"]
        sel = pd.read_csv(run / "selected.csv")
        row.update({
            "tracked_fraction": s["tracked_fraction_after_selection"],
            "climber_ids": len(s["chain"]),
            "longest_loss_s": round(max((b - a for a, b in lost), default=0.0), 2),
            "torso_usable_fraction": round(
                ((sel.status == "tracked") & (sel[[f"{t}_conf" for t in TORSO]] >= min_conf).all(axis=1)).mean(), 4),
        })
    if (run / "hold_eval.json").exists():
        h = json.loads((run / "hold_eval.json").read_text())
        for k in ("hands", "feet"):
            row.update({f"{k}_hit_rate": h[k]["hit_rate"], f"{k}_evaluated_fraction": h[k]["evaluated_fraction"],
                        f"{k}_swap_rate": h[k]["left_right"]["swap_rate"]})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", required=True, help="comma list of settings, e.g. s960,s1280,m960")
    ap.add_argument("--participants", help="comma list, e.g. p1,p3,p5 (default: all)")
    args = ap.parse_args()
    names = args.names.split(",")
    keep = set(args.participants.split(",")) if args.participants else None

    rows = []
    for name in names:
        for run in sorted((PROJECT / "outputs" / name).glob("*/meta.json")):
            r = one_run(run.parent)
            if keep is None or r["participant"] in keep:
                rows.append({"setting": name, **r})
    df = pd.DataFrame(rows)
    out = PROJECT / "outputs" / f"comparison_{'_'.join(names)}.csv"
    df.to_csv(out, index=False)

    incomplete = df[df.reindex(columns=["tracked_fraction", "hands_hit_rate"]).isna().any(axis=1)]
    if len(incomplete):
        print("Missing selection or hold evaluation:", incomplete[["setting", "video_id"]].values.tolist())

    metrics = [c for c in ["x_real_time", "tracked_fraction", "climber_ids", "longest_loss_s",
                           "torso_usable_fraction", "hands_hit_rate", "feet_hit_rate",
                           "hands_swap_rate", "feet_swap_rate"] if c in df.columns]
    # Median across videos: one bad video can't dominate, and each video counts once.
    print(df.groupby("setting")[metrics].median().round(3).T.to_string())
    print(f"\nVideos per setting: {df.groupby('setting').size().to_dict()}  ->  {out}")


if __name__ == "__main__":
    main()
