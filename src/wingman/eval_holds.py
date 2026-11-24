"""Check our joints against the dataset's own hold-contact labels.

The Way Up says, per limb, which hold it was on and in which frames (holdUsage.csv),
and where each hold is in the picture (routeAnnotation.json, a box per hold; hold id N
= the N-th box, counting from 0). Those labels were made by the dataset authors, so
they are an independent check of our model:

  "right hand on hold 7 at frame 300"  ->  is our right_wrist near hold 7's box?

We skip frames the authors marked as occluded (limb hidden), and count separately
the frames where we had no usable estimate (climber lost / joint low confidence), so
a good-looking hit rate can't hide missing data.

Also checks left/right swaps: when the two hands (or feet) are on different holds,
is our left wrist at the right hand's hold and vice versa?

Usage:
    python -m wingman.eval_holds --run outputs/s960/p1_orange
Writes <run>/hold_eval.json (summary) and <run>/hold_eval_frames.csv (every check).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]


def parse_ranges(text):
    """'0-47,123-208' -> set of frames 0..47 and 123..208 (inclusive)."""
    frames = set()
    for part in str(text).replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            frames.update(range(int(a), int(b) + 1))
        elif part.isdigit():
            frames.add(int(part))
    return frames


def load_contacts(usage_csv, limb_to_joint):
    """One row per (frame, limb, hold) where the limb was on a hold and NOT occluded."""
    raw = pd.read_csv(usage_csv, dtype=str).fillna("")
    rows, skipped = [], {"wall": 0, "unknown_code": 0, "occluded_frames": 0}
    for r in raw.itertuples(index=False):
        parts = r[0].split()
        if len(parts) < 2:
            continue
        limb, hold = parts[0], parts[1]
        if hold == "w":
            skipped["wall"] += 1
            continue
        if limb not in limb_to_joint or not hold.isdigit():
            skipped["unknown_code"] += 1
            continue
        start, end, occ = int(r[1]), int(r[2]), parse_ranges(r[3])
        for f in range(start, end + 1):
            if f in occ:
                skipped["occluded_frames"] += 1
            else:
                rows.append({"frame_idx": f, "limb": limb, "hold": int(hold)})
    return pd.DataFrame(rows, columns=["frame_idx", "limb", "hold"]), skipped


def dist_to_box(x, y, b):
    """Pixel distance from point to box edge (0 if the point is inside the box)."""
    dx = max(b[0] - x, 0, x - b[2])
    dy = max(b[1] - y, 0, y - b[3])
    return float(np.hypot(dx, dy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(PROJECT / "configs/eval.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tol, min_conf, l2j = cfg["tolerance_torso_lengths"], cfg["min_joint_conf"], cfg["limb_to_joint"]

    run = Path(args.run)
    video = Path(json.loads((run / "meta.json").read_text())["video"])
    route = video.stem  # orange / green
    holds = json.load(open(video.parent / f"{route}_routeAnnotation.json"))["holds"]
    boxes = [(h["topLeftCorner"]["x"], h["topLeftCorner"]["y"],
              h["bottomRightCorner"]["x"], h["bottomRightCorner"]["y"]) for h in holds]
    contacts, skipped = load_contacts(video.parent / f"{route}_holdUsage.csv", l2j)
    sel = pd.read_csv(run / "selected.csv").set_index("frame_idx")

    checks = []
    for c in contacts.itertuples(index=False):
        row = {"frame_idx": c.frame_idx, "limb": c.limb, "hold": c.hold, "joint": l2j[c.limb]}
        if c.frame_idx not in sel.index or sel.at[c.frame_idx, "status"] != "tracked":
            row["result"] = "climber_not_tracked"
        else:
            s = sel.loc[c.frame_idx]
            torso = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            j = l2j[c.limb]
            if any(s[f"{t}_conf"] < min_conf for t in torso):
                row["result"] = "no_torso_ruler"
            elif s[f"{j}_conf"] < min_conf:
                row["result"] = "joint_low_conf"
            else:
                sh = ((s.left_shoulder_x + s.right_shoulder_x) / 2, (s.left_shoulder_y + s.right_shoulder_y) / 2)
                hp = ((s.left_hip_x + s.right_hip_x) / 2, (s.left_hip_y + s.right_hip_y) / 2)
                torso_len = float(np.hypot(sh[0] - hp[0], sh[1] - hp[1]))
                d = dist_to_box(s[f"{j}_x"], s[f"{j}_y"], boxes[c.hold]) / torso_len
                # Sanity check of the hold-id convention: distance if ids were off by one.
                d_shift = (dist_to_box(s[f"{j}_x"], s[f"{j}_y"], boxes[c.hold + 1]) / torso_len
                           if c.hold + 1 < len(boxes) else np.nan)
                row.update({"result": "hit" if d <= tol else "miss", "dist_torso": round(d, 3),
                            "dist_if_ids_shifted": round(d_shift, 3), "torso_px": round(torso_len, 1)})
        checks.append(row)
    ch = pd.DataFrame(checks)
    ch.to_csv(run / "hold_eval_frames.csv", index=False)

    def swaps(pair):
        """Frames where the two limbs of `pair` are on different, single holds."""
        a, b = pair
        ev = ch[ch.result.isin(["hit", "miss"])]
        n_check = n_swap = 0
        for f, g in ev.groupby("frame_idx"):
            ga, gb = g[g.limb == a], g[g.limb == b]
            if len(ga) != 1 or len(gb) != 1 or ga.hold.iloc[0] == gb.hold.iloc[0]:
                continue
            s = sel.loc[f]
            ja, jb = l2j[a], l2j[b]
            ha, hb = boxes[ga.hold.iloc[0]], boxes[gb.hold.iloc[0]]
            straight = dist_to_box(s[f"{ja}_x"], s[f"{ja}_y"], ha) + dist_to_box(s[f"{jb}_x"], s[f"{jb}_y"], hb)
            crossed = dist_to_box(s[f"{ja}_x"], s[f"{ja}_y"], hb) + dist_to_box(s[f"{jb}_x"], s[f"{jb}_y"], ha)
            n_check += 1
            n_swap += crossed < straight
        return {"checkable_frames": n_check, "swapped_frames": int(n_swap),
                "swap_rate": round(n_swap / n_check, 4) if n_check else None}

    def group(limbs):
        g = ch[ch.limb.isin(limbs)]
        ev = g[g.result.isin(["hit", "miss"])]
        return {
            "contact_frames": len(g),
            "evaluated": len(ev),
            "evaluated_fraction": round(len(ev) / len(g), 4) if len(g) else None,
            "not_evaluated_reasons": g[~g.result.isin(["hit", "miss"])].result.value_counts().to_dict(),
            "hit_rate": round((ev.result == "hit").mean(), 4) if len(ev) else None,
            "median_dist_torso_lengths": round(ev.dist_torso.median(), 3) if len(ev) else None,
            "median_dist_if_ids_shifted": round(ev.dist_if_ids_shifted.median(), 3) if len(ev) else None,
        }

    summary = {
        "video": str(video), "config": cfg, "skipped_label_rows": skipped,
        "hands": {**group(["lh", "rh"]), "left_right": swaps(("lh", "rh"))},
        "feet": {**group(["lf", "rf"]), "left_right": swaps(("lf", "rf"))},
        "note": "Joints are wrists/ankles, not fingers/toes; hit = within tolerance of the "
                "labelled hold box. Not-evaluated frames are excluded from hit_rate, not counted as hits.",
    }
    (run / "hold_eval.json").write_text(json.dumps(summary, indent=2))
    for k in ("hands", "feet"):
        s = summary[k]
        print(f"{k}: hit rate {s['hit_rate']} on {s['evaluated']}/{s['contact_frames']} frames, "
              f"median dist {s['median_dist_torso_lengths']} torso lengths "
              f"(if ids shifted: {s['median_dist_if_ids_shifted']}), swaps {s['left_right']}")


if __name__ == "__main__":
    main()
