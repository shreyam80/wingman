"""Build data/manifest.csv: one row per small (720x1280, 25 FPS) dataset video.

Records provenance (source, license, participant, route), a SHA-256 hash of the
original file so we can later prove which exact input produced a result, and the
measured video properties. Original videos are only read, never modified.

Usage:
    python -m wingman.manifest --dataset "~/Downloads/wingman dataset"
"""
import argparse
import csv
import hashlib
from pathlib import Path

from wingman.video_io import video_info

SOURCE = "The Way Up (Zenodo record 15196867, DOI 10.5281/zenodo.15196867, v1.0.0)"
LICENSE = "CC-BY-4.0 (attribution required; checked via Zenodo API 2026-10-28)"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser()
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        for route in ("orange", "green"):
            video = folder / f"{route}.mp4"  # small version; annotations use these frames
            if not video.exists():
                continue
            info = video_info(video)
            rows.append({
                "video_id": f"{folder.name}_{route}",
                # p2a / p2b are the same person recorded twice (see participant_form_data.csv)
                "participant_group": folder.name.rstrip("ab"),
                "route": route,
                "path": str(video),
                "sha256": sha256(video),
                "width": info["width"], "height": info["height"],
                "fps": info["fps"], "frame_count": info["frame_count"],
                "duration_s": round(info["duration_s"], 2),
                "hold_usage_csv": (folder / f"{route}_holdUsage.csv").name,
                "source": SOURCE, "license": LICENSE,
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} videos to {args.out}")


if __name__ == "__main__":
    main()
