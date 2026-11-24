"""Load configs/follow.yaml so code can write cfg.controller.kp instead of dict lookups."""
from pathlib import Path
from types import SimpleNamespace

import yaml

PROJECT = Path(__file__).resolve().parents[3]


def _to_ns(x):
    return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()}) if isinstance(x, dict) else x


def load_config(path=None, overrides=()):
    """Returns (cfg, raw): cfg for attribute access, raw dict to save next to results.
    overrides: strings like "detector.conf=0.15" that replace single settings, so
    experiments don't require editing the file."""
    path = Path(path) if path else PROJECT / "configs/follow.yaml"
    raw = yaml.safe_load(open(path))
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, last = key.split(".")
        node = raw
        for p in parents:
            node = node[p]
        if last not in node:
            raise KeyError(f"Unknown setting {key!r} - check configs/follow.yaml")
        node[last] = yaml.safe_load(value)  # "0.15" -> 0.15, "yolo11s.pt" -> str
    return _to_ns(raw), raw
