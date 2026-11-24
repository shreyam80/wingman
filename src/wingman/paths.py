"""Where the dataset lives on THIS computer, so saved files never contain one person's
folders. Files store video paths relative to the dataset folder (e.g. "p4/orange.mp4").

Default dataset folder: ~/Downloads/wingman dataset
Somewhere else? Set it once in your terminal:  export WINGMAN_DATASET="/path/to/dataset"
"""
import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("WINGMAN_DATASET", "~/Downloads/wingman dataset")).expanduser()


def to_dataset_relative(path):
    """'<dataset folder>/p4/orange.mp4' -> 'p4/orange.mp4'."""
    p = Path(path).expanduser().resolve()
    try:
        return p.relative_to(DATASET_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)  # not inside the dataset folder: keep as given


def dataset_file(path):
    """Turn a stored path (relative, or an old absolute one) into a file on this computer."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p if p.exists() else DATASET_ROOT / Path(*p.parts[-2:])  # old absolute: keep pX/route.mp4
    return DATASET_ROOT / p
