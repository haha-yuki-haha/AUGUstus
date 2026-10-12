"""Shared path resolution for the AUGUstus_SFT data pipeline.

All locations come from environment variables defined in ``env.sh`` (repo root)
— the single place where machine-specific paths are configured. Sourcing
``env.sh`` before running any script is therefore required, e.g.::

    source ../env.sh          # from this directory
    python 0_get_useful_scenes.py

(or run everything through ``data_process.sh`` which sources it for you).
"""

import os
from pathlib import Path

# Repo locations (independent of the user's ADT_ROOT).
DATA_PROCESS_DIR = Path(__file__).resolve().parent
CODE_ROOT = DATA_PROCESS_DIR.parent
AUTODRIVE_P3_DIR = DATA_PROCESS_DIR / "AutoDrive_P3"


def _env(name: str, fallback: str) -> str:
    """Read an env var, falling back to a sibling-of-ADT_ROOT default."""
    value = os.environ.get(name)
    if value:
        return value
    # Fallback so the scripts still run when env.sh was not sourced: assume the
    # standard layout next to this repository checkout.
    return str(Path(fallback).expanduser())


def navsim_root() -> Path:
    """Raw NAVSIM dataset root (contains navsim_logs/, sensor_blobs/, maps/)."""
    return Path(_env("NAVSIM_ROOT", "~/AD_Transfusion/Data/navsim"))


def data_root() -> Path:
    """Output root for every pipeline artifact (processed data)."""
    return Path(_env("AUGUSTUS_DATA_ROOT", "~/AD_Transfusion/Data/debug"))


def annotation_root() -> Path:
    """CoT annotation root (contains transfusion_outputs_{train,test}/)."""
    return Path(_env("AUGUSTUS_ANNOTATION_ROOT", "~/AD_Transfusion/Data/data_annotation"))


# ---------------------------------------------------------------------------
# Per-stage paths (train/test split == item)
# ---------------------------------------------------------------------------

def navsim_logs(item: str) -> Path:
    """Raw navsim log pickles for the split ("train" -> trainval, "test" -> test)."""
    log_dir = "trainval" if item == "train" else "test"
    return navsim_root() / "navsim_logs" / log_dir


def sensor_blobs(item: str) -> Path:
    """Raw camera images for the split ("train" -> trainval, "test" -> test)."""
    blob_dir = "trainval" if item == "train" else "test"
    return navsim_root() / "sensor_blobs" / blob_dir


def useful_scenes_pkl(item: str) -> Path:
    return data_root() / f"0_get_useful_{item}_scenes.pkl"


def get_json_dir(item: str) -> Path:
    return data_root() / f"1_get_json_{item}"


def frames_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"2_0_get_all_frames_{item}_{x}_{y}"


def videos_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"2_1_get_all_videos_{item}_{x}_{y}"


def traj_heading_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"3_0_get_traj_heading_{item}_{x}_{y}"


def get_v_dir(item: str, x: int, y: int) -> Path:
    return data_root() / f"3_1_get_v_{item}_{x}_{y}"


def data_4_dir(x: int, y: int) -> Path:
    return data_root() / f"4_data_{x}_{y}"


def autodrive_p3_json(item: str, x: int, y: int) -> Path:
    """Shipped AutoDrive_P3 bbox/trajectory-solution JSON for the resolution."""
    return AUTODRIVE_P3_DIR / f"1_RL_{item}_{x}_{y}.json"


def annotation_outputs(item: str) -> Path:
    """reasoning_trace annotation dir for the split."""
    return annotation_root() / f"transfusion_outputs_{item}"
