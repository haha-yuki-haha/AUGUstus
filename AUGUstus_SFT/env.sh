#!/bin/bash
# ==============================================================================
# AUGUstus_SFT — central path configuration
#
# >>> THIS IS THE ONLY FILE YOU NEED TO EDIT <<<
#
# Every script (training, evaluation, data processing) sources this file and
# resolves every path from the single root below, so no absolute path is ever
# hard-coded anywhere else in the repository.
#
# Expected directory layout (all derived from ADT_ROOT):
#
#   $ADT_ROOT/
#   ├── AUGUstus_SFT/        this repository (third_party/ lives inside)
#   ├── Data/
#   │   ├── debug/                OUTPUT of the data pipeline (hf caches, ...)
#   │   ├── data_annotation/      CoT annotations (transfusion_outputs_{train,test})
#   │   └── navsim/               RAW NAVSIM dataset (navsim_logs/ sensor_blobs/ maps/)
#   └── Model/
#       ├── Qwen3.5-2B/           backbone weights
#       └── <run_name>/           training outputs
#
# The conda environment itself is NOT kept in the repository (NFS is slow):
# it lives on the fast local disk — Miniconda bootstrapper at
# /opt/augustus_sft/miniconda3 and the Python 3.12 project env at
# /opt/augustus_sft/env (AUGUSTUS_ENV_ROOT below). See docs/INSTALL.md.
#
# ==============================================================================

# ---- the single root ---------------------------------------------------------
_AUGUSTUS_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADT_ROOT="${ADT_ROOT:-$(cd "${_AUGUSTUS_ENV_DIR}/.." && pwd)}"

# ---- repository / environment ------------------------------------------------
export AUGUSTUS_CODE_ROOT="${AUGUSTUS_CODE_ROOT:-$ADT_ROOT/AUGUstus_SFT}"
export AUGUSTUS_ENV_ROOT="${AUGUSTUS_ENV_ROOT:-/opt/augustus_sft/env}"        # conda env on fast local disk
export AUGUSTUS_THIRD_PARTY="${AUGUSTUS_THIRD_PARTY:-$AUGUSTUS_CODE_ROOT/third_party}"  # navsim / nuplan-devkit sources

# ---- raw NAVSIM dataset (download separately, see docs/INSTALL.md) -----------
export NAVSIM_ROOT="${NAVSIM_ROOT:-$ADT_ROOT/Data/navsim}"

# ---- processed data output (target of the data_process pipeline) -------------
export AUGUSTUS_DATA_ROOT="${AUGUSTUS_DATA_ROOT:-$ADT_ROOT/Data/debug}"

# ---- CoT annotations (transfusion_outputs_{train,test}) ----------------------
export AUGUSTUS_ANNOTATION_ROOT="${AUGUSTUS_ANNOTATION_ROOT:-$ADT_ROOT/Data/data_annotation}"

# ---- model weights / training outputs ----------------------------------------
export AUGUSTUS_MODEL_ROOT="${AUGUSTUS_MODEL_ROOT:-$ADT_ROOT/Model}"
export QWEN35_2B_PATH="${QWEN35_2B_PATH:-$AUGUSTUS_MODEL_ROOT/Qwen3.5-2B}"

# ---- swanlab (cloud experiment tracking) --------------------------------------
# Token used when training runs with SWANLAB_MODE=cloud (see docs/TRAINING.md).
# Supply your own token through the environment only. No credential is bundled.
export AUGUSTUS_SWANLAB_TOKEN="${AUGUSTUS_SWANLAB_TOKEN:-}"
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

# ---- NAVSIM devkit environment variables (consumed by nuplan/navsim) ---------
# NOTE: shared clusters often inject stale NAVSIM_*/NUPLAN_* variables for other
# projects (e.g. a leftover NAVSIM_DEVKIT_ROOT=/opt/recogdrive or
# NAVSIM_EXP_ROOT=/data/navsim/exp). Those silently break this repo, so the
# repo layout below is AUTHORITATIVE. Override it only via the AUGUSTUS_*
# variables (AUGUSTUS_EXP_ROOT / AUGUSTUS_MAPS_ROOT), not via NAVSIM_DEVKIT_ROOT.
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${AUGUSTUS_MAPS_ROOT:-$NAVSIM_ROOT/maps}"
export NAVSIM_EXP_ROOT="${AUGUSTUS_EXP_ROOT:-$AUGUSTUS_DATA_ROOT/navsim_exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_ROOT}"
export NAVSIM_DEVKIT_ROOT="${AUGUSTUS_NAVSIM_DEVKIT_ROOT:-$AUGUSTUS_THIRD_PARTY/navsim}"
export NUPLAN_DEVKIT_ROOT="${AUGUSTUS_NUPLAN_DEVKIT_ROOT:-$AUGUSTUS_THIRD_PARTY/nuplan-devkit}"

# ---- make the devkits importable even without the editable pip install -------
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NUPLAN_DEVKIT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# ---- convenience: activate the shipped conda env -----------------------------
_activate_augustus_env() {
    if [ -f "$AUGUSTUS_ENV_ROOT/etc/profile.d/conda.sh" ]; then
        source "$AUGUSTUS_ENV_ROOT/etc/profile.d/conda.sh"
        conda activate "$AUGUSTUS_ENV_ROOT"
    elif [ -x "$AUGUSTUS_ENV_ROOT/bin/python" ]; then
        export PATH="$AUGUSTUS_ENV_ROOT/bin:$PATH"
    fi
}
