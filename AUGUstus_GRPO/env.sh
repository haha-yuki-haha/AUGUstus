#!/bin/bash
# ==============================================================================
# AUGUstus_GRPO — central path configuration
#
# >>> THIS IS THE ONLY FILE YOU NEED TO EDIT <<<
#
# Every script (RL training, HF conversion, env setup) sources this file and
# resolves every path from the single root below, so no absolute path is ever
# hard-coded anywhere else in the repository.
#
# AUGUstus_GRPO is the RL (GRPO post-training) stage on top of AUGUstus_SFT:
# the model / dataset / rollout code is REUSED from the sibling AUGUstus_SFT
# repository (TRANSFUSION_ROOT below), so that repo must sit next to this one.
#
# Expected directory layout (all derived from ADT_ROOT):
#
#   $ADT_ROOT/
#   ├── AUGUstus_GRPO/       this repository (vendored verl framework)
#   ├── AUGUstus_SFT/        sibling SFT repo (model + dataset code,
#   │                             third_party/ navsim + nuplan-devkit, configs)
#   ├── Data/
#   │   ├── debug/                OUTPUT of the (SFT) data pipeline
#   │   │   └── 4_data_2048_512/  hf_cache_{train,test}_CoT_Solution  ← RL eats these
#   │   ├── data_annotation/      CoT annotations (data-pipeline input)
#   │   └── navsim/               RAW NAVSIM dataset (navsim_logs/ sensor_blobs/ maps/)
#   └── Model/
#       ├── Qwen3.5-2B/           backbone weights (SFT only)
#       ├── navsim-qwen3p5-2b-…/  the SFT run dir (config_original.yaml + best/)
#       └── RL/<exp_name>/        RL outputs (created by training)
#
# The conda environment itself is NOT kept in the repository (NFS is slow):
# it lives on the fast local disk — Miniconda bootstrapper at
# /opt/augustus_grpo/miniconda3 and the Python 3.12 project env at
# /opt/augustus_grpo/env (AUGUSTUS_ENV_ROOT below). See docs/INSTALL.md.
#
# ==============================================================================

# ---- the single root ---------------------------------------------------------
_AUGUSTUS_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ADT_ROOT="${ADT_ROOT:-$(cd "${_AUGUSTUS_ENV_DIR}/.." && pwd)}"

# ---- repositories ------------------------------------------------------------
export AUGUSTUS_GRPO_ROOT="${AUGUSTUS_GRPO_ROOT:-$ADT_ROOT/AUGUstus_GRPO}"   # this repo
export AUGUSTUS_SFT_ROOT="${AUGUSTUS_SFT_ROOT:-$ADT_ROOT/AUGUstus_SFT}"      # sibling SFT repo

# ---- environment (conda env on fast local disk) ------------------------------
export AUGUSTUS_ENV_ROOT="${AUGUSTUS_ENV_ROOT:-/opt/augustus_grpo/env}"
export AUGUSTUS_MINICONDA_ROOT="${AUGUSTUS_MINICONDA_ROOT:-/opt/augustus_grpo/miniconda3}"

# ---- raw NAVSIM dataset + processed data (SHARED with AUGUstus_SFT) ----------
# The RL run consumes the SAME hf caches the SFT stage produced:
#   $AUGUSTUS_DATA_ROOT/4_data_2048_512/hf_cache_{train,test}_CoT_Solution
export NAVSIM_ROOT="${NAVSIM_ROOT:-$ADT_ROOT/Data/navsim}"
export AUGUSTUS_DATA_ROOT="${AUGUSTUS_DATA_ROOT:-$ADT_ROOT/Data/debug}"

# ---- model weights / training outputs ----------------------------------------
export AUGUSTUS_MODEL_ROOT="${AUGUSTUS_MODEL_ROOT:-$ADT_ROOT/Model}"
# The SFT run whose `best/` checkpoint the released RL run started from
# (its config_original.yaml is also the model-architecture config $TRANSFUSION_CONFIG).
export AUGUSTUS_SFT_RUN="${AUGUSTUS_SFT_RUN:-$AUGUSTUS_MODEL_ROOT/SFT}"
export RL_OUTPUT_ROOT="${RL_OUTPUT_ROOT:-$AUGUSTUS_MODEL_ROOT/RL}"

# ---- TRANSFUSION_* — the vars the recipe/verl workers consume ----------------
# Model + dataset code (sibling repo; has model/ + train_qwen3p5.py).
export TRANSFUSION_ROOT="${TRANSFUSION_ROOT:-$AUGUSTUS_SFT_ROOT}"
# Model-architecture config: the TRAINED SFT run's config_original.yaml (ships
# with the released SFT model). MUST match $TRANSFUSION_CKPT.
export TRANSFUSION_CONFIG="${TRANSFUSION_CONFIG:-$AUGUSTUS_SFT_RUN/config_original.yaml}"
# The trained SFT checkpoint the RL policy starts from (its `best/` holds
# pytorch_model/model.safetensors). Set TRANSFUSION_CKPT="" for a cold start.
export TRANSFUSION_CKPT="${TRANSFUSION_CKPT:-$AUGUSTUS_SFT_RUN/best}"
# Backbone architecture + in-model AutoProcessor (tokenizer + image/video
# processor): the SFT model's own HF dir. Mirrors actor_rollout_ref.model.path.
export TRANSFUSION_QWEN_MODEL_NAME="${TRANSFUSION_QWEN_MODEL_NAME:-$AUGUSTUS_SFT_RUN/best/pytorch_model}"
# Action-norm stats JSON (recipe-bundled copy; the VALUES are overwritten by the
# checkpoint's action_q01/q99 — this only needs to be an accessible file).
export TRANSFUSION_STATS_FILE="${TRANSFUSION_STATS_FILE:-$AUGUSTUS_GRPO_ROOT/recipe/transfusion/config/action_norm_stats.json}"
# ViT frozen during the RL actor update (the released runs use true).
export TRANSFUSION_FREEZE_VIT="${TRANSFUSION_FREEZE_VIT:-true}"

# ---- NAVSIM devkit environment variables (consumed by nuplan/navsim) ---------
# Same convention as AUGUstus_SFT/env.sh: the repo layout below is
# AUTHORITATIVE (shared clusters often inject stale NAVSIM_*/NUPLAN_* variables
# from other projects). Override via the AUGUSTUS_* variables only.
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${AUGUSTUS_MAPS_ROOT:-$NAVSIM_ROOT/maps}"
export NAVSIM_EXP_ROOT="${AUGUSTUS_EXP_ROOT:-$AUGUSTUS_DATA_ROOT/navsim_exp}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$NAVSIM_ROOT}"
export NAVSIM_DEVKIT_ROOT="${AUGUSTUS_NAVSIM_DEVKIT_ROOT:-$AUGUSTUS_SFT_ROOT/third_party/navsim}"
export NUPLAN_DEVKIT_ROOT="${AUGUSTUS_NUPLAN_DEVKIT_ROOT:-$AUGUSTUS_SFT_ROOT/third_party/nuplan-devkit}"

# ---- SwanLab experiment tracking ---------------------------------------------
# verl reads the token from SWANLAB_API_KEY (ray_kwargs env_vars). The released
# Cloud logging requires your own API key; offline logging is the default.
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

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
