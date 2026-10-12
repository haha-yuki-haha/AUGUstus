#!/usr/bin/env bash
# Convert a verl FSDP actor checkpoint into a Hugging Face model directory.
# Output includes model.safetensors, config.json, tokenizer files, and chat_template.
#
# Usage:
#   bash convert_actor_to_hf.sh <actor_dir> [output_dir]
#   OUTPUT_DIR=... bash convert_actor_to_hf.sh <actor_dir>
#
# actor_dir contains model_world_size_*_rank_*.pt, fsdp_config.json, and
# huggingface/ with the config and tokenizer files. The output directory
# defaults to <actor_dir>/hf_model.
#
# Use the existing training environment configured through env.sh.
# AUGUSTUS_ENV_ROOT selects its path; AUGUSTUS_MINICONDA_ROOT selects Conda.
# The converter restores the checkpoint with Torch compatibility shims and
# performs consistency checks without installing or upgrading packages.
set -euo pipefail

# =========================== Default configuration (arguments and environment overrides)==================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Paths derive from the repo's env.sh (single source of truth) — no absolute
# team-mount paths are hardcoded here. env.sh is idempotent: skip if already
# sourced (AUGUSTUS_GRPO_ROOT set), else source it from the repo root.
if [ -z "${AUGUSTUS_GRPO_ROOT:-}" ]; then
    # recipe/transfusion/scripts/* -> repo root is 3 levels up.
    source "$(cd "${SCRIPT_DIR}/../../.." && pwd)/env.sh"
fi

DEFAULT_ACTOR_DIR="${RL_OUTPUT_ROOT:-${AUGUSTUS_MODEL_ROOT}}"
ACTOR_DIR="${ACTOR_DIR:-${1:-}}"
: "${ACTOR_DIR:?Pass the stage-1 actor checkpoint directory as the first argument}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-${ACTOR_DIR}/hf_model}}"
# Trim leading and trailing whitespace from checkpoint paths
ACTOR_DIR="$(echo "${ACTOR_DIR}" | xargs)"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}" | xargs)"

CONDA_SH="${AUGUSTUS_MINICONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
# The project env is a PREFIX env (/opt/augustus_grpo/env), activated by PATH:
CONDA_ENV="${AUGUSTUS_ENV_ROOT:-/opt/augustus_grpo/env}"
# =============================================================================

# --------------------------- Activate the Conda environment --------------------------------
if [ -x "${CONDA_ENV}/bin/python" ]; then
    export PATH="${CONDA_ENV}/bin:${PATH}"
    echo "[convert_actor_to_hf] env: ${CONDA_ENV} ($(command -v python3))"
else
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[convert_actor_to_hf] conda env: ${CONDA_ENV} ($(command -v python3))"
fi

# --------------------------- Validate arguments ---------------------------------------
[[ -d "${ACTOR_DIR}" ]] || { echo "[convert_actor_to_hf] actor directory does not exist: ${ACTOR_DIR}" >&2; exit 1; }
if [[ "$(realpath "${OUTPUT_DIR}")" == "$(realpath "${ACTOR_DIR}")/huggingface" ]]; then
    echo "[convert_actor_to_hf] ERROR: OUTPUT_DIR must not be <actor>/huggingface" >&2
    echo "                         (the converter reads config/tokenizer files from this directory)." >&2
    exit 1
fi
if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]] || ! compgen -G "${ACTOR_DIR}/model_world_size_*_rank_*.pt" >/dev/null; then
    echo "[convert_actor_to_hf] ${ACTOR_DIR} does not appear to be a verl actor directory" >&2
    echo "                      (requires fsdp_config.json + model_world_size_*_rank_*.pt)" >&2
    exit 1
fi

echo "[convert_actor_to_hf] actor : ${ACTOR_DIR}"
echo "[convert_actor_to_hf] output: ${OUTPUT_DIR}"

# The converter performs three consistency checks:
# 1. Bitwise equality after the safetensors round trip.
# 2. Correct action_q01/q99/dim_weights shapes and agreement across ranks.
# 3. Matching keys, shapes, and dtypes against the SFT reference; trained
#    parameter values may differ, while fixed buffers are checked separately.
DEFAULT_REFERENCE="${AUGUSTUS_SFT_RUN:-}/best/pytorch_model"
REFERENCE="${REFERENCE:-${DEFAULT_REFERENCE}}"

# --------------------------- Run conversion ---------------------------------------
python3 "${SCRIPT_DIR}/convert_actor_to_hf.py" \
    --actor "${ACTOR_DIR}" \
    --output "${OUTPUT_DIR}" \
    --reference "${REFERENCE}"

echo "[convert_actor_to_hf] Conversion complete -> ${OUTPUT_DIR}"
