#!/bin/bash
# ==============================================================================
# AUGUstus_SFT — DeepSpeed ZeRO-2 training launch script (Qwen3.5 NAVSIM SFT)
#
# Usage:
#   bash scripts/run_deepspeed.sh              # run training (all local GPUs, ZeRO-2)
#   GPUS=4 bash scripts/run_deepspeed.sh       # train on 4 GPUs
#   source scripts/run_deepspeed.sh            # ONLY activate the conda env
#                                              # in the current shell (no launch)
# ==============================================================================

# ---- detect whether this file is being sourced or executed ---------------
# When sourced, ${BASH_SOURCE[0]} != $0.  We must avoid `set -e` and skip
# `accelerate launch`, otherwise a single non-zero exit would kill the
# user's interactive shell, and we'd block them from getting the env.
_IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _IS_SOURCED=1
fi

# Resolve project root from the script location (works for both source/exec).
_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
_SCRIPT_DIR="$(cd "$(dirname "${_SCRIPT_PATH}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"

# ---- conda env (from env.sh — the single place holding machine paths) --------
source "${_PROJECT_ROOT}/env.sh"
_activate_augustus_env
cd "${_PROJECT_ROOT}"

# ---- print environment info ----
echo "[run_deepspeed.sh] Environment info:"
echo "  which python:    $(which python)"
echo "  python version:  $(python --version 2>&1)"
echo "  conda env:       $CONDA_DEFAULT_ENV"
echo "  conda prefix:    $CONDA_PREFIX"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  CWD:             $(pwd)"

if [[ "${_IS_SOURCED}" == "1" ]]; then
    echo "[run_deepspeed.sh] sourced -> conda env activated, cwd=${_PROJECT_ROOT}"
    echo "[run_deepspeed.sh] (training NOT launched; run 'bash scripts/run_deepspeed.sh' to train)"
    return 0 2>/dev/null || true
fi

# Strict mode only when executed (not sourced).
set -euo pipefail

# ---- multi-node settings (auto-detected from the cluster launcher) ----------
# The launcher runs this SAME script on EVERY node and injects MASTER_ADDR /
# MASTER_PORT / NODE_RANK / SLURM_JOB_NUM_NODES / SLURM_GPUS_PER_NODE itself,
# so nothing has to be set by hand. The `${VAR:-fallback}` defaults below are
# only used for a plain local run.
_LOCAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-${GPUS:-${_LOCAL_GPUS}}}}  # GPUs on THIS node
NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-${WORLD_SIZE:-1}}}                 # total machines
NODE_RANK=${NODE_RANK:-${GROUP_RANK:-${RANK:-0}}}                                # 0=master
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

CONFIG=${CONFIG:-"configs/navsim_qwen3p5.yaml"}
DS_CONFIG=${DS_CONFIG:-"configs/ds_zero2.json"}

echo "Launching DeepSpeed ZeRO training: ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU = ${NUM_PROCESSES} processes"
echo "  node_rank=${NODE_RANK}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "  config=${CONFIG}"

accelerate launch \
    --config_file configs/accelerate_deepspeed.yaml \
    --num_machines ${NUM_NODES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --num_processes ${NUM_PROCESSES} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train_qwen3p5.py --config ${CONFIG}
