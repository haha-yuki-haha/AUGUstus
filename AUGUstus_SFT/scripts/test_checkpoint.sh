#!/bin/bash
# ==============================================================================
# AUGUstus_SFT — Checkpoint evaluation on NAVSIM (PDMS)
#
# Usage:
#   bash scripts/test_checkpoint.sh                               # single GPU (gpu 0)
#   GPUS=0,1,2,3 bash scripts/test_checkpoint.sh                  # multi-GPU
#   CHECKPOINT=/path/to/ckpt bash scripts/test_checkpoint.sh
#   BATCH_SIZE=4 NUM_INFERENCE_STEPS=50 bash scripts/test_checkpoint.sh
#   GENERATE_COT=0 bash scripts/test_checkpoint.sh                # action-only (skip CoT decode)
#   GENERATE_COT=1 bash scripts/test_checkpoint.sh                # force CoT on (CoT-trained ckpts)
#   SAVE_STEPS=1 NUM_INFERENCE_STEPS=1 bash scripts/test_checkpoint.sh  # dump per-step denoising inputs/outputs
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(pwd)/env.sh"
_activate_augustus_env

# ---- configurable variables (defaults from env.sh) ----
# GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
GPUS=${GPUS:-"0,1,2,3"}
CHECKPOINT=${CHECKPOINT:-"${AUGUSTUS_MODEL_ROOT}/RL/stage2/20260918_022302/best_pdms_reward.ckpt/actor/hf_model"}
CONFIG=${CONFIG:-"configs/navsim_qwen3p5.yaml"}
TEST_DATA=${TEST_DATA:-"${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution"}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-"3"}
# auto -> config default (cot.enable_cot AND cot.eval_generate_cot)
# 1    -> force CoT generation on; 0 -> action-only (skip the CoT decode)
# GENERATE_COT=${GENERATE_COT:-auto}
GENERATE_COT=${GENERATE_COT:-0}
MAX_SAMPLES=${MAX_SAMPLES:-""}
SEED=${SEED:-42}
# ---- flow matching sampler: ODE (deterministic) vs SDE (stochastic) ----
# Flow Matching can integrate the learned velocity field two ways at inference:
#   * ODE (default)  : deterministic Euler integration  dx = v(x,t) dt.
#                      Same input -> identical trajectory every run.
#   * SDE (FM_SDE=1) : marginal-preserving reverse-time SDE that injects
#                      Brownian noise each step (SiT / stochastic interpolants).
#                      Adds diversity; SDE_NOISE_SCALE tunes the stochasticity
#                      (0 -> falls back to the ODE, larger -> more randomness).
# Only affects the Flow Matching sampler (this release is FM-only).
FM_SDE=${FM_SDE:-0}                  # 1/true -> stochastic SDE sampler; 0 -> deterministic ODE
# SDE_NOISE_SCALE=${SDE_NOISE_SCALE:-1.0}  # SDE stochasticity multiplier (0 -> ODE)
SDE_NOISE_SCALE=${SDE_NOISE_SCALE:-0.0}  # SDE stochasticity multiplier (0 -> ODE)
# ---- per-step denoising input/output dump ----
SAVE_STEPS=${SAVE_STEPS:-0}          # 1/true -> dump every denoising step to <OUTPUT_DIR>/denoise_steps/

# ---- resolve output dir: embed sampler info so SDE/ODE runs are distinguishable ----
# Build a run tag from the sampler config so each setting writes to its OWN
# folder by default (results never overwrite each other). For the stochastic
# SDE sampler the seed is included too, since different seeds give different
# trajectories. Override OUTPUT_DIR=... to force a fixed path.
if [[ "${FM_SDE}" == "1" || "${FM_SDE,,}" == "true" ]]; then
    RUN_TAG="sde_ns${SDE_NOISE_SCALE}_seed${SEED}"
else
    RUN_TAG="ode"
fi
if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
    RUN_TAG="${RUN_TAG}_steps${NUM_INFERENCE_STEPS}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${CHECKPOINT}/eval_outputs_${RUN_TAG}"}

# Count GPUs from comma-separated list
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "============================================="
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Config     : ${CONFIG}"
echo "  Test data  : ${TEST_DATA}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Batch size : ${BATCH_SIZE}"
echo "  GPUs       : ${GPUS} (${NUM_GPUS} device(s))"
echo "  Sampler    : ${RUN_TAG}  (FM_SDE=${FM_SDE}, noise_scale=${SDE_NOISE_SCALE})"
echo "  Save steps : ${SAVE_STEPS}"
echo "============================================="

# Build command — use accelerate for multi-GPU, plain python for single GPU
BASE_CMD="test_qwen3p5.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --test_data ${TEST_DATA} \
    --batch_eval \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --visualize"

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
    BASE_CMD="${BASE_CMD} --num_inference_steps ${NUM_INFERENCE_STEPS}"
fi

# Eval-time CoT generation switch (auto | 1 | 0)
BASE_CMD="${BASE_CMD} --generate_cot ${GENERATE_COT}"

if [[ -n "${MAX_SAMPLES}" ]]; then
    BASE_CMD="${BASE_CMD} --max_samples ${MAX_SAMPLES}"
fi

# Flow matching SDE sampler (default is the deterministic ODE)
if [[ "${FM_SDE}" == "1" || "${FM_SDE,,}" == "true" ]]; then
    BASE_CMD="${BASE_CMD} --fm_sde --sde_noise_scale ${SDE_NOISE_SCALE}"
fi

# Per-step denoising input/output dump
if [[ "${SAVE_STEPS}" == "1" || "${SAVE_STEPS,,}" == "true" ]]; then
    BASE_CMD="${BASE_CMD} --save_steps"
fi

if [[ ${NUM_GPUS} -gt 1 ]]; then
    CMD="accelerate launch --num_processes ${NUM_GPUS} --num_machines 1 --mixed_precision bf16 ${BASE_CMD}"
else
    CMD="python ${BASE_CMD}"
fi

echo "Running: ${CMD}"
eval ${CMD}
