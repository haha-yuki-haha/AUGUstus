#!/bin/bash
# ==============================================================================
# Single-sample (batch_size=1) inference benchmark for a Transfusion checkpoint.
#
# Measures how long ONE sample takes and breaks it down per model module
# (vision encoder / LLM prefix / diffusion denoising / CoT decode / ...).
# Writes inference/bench_out/timing_results.json and inference/inference_timing.md.
#
# Usage:
#   bash inference/run_benchmark.sh                      # default checkpoint below
#   CHECKPOINT=/path/to/best/pytorch_model bash inference/run_benchmark.sh
#   CONFIG=/path/to/config.yaml TEST_DATA=/path/to/cache bash inference/run_benchmark.sh
#   RUNS=10 WARMUP=3 bash inference/run_benchmark.sh
#   GENERATE_COT=0 bash inference/run_benchmark.sh       # action-only (skip CoT)
#   NUM_INFERENCE_STEPS=2 bash inference/run_benchmark.sh
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(pwd)/env.sh"
_activate_augustus_env

# ---- configurable variables --------------------------------------------------
GPUS=${GPUS:-"0"}
CHECKPOINT="${CHECKPOINT:-${AUGUSTUS_MODEL_ROOT}/navsim-qwen3p5-2b-vit-AdaW-cot_prediction_action_first-fm-x0-adanorm-raw-dit-lr1e-5_clip0.0_diff10_bs256-2048_512/best/pytorch_model}"
# The run's own training config (contains model_type/use_action_first/diffusion/cot settings).
CONFIG="${CONFIG:-$(dirname "$(dirname "${CHECKPOINT}")")/config.yaml}"
TEST_DATA="${TEST_DATA:-${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/inference/bench_out_current}"
SEED="${SEED:-42}"
RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-2}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
MAX_COT_TOKENS="${MAX_COT_TOKENS:-}"
# auto -> use config cot.enable_cot; 1 -> force CoT generation; 0 -> action-only
GENERATE_COT="${GENERATE_COT:-auto}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "============================================="
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Config     : ${CONFIG}"
echo "  Test data  : ${TEST_DATA}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Batch size : 1"
echo "  GPU        : ${GPUS}"
echo "  Runs       : ${RUNS} (warmup ${WARMUP})"
echo "  CoT        : ${GENERATE_COT}"
echo "============================================="

CMD=(python inference/benchmark_inference.py
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --test_data "${TEST_DATA}"
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED}"
    --runs "${RUNS}"
    --warmup "${WARMUP}"
    --generate_cot "${GENERATE_COT}")

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
    CMD+=(--num_inference_steps "${NUM_INFERENCE_STEPS}")
fi
if [[ -n "${MAX_COT_TOKENS}" ]]; then
    CMD+=(--max_cot_tokens "${MAX_COT_TOKENS}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
