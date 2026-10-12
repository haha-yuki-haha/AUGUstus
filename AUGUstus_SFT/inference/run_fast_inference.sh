#!/bin/bash
# ==============================================================================
# Deeply-optimized bs=1 no-CoT single-step action inference (CUDA-graph).
#
# Baseline model.generate_trajectory is CPU-launch-bound (~135 ms/sample);
# this script records the LLM action-decoder stage into a CUDA graph and keeps
# the vision stage eager -> ~72 ms/sample, same numerics.
#
# Usage:
#   bash inference/run_fast_inference.sh                     # default checkpoint
#   CHECKPOINT=/path/to/best/pytorch_model bash inference/run_fast_inference.sh
#   RUNS=30 bash inference/run_fast_inference.sh
#   bash inference/run_fast_inference.sh --infer_only        # print trajectory only
#   bash inference/run_fast_inference.sh --no_graph          # eager fast-path (no graph)
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(pwd)/env.sh"
_activate_augustus_env

GPUS=${GPUS:-"0"}
CHECKPOINT="${CHECKPOINT:-${AUGUSTUS_MODEL_ROOT}/navsim-qwen3p5-2b-vit-AdaW-cot_prediction_action_first-fm-x0-adanorm-raw-dit-lr1e-5_clip0.0_diff10_bs256-2048_512/best/pytorch_model}"
CONFIG="${CONFIG:-$(dirname "$(dirname "${CHECKPOINT}")")/config.yaml}"
TEST_DATA="${TEST_DATA:-${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/inference/fast_bench_out}"
SEED="${SEED:-42}"
RUNS="${RUNS:-20}"
WARMUP="${WARMUP:-5}"
GRAPH_WARMUP="${GRAPH_WARMUP:-5}"
INDEX="${INDEX:-0}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "============================================="
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Config     : ${CONFIG}"
echo "  Test data  : ${TEST_DATA}"
echo "  Output     : ${OUTPUT_DIR}"
echo "  Batch size : 1 | Runs: ${RUNS} (warmup ${WARMUP}) | graph_warmup: ${GRAPH_WARMUP}"
echo "  GPU        : ${GPUS}"
echo "============================================="

python inference/fast_inference.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --test_data "${TEST_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --runs "${RUNS}" \
    --warmup "${WARMUP}" \
    --graph_warmup "${GRAPH_WARMUP}" \
    --index "${INDEX}" \
    "$@"
