#!/bin/bash
# ==============================================================================
# Full bs=1 action inference (fast CUDA-graph engine) + NAVSIM PDMS evaluation.
# Writes inference/pdms_eval_out/pdms_eval.json with:
#   - number of inferences
#   - average / total processing time (graph capture excluded, PDMS excluded)
#   - average PDMS / NC / DAC / DDC / EP / TTC / C and ADE / FDE / L2
#   - per-sample values (frame_id, inference_ms, trajectory, per-sample metrics)
#
# Usage:
#   bash inference/run_pdms_eval.sh                       # full 12146-sample test cache
#   MAX_SAMPLES=200 bash inference/run_pdms_eval.sh       # quick subset
#   NUM_WORKERS=8 bash inference/run_pdms_eval.sh
#   bash inference/run_pdms_eval.sh --no_graph --no_pdms  # action metrics only
# ==============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

source "$(pwd)/env.sh"
_activate_augustus_env

GPUS=${GPUS:-"0"}
CHECKPOINT="${CHECKPOINT:-${AUGUSTUS_MODEL_ROOT}/navsim-qwen3p5-2b-vit-AdaW-cot_prediction_action_first-fm-x0-adanorm-raw-dit-lr1e-5_clip0.0_diff10_bs256-2048_512/best/pytorch_model}"
CONFIG="${CONFIG:-$(dirname "$(dirname "${CHECKPOINT}")")/config.yaml}"
TEST_DATA="${TEST_DATA:-${AUGUSTUS_DATA_ROOT}/4_data_2048_512/hf_cache_test_CoT_Solution}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/inference/pdms_eval_out}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PDM_WORKERS="${PDM_WORKERS:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

export CUDA_VISIBLE_DEVICES="${GPUS}"

echo "============================================="
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  Config      : ${CONFIG}"
echo "  Test data   : ${TEST_DATA}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  Batch size  : 1 | DataLoader workers: ${NUM_WORKERS} | PDM workers: ${PDM_WORKERS}"
if [[ -n "${MAX_SAMPLES}" ]]; then echo "  Max samples : ${MAX_SAMPLES}"; else echo "  Max samples : full"; fi
echo "  GPU         : ${GPUS}"
echo "============================================="

CMD=(python inference/pdms_eval.py
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --test_data "${TEST_DATA}"
    --output_dir "${OUTPUT_DIR}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
    --pdm_workers "${PDM_WORKERS}")

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" "$@"
