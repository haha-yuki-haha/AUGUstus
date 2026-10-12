#!/bin/bash
# ==============================================================================
# AUGUstus_SFT — NAVSIM PDMS metric cache build
#
# Builds the metric caches used by PDMS evaluation:
#   data.metric_cache_test  (navtest split)   — used for validation PDMS
#   data.metric_cache_train (navtrain split)  — optional (training-set PDMS stats)
#
# The cache is written to $NAVSIM_EXP_ROOT/metric_cache_{split} (see env.sh,
# default $ADT_ROOT/Data/debug/navsim_exp/). To match the config keys, copy or
# symlink afterwards:
#   ln -s $NAVSIM_EXP_ROOT/metric_cache $NAVSIM_EXP_ROOT/metric_cache_test
#
# Usage:
#   bash NAVSIM_Cache.sh                    # navtest (needed for val PDMS)
#   TRAIN_TEST_SPLIT=trainval bash NAVSIM_Cache.sh
# ==============================================================================
set -euo pipefail

source "$(dirname "$0")/env.sh"
_activate_augustus_env

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
# cache output dir: navtest -> metric_cache_test, navtrain -> metric_cache_train
#
# NOTE : the TRAIN cache MUST be the `navtrain` benchmark
# split — its scene_filter carries the explicit ~104k navtrain token list with
# frame_interval: 1, which is exactly the frame structure of the training HF
# cache. Do NOT use `train_test_split=trainval` (all_scenes / frame_interval:
# null): it caches different frame windows per scene token, so the PDMS of the
# same trajectory DIFFERS from the navtrain cache (measured: 12/15 spot-checked
# scenes flipped sub-metrics) — and most navtrain tokens are missing entirely.
case "$TRAIN_TEST_SPLIT" in
    navtest)  CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_test ;;
    navtrain) CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train ;;
    trainval) echo "ERROR: TRAIN_TEST_SPLIT=trainval is NOT a valid cache split." >&2
              echo "       The train-split metric cache must be built from the" >&2
              echo "       navtrain benchmark split: TRAIN_TEST_SPLIT=navtrain." >&2
              exit 1 ;;
    *)        CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_${TRAIN_TEST_SPLIT} ;;
esac

echo "[NAVSIM_Cache.sh] split      = $TRAIN_TEST_SPLIT"
echo "[NAVSIM_Cache.sh] cache path = $CACHE_PATH"
echo "[NAVSIM_Cache.sh] maps root  = $NUPLAN_MAPS_ROOT"
echo "[NAVSIM_Cache.sh] devkit     = $NAVSIM_DEVKIT_ROOT"

# The devkit is importable via the editable pip install (see docs/INSTALL.md);
# env.sh also prepends it to PYTHONPATH for non-installed checkouts.

cd "$AUGUSTUS_CODE_ROOT"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH
