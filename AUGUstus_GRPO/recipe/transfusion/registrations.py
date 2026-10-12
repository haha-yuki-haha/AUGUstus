# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Side-effect import that registers the Transfusion rollout + training engine.

verl imports an "external lib" in every process where the model config is built
(``HFModelConfig.__post_init__`` calls ``import_external_libs(self.external_lib)``)
and, optionally, at ``verl`` package import time via the
``VERL_USE_EXTERNAL_MODULES`` environment variable. Pointing either hook at this
module guarantees the custom rollout / engine registries are populated in the
driver **and** in the Ray workers, without editing any verl core file.

Recipe config wiring::

    actor_rollout_ref:
      model:
        external_lib: recipe.transfusion.registrations

or (global) ``export VERL_USE_EXTERNAL_MODULES=recipe.transfusion.registrations``.

Importing this module is cheap and defensive: registration failures are logged,
never raised, so a partial environment cannot break config resolution.
"""

from __future__ import annotations

import logging
import os
import warnings

logger = logging.getLogger(__name__)

# SwanLab experiment-id persistence file, relative to the run's SWANLAB_LOG_DIR.
# Mirrors SFT (AUGUstus_SFT/transfusion/logger.py): <output_dir>/swanlab/run_id.txt
_SWANLAB_RUN_ID_FILE = "run_id.txt"


def _patch_swanlab_init_for_resume(swanlab_mod=None) -> bool:
    """Make ``swanlab.init`` resume the SAME cloud experiment across a relaunch.

    Learned from the supervised trainer's resume convention
    (``AUGUstus_SFT/transfusion/logger.py``): the experiment id is persisted
    in ``<logdir>/run_id.txt``. A later launch pointing at the SAME ``logdir``
    (``SWANLAB_LOG_DIR`` = ``<run_dir>/swanlab``, and a resumed run keeps its
    timestamp dir) passes that id back with ``resume="allow"`` so metrics
    continue in the SAME experiment instead of a duplicate one; a genuinely
    fresh run finds no id file, creates a new experiment and persists its id.

    Installed here (not by editing verl core) by wrapping ``swanlab.init`` —
    verl's ``Tracking.__init__`` calls it once per launch with keyword
    ``logdir=$SWANLAB_LOG_DIR``. Callers that pass an explicit ``id=`` are left
    untouched. All id-file I/O is warn-only: logging must never crash training.

    Returns True when the wrapper was newly installed.
    """
    if swanlab_mod is None:
        try:
            import swanlab as swanlab_mod
        except Exception:
            return False
    if getattr(swanlab_mod.init, "_transfusion_resume_patched", False):
        return False
    orig_init = swanlab_mod.init

    def init_with_resume(*args, **kwargs):
        # Path/mode resolution is warn-only: any failure falls back to plain init.
        try:
            logdir = kwargs.get("logdir") or os.environ.get("SWANLAB_LOG_DIR") or "swanlog"
            id_file = os.path.join(str(logdir), _SWANLAB_RUN_ID_FILE)
            # swanlab/data/sdk.py: non-cloud modes demote resume -> 'never', and
            # 'never'+id raises RuntimeError -- only inject/persist for cloud runs.
            mode = kwargs.get("mode") or os.environ.get("SWANLAB_MODE") or "cloud"
            cloud = mode in ("cloud", "shared")
        except Exception as e:  # pragma: no cover - filesystem dependent
            logger.warning("swanlab resume patch path resolution failed: %s", e)
            id_file, cloud = None, False
        if cloud and id_file and kwargs.get("id") is None and os.path.isfile(id_file):
            try:
                with open(id_file) as f:
                    existing_id = f.read().strip() or None
                if existing_id:
                    kwargs["id"] = existing_id
                    kwargs.setdefault("resume", "allow")
            except Exception as e:  # pragma: no cover - filesystem dependent
                logger.warning("swanlab resume-id read failed (%s); starting a new experiment", e)
        run = orig_init(*args, **kwargs)
        if cloud and id_file:
            try:
                run_id = getattr(run, "id", None)
                if run_id:
                    os.makedirs(os.path.dirname(id_file), exist_ok=True)
                    with open(id_file, "w") as f:
                        f.write(str(run_id))
            except Exception as e:  # pragma: no cover - filesystem dependent
                logger.warning("swanlab resume-id persist failed: %s", e)
        return run

    init_with_resume._transfusion_resume_patched = True
    swanlab_mod.init = init_with_resume
    return True

# ---------------------------------------------------------------------------
# Silence KNOWN-noisy third-party warnings.
#
# This module is imported by ``verl/__init__`` (via VERL_USE_EXTERNAL_MODULES)
# in the driver AND in every Ray worker BEFORE the verl engine / trainer modules
# that trigger these, so installing the filters here covers all processes:
#   * mindspeed/megatron NPU shim   (verl/utils/megatron/router_replay_patch.py)
#   * FSDP.state_dict_type deprecation FutureWarning (torch >= 2.6)
#   * triton tl.make_block_ptr deprecation (fused attention kernels)
#   * torchdata StatefulDataLoader 'set_vital' deprecation
#   * verl's RayPPOTrainer deprecation FutureWarning (main_ppo -> main_ppo_sync)
# Each filter matches by message PREFIX only, so genuinely new warnings from the
# same modules still surface.
# ---------------------------------------------------------------------------
for _pattern in (
    r"NPU not support router replay.*",
    r"FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated.*",
    r"tl\.make_block_ptr is deprecated.*",
    r"'set_vital' is deprecated.*",
    r"Warning: Class 'verl\.trainer\.ppo\.ray_trainer\.RayPPOTrainer' is deprecated.*",
):
    warnings.filterwarnings("ignore", message=_pattern)


def _register() -> None:
    # Legacy-torch unpickle shim: actor shards saved by torch>=2.13 conventions
    # reference torch.distributed._mesh_layout / _dtensor_spec.ShardOrderEntry /
    # DeviceMesh._rank_map, which this env's torch lacks; register data-only
    # replicas IN THIS PROCESS so checkpoint resume loading can unpickle them.
    try:
        from recipe.transfusion.scripts.convert_actor_to_hf import (
            _install_torch_legacy_shims,
        )

        _install_torch_legacy_shims()
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("torch legacy unpickle shim not installed: %s", e)

    # Rollout: inserts ("transfusion", *) into verl's _ROLLOUT_REGISTRY on import.
    try:
        from recipe.transfusion import rollout as _rollout  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Transfusion rollout not registered: %s", e)

    # Engine: registers model_type="transfusion_diffusion" with EngineRegistry.
    try:
        from recipe.transfusion import model_engine as _engine  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Transfusion engine not registered: %s", e)

    # SwanLab: wrap swanlab.init so a RELAUNCH into the same run dir (resume)
    # continues the same cloud experiment instead of a duplicate (SFT-style
    # run_id.txt under SWANLAB_LOG_DIR). verl's Tracking init then resumes for
    # free; sample_logging reuses the active run via swanlab.get_run().
    try:
        _patch_swanlab_init_for_resume()
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("swanlab resume patch not installed: %s", e)


_register()
