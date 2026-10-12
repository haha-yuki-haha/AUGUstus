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

"""Transfusion-style per-run checkpoint layout for the RL recipe.

Mirrors the supervised trainer's output convention
(``AUGUstus_SFT/train_qwen3p5.py``):

* One timestamped run dir per launch —
  ``<trainer.default_local_dir>/<YYYYMMDD_HHMMSS>/`` (analogous to SFT's
  ``<Model>/<run_name>/<timestamp>/``). Everything verl writes for the run
  (``global_step_N/`` checkpoints, ``latest_checkpointed_iteration.txt``,
  SwanLab cache, rollout/validation dumps, ``best_*`` snapshots) lands inside.

* ``trainer.save_optimizer`` selects whether periodic checkpoints carry the
  optimizer/scheduler/rng state (resumable) or weights only. It is mapped onto
  ``actor_rollout_ref.<actor|critic>.checkpoint.save_contents`` (verl's
  ``CheckpointConfig``; ``load_contents`` interpolates ``save_contents`` in
  ``actor.yaml``, so resume loading follows automatically).

* ``trainer.resume.enabled`` / ``trainer.resume.resume_path`` are the resume
  switch + parameter, like SFT's top-level ``resume:`` block. ``resume_path``
  accepts EITHER a run's timestamp dir (the latest ``global_step_*`` inside is
  used) OR a concrete ``.../global_step_N`` dir; ``null`` + ``enabled: true``
  auto-detects the latest run under ``default_local_dir`` (and falls back to a
  fresh run when nothing is found, like verl's ``resume_mode: auto``). On
  resume NO new timestamp dir is minted: ``default_local_dir`` is re-pointed at
  the resumed run's own dir so new checkpoints continue in place.

Applied exactly once at the recipe entry point (:mod:`recipe.transfusion.train`)
BEFORE any ``${trainer.default_local_dir}`` interpolation is resolved, so
``rollout_data_dir`` / ``validation_data_dir`` / ``SWANLAB_LOG_DIR`` /
``TRANSFUSION_OUTPUT_DIR`` / best-ckpt snapshots all follow automatically.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from omegaconf import OmegaConf

RUN_TS_FORMAT = "%Y%m%d_%H%M%S"
_STEP_DIR_RE = re.compile(r"^global_step_(\d+)$")

# verl CheckpointConfig contents (verl/trainer/config/actor/actor.yaml).
FULL_SAVE_CONTENTS = ["model", "optimizer", "extra"]  # resumable
WEIGHTS_ONLY_SAVE_CONTENTS = ["model", "extra"]  # no optimizer shards


def _latest_step_dir(run_dir: str) -> str | None:
    """Return ``<run_dir>/global_step_N`` with the largest N, or None.

    Prefers verl's ``latest_checkpointed_iteration.txt`` tracker (the atomic
    source of truth); falls back to scanning ``global_step_*`` dirs.
    """
    tracker = os.path.join(run_dir, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker):
        try:
            with open(tracker) as f:
                n = int(f.read().strip())
        except (OSError, ValueError):
            n = None
        if n is not None:
            step_dir = os.path.join(run_dir, f"global_step_{n}")
            if os.path.isdir(step_dir):
                return step_dir
    candidates = []
    for name in os.listdir(run_dir) if os.path.isdir(run_dir) else []:
        m = _STEP_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(run_dir, name)):
            candidates.append(int(m.group(1)))
    if not candidates:
        return None
    return os.path.join(run_dir, f"global_step_{max(candidates)}")


def _resolve_resume_path(base_dir: str, resume_path: str | None) -> tuple[str, str] | None:
    """Resolve a resume request to ``(run_dir, step_dir)``.

    Explicit ``resume_path`` (runs raising on invalid input): the path may BE a
    ``global_step_N`` dir or a run dir holding them. ``resume_path=None``:
    auto-detect the lexicographically-last run dir under ``base_dir`` that
    holds a checkpoint (timestamp names sort chronologically); returns None
    when nothing is found.
    """
    if resume_path:
        resume_path = os.path.abspath(os.path.expanduser(str(resume_path)))
        if _STEP_DIR_RE.match(os.path.basename(resume_path)):
            run_dir, step_dir = os.path.dirname(resume_path), resume_path
        else:
            run_dir, step_dir = resume_path, _latest_step_dir(resume_path)
        if step_dir is None or not os.path.isdir(step_dir):
            raise FileNotFoundError(
                f"trainer.resume.resume_path={resume_path!r} holds no global_step_* "
                "checkpoint. Point it at a run's timestamp dir (RL/<exp>/<YYYYMMDD_HHMMSS>) "
                "or a specific .../global_step_N dir."
            )
        return run_dir, step_dir

    if not os.path.isdir(base_dir):
        return None
    for name in sorted(os.listdir(base_dir), reverse=True):
        run_dir = os.path.join(base_dir, name)
        if os.path.isdir(run_dir):
            step_dir = _latest_step_dir(run_dir)
            if step_dir is not None:
                return run_dir, step_dir
    return None


def apply_checkpoint_layout(config):
    """Rewire ``config.trainer`` to the Transfusion-style run layout (in place).

    Fresh launch: ``default_local_dir`` gets a ``/YYYYmmdd_HHMMSS`` subdir
    appended and resume is disabled. ``trainer.resume.enabled: true``: the
    request is resolved (see :func:`_resolve_resume_path`), ``default_local_dir``
    is re-pointed at the resumed run's dir, "global_step_")
    warning is printed and a FRESH run starts (verl ``auto`` semantics).

    ``trainer.save_optimizer`` (when present) overrides
    ``actor_rollout_ref.<actor|critic>.checkpoint.save_contents``.
    """
    trainer = config.trainer
    base_dir = str(trainer.default_local_dir)  # interpolations resolve on access

    resume_cfg = OmegaConf.select(config, "trainer.resume")
    resume_enabled = bool(resume_cfg.get("enabled", False)) if resume_cfg else False
    resume_path = resume_cfg.get("resume_path") if resume_cfg else None
    if resume_path in ("", "null", "None"):
        resume_path = None

    if resume_enabled:
        resolved = _resolve_resume_path(base_dir, resume_path)
        if resolved is None:
            print(
                f"[ckpt_layout] trainer.resume.enabled=true but no run with a checkpoint "
                f"found under {base_dir} — starting FRESH (verl 'auto' semantics)."
            )
            resume_enabled = False
        else:
            run_dir, step_dir = resolved
            trainer.default_local_dir = run_dir
            trainer.resume_mode = "resume_path"
            trainer.resume_from_path = step_dir
            print(f"[ckpt_layout] RESUME run dir : {run_dir}")
            print(f"[ckpt_layout] RESUME from    : {step_dir}")

    if not resume_enabled:
        run_dir = os.path.join(base_dir, datetime.now().strftime(RUN_TS_FORMAT))
        trainer.default_local_dir = run_dir
        trainer.resume_mode = "disable"
        trainer.resume_from_path = None
        print(f"[ckpt_layout] FRESH run dir  : {run_dir}")

    os.makedirs(trainer.default_local_dir, exist_ok=True)

    save_optimizer = OmegaConf.select(config, "trainer.save_optimizer")
    if save_optimizer is not None:
        contents = FULL_SAVE_CONTENTS if bool(save_optimizer) else WEIGHTS_ONLY_SAVE_CONTENTS
        for role in ("actor", "critic"):
            if OmegaConf.select(config, f"actor_rollout_ref.{role}.checkpoint") is not None:
                OmegaConf.update(config, f"actor_rollout_ref.{role}.checkpoint.save_contents", list(contents))
                # IMPORTANT (resume robustness): on a RESUME, the saved optimizer
                # shards were produced by a *prior* run whose model/optimizer param
                # layout can differ from the one a fresh process builds (observed:
                # grpo 734 params -> pro_2 735; FSDP use_orig_params grouping also
                # drifts). verl's FSDPCheckpointManager does a STRICT
                # optimize().load_state_dict() and hard-crashes with
                # "parameter group doesn't match the size of optimizer's group".
                # So on resume we always LOAD the optimizer as weights-only
                # ([model, extra]) and never feed optimizer shards into a
                # mismatched optimizer. This is safe here because the model trains
                # from a from-scratch AdamW either way — and the prior runs saved
                # EMPTY optimizer moment state (state_len=0), so there is nothing
                # worth re-attaching. New checkpoints continue to SAVE the full
                # [model, optimizer, extra] for future verbatim resumability.
                if resume_enabled:
                    OmegaConf.update(
                        config,
                        f"actor_rollout_ref.{role}.checkpoint.load_contents",
                        list(WEIGHTS_ONLY_SAVE_CONTENTS),
                    )
        print(f"[ckpt_layout] save_optimizer={bool(save_optimizer)} -> checkpoint.save_contents={contents}")
        if resume_enabled:
            print(
                "[ckpt_layout] resume -> checkpoint.load_contents forced to "
                f"{WEIGHTS_ONLY_SAVE_CONTENTS} (weights-only optimizer resume, "
                "avoids param-group mismatch crash)"
            )
        elif not save_optimizer:
            print(
                "[ckpt_layout] WARNING: save_optimizer=false — this run's NEW checkpoints will be "
                "weights-only and NOT verbatim-resumable."
            )
    return config
