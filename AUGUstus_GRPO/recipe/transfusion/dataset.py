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

"""Transfusion data adapter for verl (``data.custom_cls``).

verl loads a custom dataset through ``data.custom_cls.{path,name}`` and always
instantiates it as ``cls(data_files, tokenizer, config, processor, max_samples)``
(a subclass of ``torch.utils.data.Dataset``). This adapter wraps the existing
:class:`transfusion.dataset.TransfusionHFCacheDataset` so the *same* NAVSIM
``hf_cache_*_CoT_Solution`` data used for supervised training also feeds the RL
loop — no data conversion required.

Each ``__getitem__`` returns a verl-friendly row:

* ``transfusion_inputs`` — a python dict (kept as a *non-tensor* object so verl's
  ``collate_fn`` does not try to stack the variable-length, left-padded
  multimodal tensors). The custom :mod:`recipe.transfusion.rollout` re-collates
  these per micro-batch with :class:`transfusion.dataset.TransfusionCollator`.
* ``actions`` — the ``[8, 3]`` ground-truth trajectory tensor (fixed shape,
  safely stackable) for the L2 reward.
* ``data_source`` / ``reward_model`` / ``extra_info`` — verl bookkeeping; the
  ``reward_model.ground_truth`` dict carries everything the reward needs
  (``gt_trajectory``, ``gt_cot_text``, ``frame_id``, ``metric_cache_path``).
* ``dummy_tensor`` — a 1-element tensor so ``DataProto.batch`` is never empty
  (mirrors upstream ``RLHFDataset``).

Structural knobs (``traj_token`` / ``action_dim`` / ``enable_cot`` /
``cot_components`` / ``prompt_file``) are sourced from the model's own
``$TRANSFUSION_CONFIG`` so they never diverge from the checkpoint; a matching key
under ``data.transfusion`` still overrides. The PDMS ``metric_cache`` is chosen
per split (``data.metric_cache_train`` in training, ``data.metric_cache_test`` in
validation) — see ``config/transfusion_navsim_qwen3p5.yaml``.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_transfusion_on_path(transfusion_root: Optional[str] = None) -> str:
    """Make the ``transfusion`` package importable; return the resolved root."""
    candidates: List[str] = []
    if transfusion_root:
        candidates.append(transfusion_root)
    if os.environ.get("TRANSFUSION_ROOT"):
        candidates.append(os.environ["TRANSFUSION_ROOT"])
    here = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.abspath(os.path.join(here, "..", "..", ".."))  # .../Code
    candidates.append(os.path.join(code_dir, "Transfusion"))
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "model")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
    raise ImportError(
        "Could not locate the 'model' package. Set data.transfusion.transfusion_root "
        "or the TRANSFUSION_ROOT environment variable."
    )


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    getter = getattr(node, "get", None)
    if callable(getter):
        val = getter(key, default)
        return default if val is None else val
    return getattr(node, key, default)


def _load_arch_cfg() -> Dict[str, Any]:
    """Load the transfusion ARCHITECTURE config (``$TRANSFUSION_CONFIG``) as a
    plain dict -- the SAME file the model is built from.

    Structural knobs (``traj_token`` / ``action_dim`` / CoT settings /
    ``prompt_file``) are read from here so the RL dataset can never diverge from
    the checkpoint and they are NOT duplicated in the recipe config. Returns
    ``{}`` if the file is missing/unreadable (callers fall back to
    ``data.transfusion.*`` then hard defaults).
    """
    path = os.environ.get("TRANSFUSION_CONFIG")
    if not path or not os.path.isfile(path):
        return {}
    try:
        from omegaconf import OmegaConf

        # resolve=False: only literal structural keys are read; avoids resolving
        # unrelated ${...} interpolations in the supervised training config.
        return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore[return-value]
    except Exception as e:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning("Could not read $TRANSFUSION_CONFIG (%s): %s", path, e)
        return {}


def _is_train_split(config: Any, cache_dir: str) -> bool:
    """True when ``cache_dir`` matches ``data.train_files`` (vs. the val split)."""
    train_files = _cfg_get(config, "train_files", None)

    def _first(x):
        return x[0] if isinstance(x, (list, tuple)) and x else x

    return bool(train_files) and str(_first(train_files)) == str(cache_dir)


def _resolve_metric_cache(config: Any, cache_dir: str) -> Optional[str]:
    """Pick the split-appropriate NAVSIM PDMS cache for THIS dataset instance.

    verl builds the dataset once per split (``data.train_files`` then
    ``data.val_files``); return ``data.metric_cache_train`` for the train split and
    ``data.metric_cache_test`` otherwise, so the PDMS reward is looked up from the
    RIGHT split during BOTH training and validation.
    """
    is_train_split = _is_train_split(config, cache_dir)
    return _cfg_get(config, "metric_cache_train" if is_train_split else "metric_cache_test", None)


def _resolve_metric_cache_v2(config: Any, cache_dir: str) -> Optional[str]:
    """Pick the split-appropriate NAVSIM **v2** cache (official EPDMS) for THIS
    dataset instance — the ``*_v2`` counterpart of :func:`_resolve_metric_cache`.
    May be ``None`` when the run has no v2 caches configured; EPDMS is computed
    only when explicitly enabled (``tr.epdms_enabled``), so a run without EPDMS
    never touches these paths.
    """
    is_train_split = _is_train_split(config, cache_dir)
    return _cfg_get(config, "metric_cache_train_v2" if is_train_split else "metric_cache_test_v2", None)


# The RL dataset below is a thin adapter around the supervised Transfusion HF cache
# dataset (verl sample schema + reward-model ground truth).
class TransfusionRLDataset(Dataset):
    """verl ``custom_cls`` wrapper around ``TransfusionHFCacheDataset``."""

    def __init__(
        self,
        data_files,
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        # ---- Transfusion-specific sub-config (data.transfusion.*) ----
        tcfg = _cfg_get(config, "transfusion", {}) or {}
        self.tcfg = tcfg
        transfusion_root = _cfg_get(tcfg, "transfusion_root", None)
        _ensure_transfusion_on_path(transfusion_root)

        from model.dataset import TransfusionHFCacheDataset

        # data_files is the hf_cache directory (train_files / val_files in config).
        cache_dir = data_files[0] if isinstance(data_files, (list, tuple)) else data_files

        if processor is None:
            from transformers import AutoProcessor

            qwen_model_name = _cfg_get(tcfg, "qwen_model_name", None)
            if qwen_model_name is None:
                raise ValueError(
                    "processor is None and data.transfusion.qwen_model_name is unset; "
                    "cannot build the multimodal processor."
                )
            processor = AutoProcessor.from_pretrained(qwen_model_name)
            self.processor = processor

        # ---- Structural knobs ----
        # Sourced from the SAME transfusion architecture config the model is built
        # from ($TRANSFUSION_CONFIG: trajectory.* / cot.* / data.prompt_file), so
        # they can never diverge from the checkpoint and are NOT duplicated in the
        # recipe config. An explicit data.transfusion.<key> still wins (override).
        arch = _load_arch_cfg()
        arch_traj = arch.get("trajectory", {}) or {}
        arch_cot = arch.get("cot", {}) or {}
        arch_data = arch.get("data", {}) or {}

        def _pick(key: str, arch_val: Any, default: Any) -> Any:
            override = _cfg_get(tcfg, key, None)
            if override is not None:
                return override
            return arch_val if arch_val is not None else default

        self.traj_token = int(_pick("traj_token", arch_traj.get("traj_token"), 8))
        self.action_dim = int(_pick("action_dim", arch_traj.get("action_dim"), 3))
        self.enable_cot = bool(_pick("enable_cot", arch_cot.get("enable_cot"), True))
        cot_components = _pick("cot_components", arch_cot.get("cot_components"), ["prediction"])
        if cot_components is not None and not isinstance(cot_components, list):
            cot_components = list(cot_components)
        prompt_file = _pick("prompt_file", arch_data.get("prompt_file"), None)

        # is_training -> TransfusionHFCacheDataset mode. True = yield the supervised
        # targets (GT trajectory + CoT teacher-forcing tokens) that build the
        # reward's ground truth; False = inference-only input assembly.
        self.is_training = bool(_cfg_get(tcfg, "is_training", True))
        self.data_source = _cfg_get(tcfg, "data_source", "navsim_transfusion")
        self.reward_weights = _cfg_get(tcfg, "reward_weights", None)
        # Split-aware PDMS cache (train cache in training, test cache in validation).
        self.metric_cache_path = _resolve_metric_cache(config, cache_dir)
        # Same split-aware resolution for the v2 metric cache (official
        # two-frame EPDMS, see _resolve_metric_cache_v2); None when the run
        # has no v2 caches configured, in which case EPDMS stays disabled.
        self.metric_cache_v2_path = _resolve_metric_cache_v2(config, cache_dir)
        # Keep decoded PIL video frames ONLY for the validation split (bounded by
        # data.val_max_samples, so RAM stays small) so recipe.transfusion.sample_logging
        # can show "what the model is currently looking at" (last video frame [+ bbox
        # overlay]) during test/val -- mirroring supervised Transfusion's
        # log_eval_samples. Always dropped for the (much larger, n-times-replicated)
        # train split to keep RL rollout RAM down.
        self.keep_raw_frames = not _is_train_split(config, cache_dir)

        # Pass optional dataset arguments only when the constructor supports them.
        _inner_params = inspect.signature(TransfusionHFCacheDataset.__init__).parameters
        inner_kwargs: Dict[str, Any] = dict(
            cache_dir=cache_dir,
            processor=processor,
            prompt_file=prompt_file,
            traj_token=self.traj_token,
            action_dim=self.action_dim,
            is_training=self.is_training,
            enable_cot=self.enable_cot,
            cot_components=cot_components,
        )
        if "state_config" in _inner_params:
            inner_kwargs["state_config"] = _cfg_get(tcfg, "state_config", None)
        if "pdms_loss_config" in _inner_params:
            inner_kwargs["pdms_loss_config"] = _cfg_get(tcfg, "pdms_loss_config", None)
        self.inner = TransfusionHFCacheDataset(**inner_kwargs)

        # Optional sub-sampling, mirroring verl's max_samples semantics.
        self._indices = list(range(len(self.inner)))
        if max_samples is not None and max_samples > 0 and max_samples < len(self.inner):
            rng = np.random.default_rng(_cfg_get(config, "seed", 42) or 42)
            self._indices = sorted(rng.choice(len(self.inner), size=max_samples, replace=False).tolist())

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.inner[self._indices[idx]]

        # Heavy logging-only payload (raw PIL frames): kept only for the val split
        # (self.keep_raw_frames), dropped otherwise to keep RL rollout RAM down.
        raw_frames = sample.pop("_raw_video_frames", None)

        actions = sample.get("actions")
        if actions is None:
            actions = torch.zeros(self.traj_token, self.action_dim, dtype=torch.float32)

        gt_cot_text = sample.get("_reasoning_trace", "") or ""
        frame_id = sample.get("_frame_id", "") or ""

        # The model inputs (variable length / left-padded) travel as a non-tensor
        # python object so verl's default collate_fn leaves them untouched.
        transfusion_inputs: Dict[str, Any] = {
            k: v
            for k, v in sample.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }

        ground_truth = {
            "gt_trajectory": actions.detach().cpu().numpy().tolist(),
            "gt_cot_text": gt_cot_text,
            "frame_id": frame_id,
            "metric_cache_path": self.metric_cache_path,
            "metric_cache_v2_path": self.metric_cache_v2_path,
        }
        if self.reward_weights is not None:
            ground_truth["weights"] = (
                dict(self.reward_weights) if hasattr(self.reward_weights, "items") else self.reward_weights
            )

        row: Dict[str, Any] = {
            "transfusion_inputs": transfusion_inputs,
            "actions": actions.float(),
            "data_source": self.data_source,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"index": int(idx), "frame_id": frame_id, "metric_cache_path": self.metric_cache_path},
            "index": int(idx),
            # keep DataProto.batch non-empty even before rollout populates tensors
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
        }
        if self.keep_raw_frames:
            # Top-level (NOT nested in transfusion_inputs) so the sync driver /
            # reward manager can read it straight off non_tensor_batch. The key
            # must exist for EVERY row of the val split (verl's collate_fn
            # asserts uniform non-tensor keys/length), but the actual decoded
            # PIL frames are CAPPED to the first few samples (dataset order):
            # the sample logger only ever renders the leading
            # log_val_generations (10) samples of a validation pass, and
            # validation_shuffle=false keeps that the first chunk -- every
            # other sample carries an EMPTY list (falsy -> the logger skips
            # it), otherwise decoded frames for ALL ~12k val samples would
            # cost GBs of pickle/RPC per val chunk (2026-07-29 val-speed fix).
            row["_raw_video_frames"] = (
                raw_frames
                if (raw_frames is not None and idx < _RAW_FRAMES_MAX_SAMPLES)
                else []
            )
        return row


# How many leading val samples keep decoded PIL frames (see __getitem__).
# >= trainer.log_val_generations (10); the sample logger never renders more.
_RAW_FRAMES_MAX_SAMPLES = 16

__all__ = ["TransfusionRLDataset"]
