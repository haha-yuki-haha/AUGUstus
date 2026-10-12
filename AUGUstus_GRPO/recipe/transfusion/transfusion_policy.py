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

"""Build / load the Transfusion policy and wrap its rollout + training calls.

This module replaces verl's default HuggingFace causal-LM with the
``TransfusionQwen3p5`` diffusion driving planner. It deliberately **reuses**
Transfusion's own ``build_model`` (and the checkpoint-loading logic of
``load_model``) so the large, easily-desynchronised kwargs mapping stays in one
place — see ``AUGUstus_SFT/train_qwen3p5.py`` and ``AUGUstus_SFT/test_qwen3p5.py``.

* :func:`load_transfusion_config` — read ``configs/navsim_qwen3p5.yaml`` into a
  plain dict (with overrides), exactly the structure ``build_model`` expects.
* :func:`build_transfusion_model` — construct the (trainable) model and,
  optionally, load RL-init weights from a supervised checkpoint.
* :class:`TransfusionPolicy` — a thin wrapper exposing:
    - ``rollout(...)`` — diffusion sampling via ``generate_trajectory`` returning
      ``(trajectories[B,8,3], cot_texts)`` for the reward;
    - ``forward(...)`` — the per-``t`` diffusion/LM training objective, used as
      the behaviour anchor / log-prob surrogate by the actor update.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_transfusion_on_path(transfusion_root: Optional[str] = None) -> str:
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
        "Could not locate the 'model' package. Set TRANSFUSION_ROOT or pass transfusion_root."
    )


def load_transfusion_config(
    config_path: str, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load a Transfusion YAML config into a resolved plain dict.

    Args:
        config_path: path to e.g. ``configs/navsim_qwen3p5.yaml``.
        overrides: optional nested dict merged on top (e.g. to point
            ``data.train_data_path`` at the RL split or swap ``model.qwen_model_name``).
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    resolved = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return resolved


def _apply_model_compat_aliases(model: torch.nn.Module) -> None:
    """Provide default flags for optional conditioning systems.

    Missing ViT, scene, and state conditioning flags default to False.
    """
    for _flag in ("vit_feature_enabled", "scene_cond_enabled", "state_cond_enabled"):
        if not hasattr(model, _flag):
            setattr(model, _flag, False)


def _load_state_dict_into(model: torch.nn.Module, ckpt_path: str) -> Tuple[int, int]:
    """Load a (possibly sharded) Transfusion checkpoint into ``model`` in place.

    Mirrors the state-dict handling in ``AUGUstus_SFT/test_qwen3p5.py::load_model``
    (prefix stripping, fp32->bf16 cast, ``strict=False``) but keeps the model's
    *training* freeze flags intact (``build_model`` already set them).
    """
    cp = Path(ckpt_path)
    # Search the checkpoint dir AND a nested ``pytorch_model/`` subdir, since
    # Transfusion saves the full model there as a single ``model.safetensors``
    # (e.g. .../best/pytorch_model/model.safetensors). Covers both single-file
    # and sharded (``model-*.safetensors``) layouts in either location.
    search_dirs = [cp]
    if (cp / "pytorch_model").is_dir():
        search_dirs.append(cp / "pytorch_model")

    def _load_single(fp: Path):
        if fp.suffix == ".bin":
            return torch.load(fp, map_location="cpu", weights_only=False)
        from safetensors.torch import load_file

        return load_file(str(fp))

    sd = None
    # 1) single-file checkpoints (model.safetensors / pytorch_model.bin)
    for d in search_dirs:
        for name in ("model.safetensors", "pytorch_model.bin"):
            fp = d / name
            if fp.exists():
                sd = _load_single(fp)
                break
        if sd is not None:
            break
    # 2) sharded safetensors (model-*.safetensors)
    if sd is None:
        from safetensors.torch import load_file

        for d in search_dirs:
            shards = sorted(d.glob("model-*.safetensors"))
            if shards:
                sd = {}
                for shard in shards:
                    sd.update(load_file(str(shard)))
                break
    if sd is None:
        raise FileNotFoundError(f"No checkpoint weights found under {ckpt_path}")

    # Strip 'module.'/'model.' prefixes iteratively.
    def _strip(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in ("module.", "model."):
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    sd = {_strip(k): v for k, v in sd.items()}
    for k in list(sd.keys()):
        if isinstance(sd[k], torch.Tensor) and sd[k].dtype == torch.float32:
            sd[k] = sd[k].to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del sd
    return len(missing), len(unexpected)


def _apply_loading_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Redirect the qwen backbone / processor / action-norm stats off the original
    supervised-training mount so a trained model loads purely from its own
    checkpoint dir (the ``best/`` the user provides).

    The supervised ``$TRANSFUSION_CONFIG`` (config_original.yaml) hard-codes the
    original Qwen3.5-2B backbone and the training-data mount. We override, from
    env vars set in the recipe config / run scripts:

    * ``TRANSFUSION_QWEN_MODEL_NAME`` -> ``model.qwen_model_name``. Used inside
      ``build_model`` for BOTH ``Qwen3_5AdaForConditionalGeneration.from_pretrained``
      (backbone architecture) and ``AutoProcessor.from_pretrained`` (tokenizer /
      image+video processor). Point it at the checkpoint's ``pytorch_model/`` HF
      dir. The backbone *weights* built here are overwritten by the
      ``TRANSFUSION_CKPT`` load (it carries the full ``qwen_model.*`` + head state
      dict), so this only needs a loadable HF dir + a working processor.
    * ``TRANSFUSION_STATS_FILE`` -> ``normalization.stats_file``. The checkpoint
      already carries the real ``action_q01/q99/dim_weights`` (they overwrite
      whatever is loaded here), so this only needs to be an accessible stats JSON
      with per-timestep quantiles — otherwise ``build_model`` would fall back to
      reading the original train-data mount to (re)compute them.
    """
    qwen = os.environ.get("TRANSFUSION_QWEN_MODEL_NAME", "").strip()
    stats = os.environ.get("TRANSFUSION_STATS_FILE", "").strip()
    if not qwen and not stats:
        return cfg
    import copy

    cfg = copy.deepcopy(cfg)
    if qwen:
        cfg.setdefault("model", {})["qwen_model_name"] = qwen
        logger.info("Self-contained load: model.qwen_model_name -> %s", qwen)
    if stats:
        cfg.setdefault("normalization", {})["stats_file"] = stats
        logger.info("Self-contained load: normalization.stats_file -> %s", stats)
    return cfg


def build_transfusion_model(
    cfg: Dict[str, Any],
    ckpt_path: Optional[str] = None,
    transfusion_root: Optional[str] = None,
):
    """Build the (trainable) ``TransfusionQwen3p5`` and optionally load RL-init weights.

    ``cfg`` is the dict returned by :func:`load_transfusion_config`. We call the
    upstream ``build_model`` so the model is byte-for-byte the one used in
    supervised training; passing ``ckpt_path`` then loads a TRAINED Transfusion
    checkpoint (supervised SFT model or a previous RL model) on top of the qwen
    backbone — this is the "load a trained model" hook, wired from the config /
    run script via the ``TRANSFUSION_CKPT`` env var.
    """
    _ensure_transfusion_on_path(transfusion_root)
    from train_qwen3p5 import build_model  # reuse the canonical kwargs mapping

    # Redirect backbone / processor / stats to the checkpoint's own dir so we do
    # NOT touch the original Qwen3.5-2B backbone or the training-data mount.
    cfg = _apply_loading_overrides(cfg)
    # Treat empty / whitespace paths (e.g. an unset ${oc.env:TRANSFUSION_CKPT,""})
    # as "no checkpoint" -> cold-start from the backbone.
    ckpt_path = ckpt_path.strip() if isinstance(ckpt_path, str) else ckpt_path
    # A full Transfusion checkpoint supplies the backbone weights, so skip the
    # separate backbone weight load. Without a checkpoint, load the HF backbone.
    # TRANSFUSION_DISABLE_FAST_BACKBONE=1 enables separate backbone loading.
    fast_backbone = bool(ckpt_path) and os.environ.get(
        "TRANSFUSION_DISABLE_FAST_BACKBONE", ""
    ).strip().lower() not in ("1", "true", "yes", "on")
    if fast_backbone:
        import copy

        cfg = copy.deepcopy(cfg)
        cfg.setdefault("model", {})["skip_backbone_weight_load"] = True
    model = build_model(cfg)
    # Provide optional conditioning flags expected by the RL adapter.
    _apply_model_compat_aliases(model)
    # RL-only: let the WITH-GRAD denoiser re-scoring use the grad-safe KV-cache clone
    # (reassign instead of the stock in-place copy_ that breaks the FLA backward).
    # SFT models are built via train_qwen3p5.build_model directly and never set this,
    # so supervised training keeps the original in-place cache untouched.
    try:
        model.rl_grad_safe_cache = True
    except Exception:
        pass
    if ckpt_path:
        n_missing, n_unexpected = _load_state_dict_into(model, ckpt_path)
        logger.info(
            "Loaded trained Transfusion checkpoint from %s (missing=%d, unexpected=%d). "
            "A large 'missing' count usually means $TRANSFUSION_CONFIG does not match "
            "the checkpoint's architecture.",
            ckpt_path, n_missing, n_unexpected,
        )
        if fast_backbone and (n_missing or n_unexpected):
            # Under the fast path the HF backbone was built from CONFIG ONLY:
            # any key the Transfusion ckpt does not cover stays ZERO (not the
            # stock HF weight / stock random init the legacy path would leave).
            logger.warning(
                "Fast backbone build: %d missing / %d unexpected keys loading %s — those "
                "backbone weights are ZERO now. If this is intended (partial/arch-mismatched "
                "ckpt is NOT supported by the fast path), relaunch with "
                "TRANSFUSION_DISABLE_FAST_BACKBONE=1 for the legacy full-HF-load path.",
                n_missing, n_unexpected, ckpt_path,
            )
    else:
        logger.info("No TRANSFUSION_CKPT set -> cold-starting Transfusion from the backbone only.")

    # RL-only OPTIONAL ViT freeze (default off): reuses the SFT freeze mechanism
    # (TransfusionQwen3p5's freeze_vision_encoder) but applied AFTER the trained
    # checkpoint load, driven by the TRANSFUSION_FREEZE_VIT env var so it reaches
    # every build site (actor engine, rollout policy, KL reference) without
    # touching verl's strict model config. NOTE (FSDP1, use_orig_params=False):
    # mixing frozen + trainable params inside one FSDP flat-param unit raises at
    # wrap time; if that happens set
    # actor_rollout_ref.actor.fsdp_config.use_orig_params: true in the YAML.
    if os.environ.get("TRANSFUSION_FREEZE_VIT", "").strip().lower() in ("1", "true", "yes", "on"):
        n_frozen = 0
        for p in model._visual.parameters():
            p.requires_grad_(False)
            n_frozen += 1
        logger.info("TRANSFUSION_FREEZE_VIT=on -> froze %d ViT (visual tower) parameters.", n_frozen)
    return model


class TransfusionPolicy:
    """Thin rollout/forward wrapper around a built ``TransfusionQwen3p5`` model.

    Not tied to any particular distributed backend — it operates on an already
    placed ``model`` (single GPU, FSDP-wrapped, or DDP). The verl engine /
    rollout glue (:mod:`recipe.transfusion.rollout`,
    :mod:`recipe.transfusion.model_engine`) delegates the actual model calls here.
    """

    def __init__(self, model, processor, traj_token: int = 8, action_dim: int = 3, cfg_scale: float = 1.0):
        _ensure_transfusion_on_path()
        from model.dataset import TransfusionCollator

        self.model = model
        self.processor = processor
        self.traj_token = traj_token
        self.action_dim = action_dim
        self.cfg_scale = float(cfg_scale)
        self.collator = TransfusionCollator(processor, traj_token=traj_token, action_dim=action_dim)
        self._adapter = None

    # -- input assembly -----------------------------------------------------
    def collate(self, transfusion_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Left-pad a list of per-sample model-input dicts via ``TransfusionCollator``."""
        # The collator mutates/pops keys, so feed it shallow copies.
        return self.collator([dict(x) for x in transfusion_inputs])

    @staticmethod
    def _to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        return out

    # -- rollout (inference) ------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        transfusion_inputs: List[Dict[str, Any]],
        *,
        generate_cot: bool = True,
        cot_temperature: float = 0.9,
        cot_top_p: float = 0.95,
        cot_top_k: int = 100,
        return_normed: bool = False,
        num_steps: Optional[int] = None,
        device: Optional[Any] = None,
        cot_max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run diffusion sampling for a micro-batch.

        Returns ``{"trajectories": np[B,8,3], "cot_texts": List[str],
        "cot_ids": List[List[int]]}``. Trajectories are de-normalised (metres /
        radians), exactly what the L2 / PDMS rewards expect.

        ``num_steps`` overrides ``model.diffusion.num_inference_steps`` for this
        call only (the supervised checkpoint trains with ``num_inference_steps=1``,
        i.e. one-step denoising; RL rollouts typically want a multi-step denoising
        chain for the action). The attribute is restored in a ``finally`` block so
        the model's own config is never mutated permanently — no Transfusion core
        file is edited.
        """
        model = self.model
        device = device or next(model.parameters()).device
        import time as _time
        _t0 = _time.time()
        batch = self._to_device(self.collate(transfusion_inputs), device)
        _t_collate = _time.time() - _t0

        diffusion = getattr(model, "diffusion", None)
        saved_steps = getattr(diffusion, "num_inference_steps", None)
        override = (
            num_steps is not None
            and diffusion is not None
            and hasattr(diffusion, "num_inference_steps")
            and int(num_steps) != int(saved_steps)
        )
        if override:
            diffusion.num_inference_steps = int(num_steps)
        try:
            # Model-vintage compatibility: the ORIGINAL Code/Transfusion
            # generate_trajectory accepted `use_ddim` (DDPM/DDIM sampler switch) and
            # `state_features` (StateCondEncoder experiment); the cleaned
            # AUGUstus_SFT model removed both (flow-matching only, no state
            # conditioning). Drop kwargs the installed model does not accept.
            import inspect as _inspect

            _gt_params = _inspect.signature(model.generate_trajectory).parameters
            _gt_kwargs: Dict[str, Any] = dict(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
                second_per_grid_ts=batch.get("second_per_grid_ts"),
                generate_cot=generate_cot,
                cot_temperature=cot_temperature,
                cot_top_p=cot_top_p,
                cot_top_k=cot_top_k,
                return_normed=return_normed,
                max_cot_tokens=cot_max_tokens,
            )
            if "state_features" in _gt_params:
                _gt_kwargs["state_features"] = batch.get("state_features")
            if "use_ddim" in _gt_params:
                _gt_kwargs["use_ddim"] = True
            result = model.generate_trajectory(**_gt_kwargs)
        finally:
            if override:
                diffusion.num_inference_steps = saved_steps
        if return_normed:
            traj, cot_ids, _normed, _denoised = result
        else:
            traj, cot_ids = result

        trajectories = traj.float().cpu().numpy()
        cot_texts: List[str] = []
        cot_id_lists: List[List[int]] = []
        if cot_ids is not None:
            for i in range(cot_ids.shape[0]):
                ids = cot_ids[i].tolist()
                cot_id_lists.append(ids)
                cot_texts.append(
                    self.processor.tokenizer.decode(cot_ids[i], skip_special_tokens=True)
                )
        else:
            cot_texts = [""] * trajectories.shape[0]
            cot_id_lists = [[] for _ in range(trajectories.shape[0])]

        logger.warning(
            "[time] rollout(val): collate=%.1fs generate=%.1fs total=%.1fs (B=%d)",
            _t_collate, _time.time() - _t0 - _t_collate, _time.time() - _t0,
            trajectories.shape[0],
        )
        return {"trajectories": trajectories, "cot_texts": cot_texts, "cot_ids": cot_id_lists}

    # -- training objective (behaviour anchor / surrogate logprob) ----------
    def forward(self, transfusion_inputs: List[Dict[str, Any]], device: Optional[Any] = None):
        """Per-``t`` diffusion/LM objective; returns the model's ``TransfusionOutput``.

        Used by the actor engine as the differentiable surrogate around which the
        policy-gradient update is built (the diffusion/CoT loss provides the
        log-likelihood proxy for the sampled action). See ``docs/ADAPTATION.md``.
        """
        device = device or next(self.model.parameters()).device
        batch = self._to_device(self.collate(transfusion_inputs), device)
        return self.model(**batch)

    # -- diffusion-RL adapter (Flow-GRPO / DiffusionNFT) --------------------
    @property
    def adapter(self):
        """Lazy :class:`TransfusionDiffusionAdapter` over the wrapped model."""
        if self._adapter is None:
            from recipe.transfusion.algorithms.diffusion_adapter import (
                TransfusionDiffusionAdapter,
            )

            self._adapter = TransfusionDiffusionAdapter(self.model, cfg_scale=self.cfg_scale)
        return self._adapter

    @torch.no_grad()
    def _decode_cot(self, cot_ids, batch_size: int):
        """Split a ``[B, C]`` CoT id tensor into per-sample id lists + decoded texts."""
        if cot_ids is None:
            return [""] * batch_size, [[] for _ in range(batch_size)]
        texts: List[str] = []
        id_lists: List[List[int]] = []
        for i in range(cot_ids.shape[0]):
            ids = cot_ids[i].tolist()
            id_lists.append(ids)
            texts.append(self.processor.tokenizer.decode(cot_ids[i], skip_special_tokens=True))
        return texts, id_lists

    @torch.no_grad()
    def sample_grpo(
        self,
        transfusion_inputs: List[Dict[str, Any]],
        *,
        num_steps: int = 10,
        noise_level=0.7,
        noise_schedule: str = "flow_grpo",
        generate_cot: bool = True,
        cot_temperature: float = 0.9,
        cot_top_p: float = 0.95,
        cot_top_k: int = 100,
        cot_max_tokens: Optional[int] = None,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Combined GRPO data collection: SDE trajectory + post-hoc CoT.

        ``cot_max_tokens`` caps the number of generated CoT CONTENT tokens
        (forwarded to ``_generate_cot_after_action``; the RL rollout passes
        ``response_length - 1`` so the forced EOT column stays inside the
        response window kept by verl). Sampling stops on the SFT stop token
        <|eot|>; the sampled EOT is kept in the returned ids so the RL loss
        scores P(EOT | prefix).

        Samples the action with the reverse-time SDE (recording the per-step
        Gaussian transitions the closed-form JiT-GRPO ratio re-scores) and then,
        conditioned on the *clean* sampled trajectory, autoregressively generates
        the CoT (recording its token ids so the token-probability ratio can be
        re-scored). Both share one multimodal context encode.

        Returns ``{"trajectories": np[B,8,3] (de-normalised), "x0_normed":
        tensor[B,T,D], "record": {...SDE per-step tensors...}, "batch": collated
        inputs, "cot_texts": List[str], "cot_ids": List[List[int]]}``.
        """
        device = device or next(self.model.parameters()).device
        import time as _time
        _t0 = _time.time()
        batch = self._to_device(self.collate(transfusion_inputs), device)
        ctx = self.adapter.prepare_context(batch, grad=False)
        _t_ctx = _time.time() - _t0
        record = self.adapter.sample_sde_with_logprob(
            ctx, num_steps=num_steps,
            noise_level=noise_level,
            noise_schedule=noise_schedule,
        )
        _t_sde = _time.time() - _t0 - _t_ctx
        # Clamp the SDE's clean prediction to the normalised action range before
        # de-normalising / conditioning the CoT, matching the model's own inference
        # (``torch.clamp(pred_x0_norm, -1, 1)``). The raw reverse-SDE output can
        # overshoot [-1, 1] (few steps + injected noise) and de-normalise into
        # wildly out-of-range waypoints; the per-step SDE record keeps the RAW
        # states, so the closed-form action ratio is unaffected.
        x0_normed = record["x0_normed"].clamp(-1.0, 1.0)  # [B, T, D] normalised clean traj
        trajectories = self.adapter.denormalize(x0_normed).float().cpu().numpy()

        # CoT conditioned on the clean sampled action (action_first explanation).
        # record_logp=True: the sampler also returns the per-token RAW-distribution
        # log-probs / entropies of the generated CoT — the behaviour-policy
        # ``old_log_probs`` / ``entropys`` for the PPO ratio, captured for free
        # during sampling so the separate teacher-forced recompute phase
        # (timing_s/old_log_prob) can be skipped entirely.
        cot_ids = None
        cot_logps = None
        cot_ents = None
        if generate_cot and getattr(self.model, "use_action_first", False) \
                and getattr(self.model, "action_first_generate_cot", False):
            cot_ids, cot_logps, cot_ents = self.model._generate_cot_after_action(
                ctx["ctx_embeds"], x0_normed, ctx["prefix_pos"], ctx.get("attention_mask"),
                ctx["B"], device, ctx["prefix_adarms_cond"],
                temperature=cot_temperature, top_p=cot_top_p, top_k=cot_top_k,
                record_logp=True, max_cot_tokens=cot_max_tokens,
            )
        cot_texts, cot_id_lists = self._decode_cot(cot_ids, trajectories.shape[0])
        # cot_gen is derived (total minus the ctx and SDE phases) so the log
        # stays honest even when the CoT forward is skipped above.
        logger.warning(
            "[time] sample_grpo: ctx=%.1fs sde=%.1fs cot_gen=%.1fs total=%.1fs (B=%d)",
            _t_ctx, _t_sde, _time.time() - _t0 - _t_ctx - _t_sde, _time.time() - _t0,
            trajectories.shape[0],
        )
        return {
            "trajectories": trajectories,
            "x0_normed": x0_normed,
            "record": record,
            "batch": batch,
            "cot_texts": cot_texts,
            "cot_ids": cot_id_lists,
            "cot_logps": cot_logps,
            "cot_entropies": cot_ents,
        }

    def cot_logprob(
        self,
        transfusion_inputs: List[Dict[str, Any]],
        actions_normed: torch.Tensor,
        cot_ids: torch.Tensor,
        *,
        ctx_embeds: Optional[torch.Tensor] = None,
        device: Optional[Any] = None,
    ):
        """Per-token CoT log-prob + entropy of ``cot_ids`` under ``actions_normed``.

        Thin wrapper over :meth:`TransfusionQwen3p5.cot_logprob` that collates the
        model inputs. Honours the caller's grad context (``torch.no_grad`` for the
        behaviour-policy old log-prob; grad-enabled for the actor update). Returns
        ``(token_logp[B, C], token_entropy[B, C])``.
        """
        device = device or next(self.model.parameters()).device
        batch = self._to_device(self.collate(transfusion_inputs), device)
        # Model-vintage compatibility: drop `state_features` when the installed
        # cot_logprob does not accept it (removed in the cleaned AUGUstus_SFT).
        import inspect as _inspect

        _cl_kwargs: Dict[str, Any] = dict(
            input_ids=batch["input_ids"],
            actions_normed=actions_normed.to(device),
            cot_ids=cot_ids.to(device),
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            ctx_embeds=ctx_embeds,
        )
        if "state_features" in _inspect.signature(self.model.cot_logprob).parameters:
            _cl_kwargs["state_features"] = batch.get("state_features")
        return self.model.cot_logprob(**_cl_kwargs)

__all__ = [
    "load_transfusion_config",
    "build_transfusion_model",
    "TransfusionPolicy",
]
