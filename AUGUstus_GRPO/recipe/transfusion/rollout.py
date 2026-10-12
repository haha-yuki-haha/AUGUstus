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

"""Diffusion-trajectory rollout for the Transfusion policy.

verl's stock rollouts (vLLM / SGLang / TRT-LLM ``ServerAdapter``) are
token-in/token-out LLM servers and cannot drive a continuous-action diffusion
planner. This module provides a **sync** :class:`BaseRollout` that instead runs
Transfusion's own ``generate_trajectory`` denoiser and packs the results so the
reward stage can consume them.

What it produces per sample (into the returned ``DataProto.non_tensor_batch``):

* ``pred_trajectory`` — ``[8, 3]`` de-normalised waypoints (for L2 / PDMS reward).
* ``pred_cot_text``   — decoded Chain-of-Thought string (for the CoT reward).
* ``responses``       — the CoT token ids (right-padded) so verl's reward
  manager can ``tokenizer.decode`` them and GRPO can attach the scalar reward.
* ``pred_trajectory`` is *also* mirrored into each sample's ``extra_info`` so the
  simple ``custom_reward_function`` path sees it without a custom manager.

The class registers itself under ``("transfusion", "sync")`` /
``("transfusion", "async")`` in verl's rollout registry on import, so the recipe
config can select it with ``actor_rollout_ref.rollout.name=transfusion`` without
editing any core file.

NOTE (cluster-side work, see ``docs/ADAPTATION.md``): weight resharding from the
FSDP actor into this rollout (``update_weights``) and the continuous-action
policy-gradient *loss* are the parts that must be finished against your training
cluster; they are intentionally minimal here so the data/model/reward wiring can
be exercised and unit-tested first.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch

from verl import DataProto
from verl.workers.rollout.base import _ROLLOUT_REGISTRY, BaseRollout

logger = logging.getLogger(__name__)


class TransfusionRollout(BaseRollout):
    """Synchronous diffusion-trajectory rollout wrapping :class:`TransfusionPolicy`."""

    def __init__(self, config=None, model_config=None, device_mesh=None, *args, **kwargs):
        # BaseRollout.__init__ expects dataclass-convertible configs; tolerate
        # being constructed standalone (offline smoke tests) where they are absent.
        if config is not None and model_config is not None:
            super().__init__(config, model_config, device_mesh, *args, **kwargs)
        else:
            self.config = config
            self.model_config = model_config
            self.device_mesh = device_mesh

        self._policy = kwargs.get("policy")  # allow injection for tests
        self._train_only_action = bool(
            kwargs.get("train_only_action", False)
            or (getattr(config, "train_only_action", False) if config is not None else False)
        )
        self._generate_cot = bool(kwargs.get("generate_cot", True)) and not self._train_only_action
        self._transfusion_config_path = kwargs.get("transfusion_config_path") or os.environ.get(
            "TRANSFUSION_CONFIG"
        )
        self._ckpt_path = kwargs.get("ckpt_path") or os.environ.get("TRANSFUSION_CKPT")

        def _cfg_get(name, default):
            val = kwargs.get(name)
            if val is None and config is not None:
                val = getattr(config, name, None)
            if val is None:
                # Env fallback: prefer the TRANSFUSION_-prefixed var exported by
                # the run scripts (TRANSFUSION_RL_NUM_STEPS, ...), then the bare
                # name for backward compatibility.
                val = os.environ.get(f"TRANSFUSION_{name.upper()}")
                if val is None:
                    val = os.environ.get(name.upper())
            if val is None or (isinstance(val, str) and val.strip().lower() in ("", "none", "null")):
                return default
            return val

        # Number of denoising steps used to produce the action, shared by ALL
        # three algorithms (original GRPO / Flow-GRPO / DiffusionNFT); set via
        # actor_rollout_ref.rollout.rl_num_steps in the recipe config.
        self._rl_num_steps = int(_cfg_get("rl_num_steps", 10))
        # Optional SEPARATE step count for the in-training validation rollouts
        # (verl's ``_validate`` tags the gen batch with meta_info['validate']);
        # None -> validation uses the same ``rl_num_steps`` as training.
        _val_steps = _cfg_get("rl_val_num_steps", None)
        self._rl_val_num_steps = None if _val_steps is None else int(_val_steps)

        # OOM guard: chunk generation into smaller forward passes. None = off.
        self._rl_gen_micro_batch_size = _cfg_get("rl_gen_micro_batch_size", None)
        if self._rl_gen_micro_batch_size is not None:
            self._rl_gen_micro_batch_size = int(self._rl_gen_micro_batch_size)

        def _parse_noise_level(val):
            """Parse a noise level: scalar, per-dim list, or the NESTED per-group
            form ``[[nl_x, nl_y, nl_h], ...]`` (inner i-th list drives the i-th
            contiguous chunk of every rollout.n group; last chunk may be smaller).
            Env strings accept JSON ``[[0.05, 0.05, 0.02], [0.01, 0.01, 0.01]]``
            or the legacy flat form ``0.1,0.1,0.01``."""
            if isinstance(val, str):
                s = val.strip()
                if s.startswith("[["):
                    import json as _json
                    return _parse_noise_level(_json.loads(s))
                s = s.strip("[]()")
                parts = [p for p in s.replace(",", " ").split() if p]
                return [float(p) for p in parts] if len(parts) > 1 else float(parts[0])
            if isinstance(val, (int, float)):
                return float(val)
            try:
                seq = list(val)
            except TypeError:
                return float(val)
            if not seq:
                return 0.0
            def _is_seq(x):
                return isinstance(x, (list, tuple)) or type(x).__name__ == "ListConfig"
            if any(_is_seq(x) for x in seq):
                return [[float(w) for w in x] if _is_seq(x) else float(x) for x in seq]
            return [float(v) for v in seq]

        self._rl_noise_level = _parse_noise_level(_cfg_get("rl_noise_level", 0.7))

        # CoT sampling knobs for TRAINING rollouts (group diversity). These are
        # verl's standard rollout sampling fields, forwarded to the Transfusion
        # autoregressive CoT sampler; validation uses rl_val_cot_* below when set.
        self._cot_temperature = float(_cfg_get("temperature", 0.9))
        _tp = float(_cfg_get("top_p", 0.95))
        self._cot_top_p = _tp if 0.0 < _tp <= 1.0 else 1.0
        _tk = int(_cfg_get("top_k", 100))
        self._cot_top_k = _tk if _tk > 0 else 0  # <=0 -> disabled (vLLM convention)
        # Optional SEPARATE CoT sampling knobs for the in-training VALIDATION
        # rollouts (None -> reuse the training-rollout knobs above). Before these
        # existed, validation fell back to the policy / SFT-eval defaults, which
        # are less diverse than the training sampler. Env fallbacks:
        # TRANSFUSION_RL_VAL_COT_TEMPERATURE / _TOP_P / _TOP_K.
        _vt = _cfg_get("rl_val_cot_temperature", None)
        self._cot_val_temperature = self._cot_temperature if _vt is None else float(_vt)
        _vp = _cfg_get("rl_val_cot_top_p", None)
        _vp = self._cot_top_p if _vp is None else float(_vp)
        self._cot_val_top_p = _vp if 0.0 < _vp <= 1.0 else 1.0
        _vk = _cfg_get("rl_val_cot_top_k", None)
        _vk = self._cot_top_k if _vk is None else int(_vk)
        self._cot_val_top_k = _vk if _vk > 0 else 0

        # RL CoT generation cap: generate at most ``response_length - 1`` CONTENT
        # tokens so the forced EOT column appended for never-finished samples
        # (see TransfusionQwen3p5._generate_cot_tokens) fits inside the response
        # window that verl keeps (rollout.response_length). Sampling itself
        # stops on the SFT stop token <|eot|>; the cap is only a safety bound.
        # It also stops the sampler from wasting compute generating up to the
        # model's max_cot_tokens (512) tokens that would be truncated to
        # response_length (256) by _pack anyway.
        _rl_cot_budget = int(_cfg_get("response_length", 512))
        self._cot_max_tokens = max(1, _rl_cot_budget - 1)

        # Reverse-SDE diffusion coefficient g(t) schedule (flow_grpo / dancegrpo /
        # lambda_t); rl_noise_level is its scalar hyper-parameter (a / eta / lambda).
        # Validate/canonicalise here so a typo fails loudly at construction rather
        # than deep inside the SDE step. Env fallback TRANSFUSION_RL_NOISE_SCHEDULE.
        from recipe.transfusion.algorithms.fm_sde_logprob import (
            normalize_noise_schedule as _norm_sched,
        )

        self._rl_noise_schedule = _norm_sched(_cfg_get("rl_noise_schedule", "flow_grpo"))

    @staticmethod
    def _is_validation(prompts: DataProto) -> bool:
        """True for the in-training validation gen batch (verl tags it)."""
        meta = getattr(prompts, "meta_info", None) or {}
        return bool(meta.get("validate", False))

    def _num_steps_for(self, prompts: DataProto) -> int:
        """Denoising steps for this batch: validation batches (``_validate`` tags
        them with ``meta_info['validate']=True``) use ``rl_val_num_steps`` when
        configured, otherwise both train and val use ``rl_num_steps``."""
        if self._is_validation(prompts) and self._rl_val_num_steps is not None:
            return self._rl_val_num_steps
        return self._rl_num_steps

    def _noise_level_for(self, prompts: DataProto):
        """SDE noise level (schedule hyper-parameter) for this batch.

        A scalar or a per-dim list (e.g. ``[nl_x, nl_y, nl_heading]``)."""
        return self._rl_noise_level

    def _noise_schedule_for(self, prompts: DataProto) -> str:
        """Reverse-SDE g(t) schedule for this batch."""
        return self._rl_noise_schedule

    def _group_noise_plan_for(self, prompts: DataProto, num_rows: int):
        """Per-sample noise tensor ``[num_rows, T_law, D]`` for the NESTED
        per-group ``rl_noise_level`` form, or ``None`` for scalar / per-dim.

        The within-group slots (row ``i`` of this shard is slot ``i % G``) map to
        inner noise lists by :func:`fm_sde_logprob.group_noise_slot_mapping`
        (chunk = ceil(G/L); last chunk smaller when G is indivisible). Assumes the
        shard holds whole consecutive groups (true whenever the trainer batch is
        divisible by the worker count — the framework's required divisibility).
        """
        from recipe.transfusion.algorithms.fm_sde_logprob import (
            build_group_noise_plan,
            is_group_noise_plan,
        )

        nl = self._noise_level_for(prompts)
        if not is_group_noise_plan(nl):
            return None
        group_size = max(1, int(getattr(self.config, "n", 1) or 1))
        action_dim = int(self.policy.adapter.action_dim)
        plan = build_group_noise_plan(
            nl, num_rows=num_rows, group_size=group_size, action_dim=action_dim,
        )
        if not getattr(self, "_group_noise_plan_logged", False):
            self._group_noise_plan_logged = True
            logger.info(
                "rl_noise_level is NESTED per-group: %d inner lists over group_size=%d "
                "-> per-sample plan shape %s (first group slots: %s).",
                len(nl), group_size, tuple(plan.shape),
                plan[:group_size, 0, 0].tolist(),
            )
        return plan

    def _cot_sampler_for(self, prompts: DataProto) -> Dict[str, Any]:
        """CoT sampling kwargs for this batch: validation batches use the
        ``rl_val_cot_*`` knobs (falling back to the training-rollout knobs)."""
        if self._is_validation(prompts):
            return {
                "cot_temperature": self._cot_val_temperature,
                "cot_top_p": self._cot_val_top_p,
                "cot_top_k": self._cot_val_top_k,
            }
        return {
            "cot_temperature": self._cot_temperature,
            "cot_top_p": self._cot_top_p,
            "cot_top_k": self._cot_top_k,
        }


    # -- lazy policy construction ------------------------------------------
    @property
    def policy(self):
        if self._policy is None:
            from recipe.transfusion.transfusion_policy import (
                TransfusionPolicy,
                build_transfusion_model,
                load_transfusion_config,
            )

            if not self._transfusion_config_path:
                raise ValueError(
                    "TransfusionRollout needs a Transfusion config: pass transfusion_config_path "
                    "or set $TRANSFUSION_CONFIG."
                )
            cfg = load_transfusion_config(self._transfusion_config_path)
            model = build_transfusion_model(cfg, ckpt_path=self._ckpt_path)
            # ``build_transfusion_model`` (train_qwen3p5.build_model + the checkpoint
            # state-dict load) never places the module on CUDA itself — the HF
            # backbone loaded via ``from_pretrained`` may end up on GPU, but bespoke
            # modules constructed as plain ``nn.Parameter``/``nn.Module`` (e.g. the
            # GatedDeltaNet ``dt_bias``/``A_log``/``in_proj_*``, action encoder/decoder,
            # time embed) default to CPU and ``load_state_dict(strict=False)`` copies
            # checkpoint values in place without changing device. Under the slow pure
            # PyTorch fallback (no ``fla`` installed) this silently limped along on
            # whichever device each tensor happened to be on; the fused Triton kernels
            # (once ``flash-linear-attention`` is installed) hard-require CUDA tensors
            # and raise "Pointer argument ... cannot be accessed from Triton (cpu
            # tensor?)" otherwise. Explicitly move the whole (un-sharded, eager) rollout
            # module to this worker's GPU so every parameter/buffer is consistently
            # placed before inference.
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda", torch.cuda.current_device()))
            model.eval()
            processor = getattr(model, "processor", None)
            self._policy = TransfusionPolicy(
                model,
                processor,
                traj_token=cfg["trajectory"]["traj_token"],
                action_dim=cfg["trajectory"]["action_dim"],
            )
        return self._policy

    @classmethod
    def from_policy(cls, policy, generate_cot: bool = True) -> "TransfusionRollout":
        """Build a rollout around an already-constructed :class:`TransfusionPolicy` (offline)."""
        return cls(config=None, model_config=None, device_mesh=None, policy=policy, generate_cot=generate_cot)

    # -- core generation ----------------------------------------------------
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Run diffusion sampling for a batch and return a DataProto with predictions.

        ``prompts.non_tensor_batch["transfusion_inputs"]`` is the object array of
        per-sample model-input dicts produced by
        :class:`recipe.transfusion.dataset.TransfusionRLDataset`.
        """
        nt = prompts.non_tensor_batch
        if "transfusion_inputs" not in nt:
            raise KeyError(
                "TransfusionRollout expects 'transfusion_inputs' in non_tensor_batch; "
                "use recipe.transfusion.dataset.TransfusionRLDataset."
            )
        transfusion_inputs: List[Dict[str, Any]] = list(nt["transfusion_inputs"])

        # OOM guard: chunk the generation into smaller forward passes. The
        # synchronous Transfusion rollout has NO internal micro-batching, so a
        # large train_batch_size * rollout.n OOMs on the 2B multimodal model.
        # ``rl_gen_micro_batch_size`` (rollout config) caps the per-chunk batch
        # sent to ``model.generate_trajectory`` / ``sample_grpo``. None = no
        # chunking (old behaviour). The results are concatenated back into a
        # single DataProto so the rest of the pipeline is unchanged.
        gen_mb = self._rl_gen_micro_batch_size
        total = len(transfusion_inputs)
        # NESTED per-group noise (rl_noise_level: [[..], [..]]): resolve ONCE for
        # the whole shard, THEN slice along micro-batch boundaries — otherwise
        # chunk borders would mis-shift the within-group noise assignment. This
        # is a no-op (None) for scalar / flat per-dim configs (unchanged path).
        noise_plan = self._group_noise_plan_for(prompts, total)
        if gen_mb is not None and gen_mb > 0 and total > gen_mb:
            return self._generate_chunked(prompts, transfusion_inputs, gen_mb, noise_plan=noise_plan)

        # verl-native token-level GRPO with the
        # closed-form action ratio + token-probability CoT ratio (the design in
        # ``docs/SINGLE_STEP_X0_GRPO.md`` combined with an autoregressive-CoT PPO
        # term). TRAINING rollouts sample the action stochastically (reverse-SDE)
        # and record the per-step transitions + the sampled CoT token ids so the
        # actor update can re-score BOTH heads on-policy; VALIDATION keeps the
        # deterministic sampler (``rl_val_num_steps`` / DDIM) for clean metrics —
        # its noisy SDE record would be meaningless and is not needed there.
        if not self._is_validation(prompts):
            return self._generate_grpo(prompts, transfusion_inputs, noise_plan=noise_plan)

        out = self.policy.rollout(
            transfusion_inputs,
            generate_cot=self._generate_cot,
            return_normed=False,
            num_steps=self._num_steps_for(prompts),
            **self._cot_sampler_for(prompts),
        )
        trajectories: np.ndarray = out["trajectories"]  # [B, 8, 3]
        cot_texts: List[str] = out["cot_texts"]
        cot_ids: List[List[int]] = out["cot_ids"]

        return self._pack(prompts, trajectories, cot_texts, cot_ids)

    # -- diffusion-RL collection --------------------------------------------
    def _generate_grpo(self, prompts: DataProto, transfusion_inputs, noise_plan=None):
        """Combined GRPO rollout: SDE action + post-hoc CoT + per-sample records.

        Produces the de-normalised trajectory + CoT (for the reward) AND attaches
        the per-step SDE transition record (``flow_grpo_record``) and the clean
        sampled trajectory (``x0_normed``) so the actor update can compute the
        closed-form action ratio and the token-probability CoT ratio on-policy.

        ``noise_plan`` (per-sample tensor from :meth:`_group_noise_plan_for`)
        overrides the batch-wide ``rl_noise_level`` when the NESTED per-group
        form is configured.
        """
        out = self.policy.sample_grpo(
            transfusion_inputs,
            num_steps=self._num_steps_for(prompts),
            noise_level=noise_plan if noise_plan is not None else self._noise_level_for(prompts),
            noise_schedule=self._noise_schedule_for(prompts),
            generate_cot=self._generate_cot,
            cot_temperature=self._cot_temperature,
            cot_top_p=self._cot_top_p,
            cot_top_k=self._cot_top_k,
            cot_max_tokens=self._cot_max_tokens,
        )
        packed = self._pack(prompts, out["trajectories"], out["cot_texts"], out["cot_ids"])
        self._attach_per_sample(packed, "x0_normed", out["x0_normed"])
        self._attach_flow_grpo_record(packed, out["record"])
        self._attach_cot_logps(packed, out.get("cot_logps"), out.get("cot_entropies"))
        del out
        return packed

    def _generate_chunked(self, prompts: DataProto, transfusion_inputs, gen_mb: int, noise_plan=None) -> DataProto:
        """Chunked generation: run ``generate_sequences`` on slices of at most
        ``gen_mb`` samples and concatenate the per-chunk DataProtos.

        The OOM guard is transparent to the rest of the pipeline: the returned
        DataProto has the same keys / shapes as a single-call generation would
        produce. Each chunk is a independent forward pass (``model.generate_
        trajectory`` / ``sample_grpo``), so peak GPU memory scales with
        ``gen_mb`` instead of the full per-GPU shard.

        GPU memory is explicitly freed between chunks (``del`` the per-chunk
        ``out`` dict + ``gc.collect`` + ``cuda.empty_cache``). Without this,
        PyTorch's caching allocator may hold onto the previous chunk's
        activations, causing peak memory to grow across chunks and eventually
        OOM-hang the GPU driver (the raylet heartbeat timeout / VS Code
        disconnect symptom).
        """
        import gc
        from verl import DataProto as _DP

        total = len(transfusion_inputs)
        chunks: List[DataProto] = []
        n_chunks = (total + gen_mb - 1) // gen_mb
        for ci, start in enumerate(range(0, total, gen_mb)):
            end = min(start + gen_mb, total)
            chunk_inputs = transfusion_inputs[start:end]
            logger.warning(
                "[chunked] chunk %d/%d: samples [%d:%d] (B=%d)",
                ci + 1, n_chunks, start, end, end - start,
            )
            # Slice the corresponding non_tensor keys from the original batch.
            chunk_nt = {}
            for k, v in prompts.non_tensor_batch.items():
                if isinstance(v, np.ndarray):
                    chunk_nt[k] = v[start:end]
                elif isinstance(v, list):
                    chunk_nt[k] = v[start:end]
                else:
                    chunk_nt[k] = v
            chunk_plan = None if noise_plan is None else noise_plan[start:end]
            chunk_dp = _DP.from_dict(non_tensors=chunk_nt)
            chunk_dp.meta_info = dict(getattr(prompts, "meta_info", {}) or {})
            # Dispatch to the right generator for this chunk.
            if not self._is_validation(prompts):
                chunk_dp = self._generate_grpo(chunk_dp, chunk_inputs, noise_plan=chunk_plan)
            else:
                # Validation path.
                out = self.policy.rollout(
                    chunk_inputs,
                    generate_cot=self._generate_cot,
                    return_normed=False,
                    num_steps=self._num_steps_for(prompts),
                    **self._cot_sampler_for(prompts),
                )
                chunk_dp = self._pack(chunk_dp, out["trajectories"], out["cot_texts"], out["cot_ids"])
                del out
            chunks.append(chunk_dp)
            # Free GPU memory between chunks so the caching allocator doesn't
            # hold the previous chunk's activations, which would make peak
            # memory grow across chunks (defeating the purpose of chunking).
            if ci + 1 < n_chunks:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        # Concatenate all chunk DataProtos back into one.
        return _DP.concat(chunks)

    @staticmethod
    def _attach_cot_logps(prompts: DataProto, cot_logps, cot_entropies):
        """Attach the sampler-recorded behaviour-policy CoT log-probs / entropies.

        The tensors are ``[B, W]`` (W = generated width, variable per worker);
        they are right-zero-padded to the SAME width as ``responses`` so they
        concat cleanly across DP workers and land in ``batch`` as
        ``old_log_probs`` / ``entropys`` — letting the trainer skip the
        teacher-forced old-log-prob recompute phase entirely. Padding with 0 is
        safe: ``response_mask`` selects the scored positions downstream.
        """
        if cot_logps is None or prompts.batch is None or "responses" not in prompts.batch:
            return
        b, max_r = prompts.batch["responses"].shape
        old = torch.zeros((b, max_r), dtype=torch.float32)
        ent = torch.zeros((b, max_r), dtype=torch.float32)
        n = min(int(cot_logps.shape[1]), max_r)
        if n > 0:
            old[:, :n] = cot_logps[:, :n].float().cpu()
            if cot_entropies is not None:
                ent[:, :n] = cot_entropies[:, :n].float().cpu()
        prompts.batch["old_log_probs"] = old
        prompts.batch["entropys"] = ent

    @staticmethod
    def _attach_per_sample(prompts: DataProto, key: str, batched: torch.Tensor):
        """Stash a ``[B, ...]`` tensor as a per-sample object array in non_tensor_batch.

        Pre-allocate a 1-D ``object`` array and assign each per-sample slice so
        numpy does NOT auto-stack the uniform-shape rows into a MULTI-dimensional
        object array. That auto-stacking made ``non_tensor_batch[key][i]`` a
        numpy ``object`` ndarray which ``torch.as_tensor`` rejects ("can't convert
        np.ndarray of type numpy.object_") when the actor update / old-log-prob
        pass restack it -- crashing both into their zero fallbacks.
        """
        arr = batched.detach().float().cpu().numpy()
        out = np.empty(arr.shape[0], dtype=object)
        for i in range(arr.shape[0]):
            out[i] = arr[i].copy()
        prompts.non_tensor_batch[key] = out

    @staticmethod
    def _attach_flow_grpo_record(prompts: DataProto, record: Dict[str, Any]):
        """Slice the batched SDE record into per-sample dicts for non_tensor_batch.

        Batched tensors are ``x_in``/``next_sample`` ``[S, B, T, D]`` and
        ``sigma``/``sigma_prev``/``log_probs`` ``[S, B]``; each sample keeps its
        own ``[S, ...]`` slice so the Flow-GRPO loss can re-batch and re-score it.
        """
        b = record["x_in"].shape[1]
        # noise_level may be a scalar, a per-dim/per-step tensor ([D] / [T, D]),
        # or a PER-SAMPLE plan tensor ([B, T?, D] from the NESTED per-group
        # rl_noise_level form). Store tensors as numpy so the per-sample object
        # record stays picklable and collate_flow_grpo_records can rebuild the
        # tensor for the re-score. A per-sample tensor is sliced ROW-WISE so the
        # training-time re-score uses EXACTLY the noise this sample drew with.
        _nl = record["noise_level"]
        if torch.is_tensor(_nl):
            _nl = _nl.detach().float().cpu().numpy()
        # [T, D] per-step/per-dim plan tensors must NOT be sliced row-wise
        # (row 0 of a [T, D] plan is one TIMESTEP, not one sample): only
        # genuinely per-sample plans ([B, ...] with B rows, and a 2-D plan
        # that is NOT exactly [T, D]) are split per sample.
        _t_d = (int(record["x_in"].shape[2]), int(record["x_in"].shape[3]))
        if isinstance(_nl, np.ndarray) and (
            (_nl.ndim >= 3 and _nl.shape[0] == b)
            or (_nl.ndim == 2 and _nl.shape[0] == b and tuple(_nl.shape) != _t_d)
        ):
            noise_levels = [_nl[i].copy() for i in range(b)]
        else:
            noise_levels = [_nl] * b
        scalars = {
            "sigma_max": record["sigma_max"],
            # Propagate the g(t) schedule so the training-time closed-form
            # re-score uses the SAME coefficients the rollout sampled under
            # (otherwise it silently defaults to 'flow_grpo' and the ratio/KL are
            # computed with the wrong kernel).
            "noise_schedule": record.get("noise_schedule", "flow_grpo"),
        }
        has_x0_old = "x0_old" in record  # JiT-GRPO closed-form ratio input
        per_sample = []
        for i in range(b):
            rec_i = {
                "x_in": record["x_in"][:, i].detach().float().cpu().clone(),
                "next_sample": record["next_sample"][:, i].detach().float().cpu().clone(),
                "sigma": record["sigma"][:, i].detach().float().cpu().clone(),
                "sigma_prev": record["sigma_prev"][:, i].detach().float().cpu().clone(),
                "log_probs": record["log_probs"][:, i].detach().float().cpu().clone(),
                "noise_level": noise_levels[i],
                **scalars,
            }
            if has_x0_old:
                rec_i["x0_old"] = record["x0_old"][:, i].detach().float().cpu().clone()
            per_sample.append(rec_i)
        prompts.non_tensor_batch["flow_grpo_record"] = np.array(per_sample, dtype=object)


    def _pack(
        self,
        prompts: DataProto,
        trajectories: np.ndarray,
        cot_texts: List[str],
        cot_ids: List[List[int]],
    ) -> DataProto:
        """Attach predictions onto the input ``DataProto`` in place and return it."""
        b = trajectories.shape[0]

        # Right-pad CoT token ids into a [B, R] responses tensor for verl bookkeeping.
        # Use a FIXED max_response_length (from the rollout config, default 512) so
        # all DP workers produce the same tensor width — otherwise DataProto.concat
        # across workers fails with "Sizes of tensors must match except in dim 0".
        pad_id = 0
        tok = getattr(self.policy, "processor", None)
        if tok is not None and getattr(tok, "tokenizer", None) is not None:
            pad_id = tok.tokenizer.pad_token_id or 0
        # EOT-aware mask: when the model emits its EOT token MID-response (e.g.
        # a truncated CoT), everything after the FIRST EOT is padding for the
        # RL loss too — mask only up to and including that EOT.
        eot_id = 0
        _model = getattr(getattr(self.policy, "model", None), "eot_token_id", None)
        if _model:
            eot_id = int(_model)
        max_r = getattr(self.config, "response_length", None) if self.config else None
        if max_r is None:
            max_r = max(max((len(x) for x in cot_ids), default=1), 1)
        else:
            max_r = int(max_r)
        responses = torch.full((b, max_r), pad_id, dtype=torch.long)
        response_mask = torch.zeros((b, max_r), dtype=torch.long)
        for i, ids in enumerate(cot_ids):
            if not ids:
                continue
            n = min(len(ids), max_r)
            responses[i, : n] = torch.tensor(ids[:n], dtype=torch.long)
            if eot_id:
                row = responses[i, :n]
                eot_hits = (row == eot_id).nonzero()
                valid = int(eot_hits[0]) + 1 if len(eot_hits) else n
            else:
                valid = n
            response_mask[i, : valid] = 1

        # train_only_action (noCoT): no CoT tokens are generated, so the mask
        # built above is ALL-ZERO. That silently kills the whole RL update:
        #   * verl's GRPO outcome advantage is broadcast as
        #       normalized_score.unsqueeze(-1) * response_mask   (core_algos)
        #     -> every advantage becomes 0, and
        #   * update_transfusion_actor reduces the per-sample advantage via
        #     (adv * response_mask).sum / response_mask.sum -> 0 as well,
        # so only the KL-to-ref anchor backpropagates (the observed
        # actor/adv_mean == 0 and "no learning" run). The diffusion trajectory
        # IS the single action here, so mark ONE virtual action token at
        # position 0: the reward's EOS placement (NaiveRewardManager,
        # valid_response_length == 1) and the advantage broadcast/reduce then
        # carry the per-sample GRPO score through to the action head.
        # response_length becomes 1, which is CORRECT semantics (1 action).
        if self._train_only_action:
            response_mask[:, 0] = 1

        # Predicted trajectory + CoT as non-tensor object arrays.
        prompts.non_tensor_batch["pred_trajectory"] = np.array(
            [trajectories[i] for i in range(b)], dtype=object
        )
        prompts.non_tensor_batch["pred_cot_text"] = np.array(cot_texts, dtype=object)

        # NOTE: ``pred_trajectory`` is intentionally kept as its OWN non_tensor key
        # and NOT mirrored into ``extra_info``. verl's DataProto.union (used in
        # _validate: test_batch.union(gen_output)) asserts that any non_tensor key
        # present in BOTH operands is deep-equal, so mutating the shared
        # ``extra_info`` here would raise "extra_info ... not the same object". The
        # reward stage injects ``pred_trajectory`` into ``extra_info`` itself.

        # Response tokens (CoT) for reward decoding + GRPO scalar attachment.
        if prompts.batch is not None:
            prompts.batch["responses"] = responses
            prompts.batch["response_mask"] = response_mask
        else:
            # The incoming gen batch carries only non-tensor data (the
            # ``transfusion_inputs`` object array), so ``prompts.batch`` is None.
            # ``DataProto.union`` assumes ``self.batch`` exists (it dereferences
            # ``self.batch.batch_size``), so attach the new TensorDict directly.
            prompts.batch = DataProto.from_dict(
                tensors={"responses": responses, "response_mask": response_mask}
            ).batch
        # verl's text-RL _validate logs decoded ``batch["prompts"]`` (the input
        # token ids). The Transfusion prompt is a non-tensor multimodal bundle
        # (``transfusion_inputs``), so there is no such tensor; add a minimal
        # placeholder purely to satisfy verl's logging/bookkeeping. It is NOT
        # consumed by the Transfusion model (which reads ``transfusion_inputs``).
        if "prompts" not in prompts.batch.keys():
            prompts.batch["prompts"] = torch.full((b, 1), pad_id, dtype=torch.long)

        # verl's ``fit`` reads ``attention_mask`` (for global_token_num /
        # _balance_batch) and ``multi_modal_inputs`` (for images_seqlens). The
        # Transfusion prompt is a non-tensor multimodal bundle, so there is no
        # real token-level attention mask; synthesise a [B, 1+max_r] mask (1 for
        # the prompt placeholder + the response_mask) so these verl bookkeeping
        # paths do not KeyError.
        if "attention_mask" not in prompts.batch.keys():
            prompt_mask = torch.ones((b, 1), dtype=torch.long)
            prompts.batch["attention_mask"] = torch.cat([prompt_mask, response_mask], dim=1)

        # verl's ``fit`` iterates ``batch.non_tensor_batch["multi_modal_inputs"]``
        # to collect ``images_seqlens``. The Transfusion model's multimodal data
        # lives inside ``transfusion_inputs`` (not in this verl-level key), so
        # provide an empty list per sample to make the loop a no-op.
        if "multi_modal_inputs" not in prompts.non_tensor_batch.keys():
            prompts.non_tensor_batch["multi_modal_inputs"] = np.array(
                [{} for _ in range(b)], dtype=object
            )

        # Real per-sample prompt / response token counts so verl's data metrics
        # report a meaningful ``prompt_length`` / ``response_length`` instead of the
        # length-1 prompt placeholder. metric_utils._compute_response_info uses
        # ``batch['prompt_length']`` / ``['response_length']`` DIRECTLY when both are
        # present; otherwise it derives prompt_length from ``attention_mask[:, :-R]``
        # — which is the ``[b, 1]`` placeholder mask here, hence the constant 1. As
        # ``[B]`` tensors they concat cleanly across DP workers (a variable-width real
        # prompt tensor would not).
        ti = prompts.non_tensor_batch.get("transfusion_inputs")
        if ti is not None:
            plen = torch.zeros(b, dtype=torch.float32)
            for i in range(b):
                ids = ti[i].get("input_ids") if isinstance(ti[i], dict) else None
                if ids is not None:
                    plen[i] = float(ids.shape[-1] if hasattr(ids, "shape") else len(ids))
            prompts.batch["prompt_length"] = plen
        prompts.batch["response_length"] = response_mask.sum(-1).float()

        return prompts

    # -- BaseRollout lifecycle (minimal; see module docstring) --------------
    async def resume(self, tags: List[str]):  # noqa: D401 - simple no-op
        return None

    async def update_weights(
        self, weights: Generator[Any, None, None], **kwargs
    ):
        """Load fresh actor weights into the rollout policy.

        Minimal eager implementation: copy matching tensors by name. Full
        FSDP-sharded resharding is a cluster-side extension (see docs).
        """
        if self._policy is None:
            # Materialise the policy first so there is a model to update.
            _ = self.policy
        model = self._policy.model
        own = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in weights:
                if name in own and own[name].shape == tensor.shape:
                    own[name].copy_(tensor.to(own[name].dtype).to(own[name].device))
        return None

    async def release(self):  # noqa: D401 - simple no-op
        return None


# Self-register so config `actor_rollout_ref.rollout.name=transfusion` resolves
# without editing verl core. Both modes point at this sync implementation.
for _key in (("transfusion", "sync"), ("transfusion", "async")):
    _ROLLOUT_REGISTRY.setdefault(_key, "recipe.transfusion.rollout.TransfusionRollout")


__all__ = ["TransfusionRollout"]
