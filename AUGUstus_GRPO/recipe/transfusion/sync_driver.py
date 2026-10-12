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

"""Driver-side glue to run the SYNCHRONOUS Transfusion rollout under verl's
async-only PPO trainer — no verl core file is edited.

This verl version removed the sync rollout path: ``RayPPOTrainer.init_workers``
unconditionally builds an async LLM server (``LLMServerManager``) + an
``AgentLoopManager``, and ``fit()`` / ``_validate()`` generate exclusively via
``self.async_rollout_manager.generate_sequences(...)``. ``LLMServerManager`` only
knows the server replicas ``vllm`` / ``sglang`` / ``trtllm`` (a SEPARATE registry
from the recipe's ``_ROLLOUT_REGISTRY``), so the custom synchronous
``TransfusionRollout`` cannot plug in and ``init_workers`` dies with::

    ValueError: Unknown rollout mode: transfusion. Available: ['vllm','sglang','trtllm']

This module bridges the gap:

* :class:`TransfusionActorRolloutRefWorker` — adds a
  ``@register(Dispatch.DP_COMPUTE_PROTO)`` ``generate_sequences`` worker method
  that calls the worker-side sync rollout (``self.rollout.generate_sequences``),
  so the actor-rollout ``WorkerGroup`` exposes a synchronous generate proxy.
  ``DP_COMPUTE_PROTO`` splits/collects a ``DataProto`` (preserving
  ``non_tensor_batch['transfusion_inputs']``), unlike the TensorDict-based
  ``make_nd_compute_dataproto_dispatch_fn`` used by the stock engine methods.
* :class:`SyncRolloutTrainerMixin` — overrides ``init_workers`` to swap the async
  LLM-server / AgentLoop / checkpoint-engine managers for lightweight shims (via a
  scoped monkeypatch of the classes ``init_workers`` imports) and routes
  ``async_rollout_manager.generate_sequences`` to the worker group. ``fit()`` /
  ``_validate()`` then run unchanged.

NOTE on weight sync: the async checkpoint-engine manager is shimmed out, but the
actor->rollout weight sync itself IS implemented — after every actor update
``TransfusionActorRolloutRefWorker._sync_actor_weights_to_rollout`` copies the
trained (FSDP full-state-dict) weights into the eager rollout policy on each
worker, so the next rollout samples from the freshly updated policy (on-policy).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker

logger = logging.getLogger(__name__)


def _is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` is a CUDA out-of-memory (including Triton / cuBLAS /
    cuDNN OOMs that surface as plain ``RuntimeError``, NOT ``torch.OutOfMemoryError``).

    Without this, a Triton ``RuntimeError: ...out of memory`` inside an FLA
    backward is swallowed by a generic ``except Exception`` -- the FSDP module is
    then left with its sharded flat-parameters unre-materialised, so every later
    ``update_transfusion_actor`` dies on ``next(model.parameters())`` ->
    ``StopIteration`` while training silently spins on ``error=1.0``.
    """
    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "cuda error", "device-side assert",
        "cublas", "cudnn", "triton error [cuda]",
        "nccl", "cuda oom",
    )
    return any(m in msg for m in oom_markers)


# ---------------------------------------------------------------------------
# Worker subclass: expose a SYNC generate_sequences proxy on the WorkerGroup.
# Defined at MODULE level (not inside a factory) so Ray can serialise it by
# reference (module + qualname) into every worker process.
# ---------------------------------------------------------------------------
class TransfusionActorRolloutRefWorker(ActorRolloutRefWorker):
    """``ActorRolloutRefWorker`` + a synchronous ``generate_sequences``.

    ``self.rollout`` is the :class:`recipe.transfusion.rollout.TransfusionRollout`
    built in :meth:`ActorRolloutRefWorker.init_model` (``get_rollout_class``).
    ``DP_COMPUTE_PROTO`` chunks the input ``DataProto`` across the worker group
    (keeping ``non_tensor_batch``), runs the diffusion sampler on each shard, and
    concatenates the per-rank ``DataProto`` outputs.
    """

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        return self.rollout.generate_sequences(prompts)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def validate_transfusion_sft(self, payload):
        """Run SFT-style worker-local validation on each rank's own shard.

        The payload contains only paths/scalars; dataset decode happens on the
        worker, avoiding the slow driver->Ray multimodal ``DataProto`` path.
        """
        from recipe.transfusion.validation import validate_shard

        return validate_shard(self, payload)

    # ------------------------------------------------------------------
    # Transfusion-native log-prob / actor-update proxies.
    #
    # verl's stock ``compute_log_prob`` / ``update_actor`` delegate to the FSDP
    # engine's ``infer_batch`` / ``train_batch``, which assume a standard HF
    # causal-LM (``output.logits``). The Transfusion model returns a
    # ``TransfusionOutput`` (diffusion + LM loss, no ``logits``), so those paths
    # crash. These proxies compute the two-headed GRPO signals directly:
    #   * the action (trajectory) importance ratio uses the closed-form x0
    #     Gaussian log-ratio (docs/SINGLE_STEP_X0_GRPO.md);
    #   * the CoT importance ratio uses the plain per-token probability ratio.
    # ------------------------------------------------------------------
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_transfusion_cot_logp(self, data: DataProto) -> DataProto:
        """Behaviour-policy per-token CoT log-prob + entropy for the GRPO ratio.

        Scores the sampled CoT tokens under the ROLLOUT (behaviour) policy — the
        model that actually generated them — conditioned on the sampled clean
        trajectory ``x0_normed`` (matching how the CoT was sampled). Returns
        per-token ``old_log_probs`` and ``entropys`` ``[B, R]`` (padding zeroed via
        ``response_mask``). The CoT token entropy is what verl aggregates into
        ``actor/entropy`` (previously hard-coded to 0). The action's behaviour
        log-prob is the SDE record recorded during rollout, so only the CoT half
        is (re)scored here.

        Runs in chunks of ``rollout.log_prob_micro_batch_size_per_gpu`` samples so
        each sample's video passes through the vision tower separately instead of
        one giant activation tensor (see docs/BATCH_SIZE.md).
        """
        nt = data.non_tensor_batch
        responses = data.batch.get("responses")
        response_mask = data.batch.get("response_mask")
        if responses is None:
            return DataProto.from_dict(tensors={
                "old_log_probs": torch.zeros(1, 1),
                "entropys": torch.zeros(1, 1),
            })
        b, r = responses.shape
        log_probs = torch.zeros((b, r), dtype=torch.float32)
        entropys = torch.zeros((b, r), dtype=torch.float32)
        try:
            policy = self.rollout.policy  # behaviour (eager) model that sampled the data
            ti_list = list(nt["transfusion_inputs"]) if "transfusion_inputs" in nt else []
            x0_list = list(nt["x0_normed"]) if "x0_normed" in nt else []
            if ti_list and x0_list:
                device = next(policy.model.parameters()).device
                micro_bsz = self.config.rollout.log_prob_micro_batch_size_per_gpu or b
                with torch.no_grad():
                    for start in range(0, b, micro_bsz):
                        end = min(start + micro_bsz, b)
                        # np.array(..., copy=True) detaches from DataProto's
                        # (non-writable) numpy storage; torch.from_numpy on the
                        # fresh copy avoids PyTorch's "given NumPy array is not
                        # writable" UserWarning without sharing memory.
                        x0 = torch.stack(
                            [torch.from_numpy(np.array(x0_list[i], dtype=np.float32))
                             for i in range(start, end)], dim=0,
                        ).to(device).float()
                        cot_ids = responses[start:end].to(device)
                        tlp, tent = policy.cot_logprob(
                            ti_list[start:end], x0, cot_ids, device=device,
                        )
                        log_probs[start:end] = tlp.float().cpu()
                        entropys[start:end] = tent.float().cpu()
            if response_mask is not None:
                m = response_mask.float()
                log_probs = log_probs * m
                entropys = entropys * m
        except torch.OutOfMemoryError:
            # GPU memory exhaustion must be fatal (crash the run), not silently
            # skipped -- a silent zero-fallback here would corrupt the PPO ratio
            # for this step without any visible signal.
            raise
        except Exception as e:
            logger.warning("compute_transfusion_cot_logp failed (using zeros): %s", e)
        return DataProto.from_dict(tensors={
            "old_log_probs": log_probs,
            "entropys": entropys,
        })

    def _hybrid_loss_cfg(self) -> dict:
        """Assemble the hybrid-GRPO loss hyper-parameters from the actor config."""
        actor = self.config.actor
        rollout = self.config.rollout

        def _get(obj, name, default):
            try:
                val = getattr(obj, name)
            except Exception:
                val = None
            return default if val is None else val

        return {
            "action_clip_low": float(_get(rollout, "rl_action_clip_low", None) or _get(actor, "clip_ratio", 0.2)),
            "action_clip_high": (_get(rollout, "rl_action_clip_high", None)
                                 if _get(rollout, "rl_action_clip_high", None) is not None
                                 else _get(actor, "clip_ratio_high", None)),
            "cot_clip_low": float(_get(rollout, "rl_cot_clip_low", None) or _get(actor, "clip_ratio", 0.2)),
            "cot_clip_high": (_get(rollout, "rl_cot_clip_high", None)
                              if _get(rollout, "rl_cot_clip_high", None) is not None
                              else (_get(rollout, "rl_action_clip_high", None)
                                    if _get(rollout, "rl_action_clip_high", None) is not None
                                    else _get(actor, "clip_ratio_high", None))),
            "entropy_coef": float(_get(actor, "entropy_coeff", 0.0)),
            "timestep_fraction": float(_get(rollout, "rl_timestep_fraction", 1.0)),
            "action_coef": float(_get(rollout, "rl_action_coef", 1.0)),
            "cot_coef": float(_get(rollout, "rl_cot_coef", 1.0)),
            # KL-to-REFERENCE penalty (recipe-level rollout.rl_kl_coef -- NOT
            # verl's actor.use_kl_loss, which would build the stock HF-logits
            # reference worker). Applied per head with the same machinery as the
            # importance ratios: closed-form Gaussian KL for the trajectory,
            # k3 token-probability KL for the CoT (see
            # loss.hybrid_grpo_loss_from_rollout). The per-head overrides
            # (rl_action_kl_coef / rl_cot_kl_coef) fall back to the shared
            # rl_kl_coef when unset (None).
            "kl_coef": float(_get(rollout, "rl_kl_coef", 0.0)),
            "action_kl_coef": float(_get(rollout, "rl_action_kl_coef", float(_get(rollout, "rl_kl_coef", 0.0)))),
            "cot_kl_coef": float(_get(rollout, "rl_cot_kl_coef", float(_get(rollout, "rl_kl_coef", 0.0)))),
            # Diagnostic CoT new-vs-old approx-KL estimator ("k1"/"k3"; metrics
            # only, see loss.hybrid_grpo_loss_from_rollout).
            "cot_kl_estimator": str(_get(rollout, "rl_cot_kl_estimator", "k1")),
            # Cap on |log(pi/ref)| for CoT tokens (soft clip weight); 5.0 = off.
            "cot_max_log_ratio": float(_get(rollout, "rl_cot_max_log_ratio", 5.0)),
            # Skip all CoT computation (loss, KL, ref-logprob, forward) when true.
            "train_only_action": bool(_get(rollout, "train_only_action", False)),
        }

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_transfusion_actor(self, data: DataProto) -> DataProto:
        """Combined GRPO actor update: closed-form action ratio + token-prob CoT ratio.

        For each micro-batch the differentiable model quantities (the per-step x0
        re-predictions and the per-token CoT log-probs) are computed INSIDE one
        ``model(rl_score_fn=...)`` call so FSDP-1 gathers the (root) parameters the
        denoiser / ``cot_logprob`` submodules need; the GRPO clipped surrogate then
        runs on the returned tensors (see
        :func:`recipe.transfusion.loss.hybrid_grpo_loss_from_rollout`). Gradients
        are accumulated across chunks of ``actor.ppo_micro_batch_size_per_gpu`` and
        applied with one clipped optimizer step.
        """
        from recipe.transfusion.loss import (
            collate_flow_grpo_records,
            hybrid_grpo_loss_from_rollout,
        )

        nt = data.non_tensor_batch
        advantages = data.batch.get("advantages")
        responses = data.batch.get("responses")
        response_mask = data.batch.get("response_mask")
        old_log_probs = data.batch.get("old_log_probs")
        ti_list = list(nt["transfusion_inputs"]) if "transfusion_inputs" in nt else []
        records = list(nt["flow_grpo_record"]) if "flow_grpo_record" in nt else []
        x0_list = list(nt["x0_normed"]) if "x0_normed" in nt else []
        b = len(ti_list)

        agg: dict = {}
        if b == 0 or not records or advantages is None:
            logger.warning(
                "update_transfusion_actor: missing SDE record / advantages "
                "(b=%d, records=%d) -- skipping the update.", b, len(records),
            )
            # Keep the (collective) actor->rollout weight sync aligned across
            # ranks: peers that DID update will enter the FSDP full-state-dict
            # all-gather below, so an empty-shard rank must join it too.
            synced = self._sync_actor_weights_to_rollout()
            out = self._collect_actor_metrics({}, 0.0)
            out["rollout_synced"] = 1.0 if synced else 0.0
            return self._actor_metrics_proto(out)

        micro_bsz = self.config.actor.ppo_micro_batch_size_per_gpu or b
        cfg = self._hybrid_loss_cfg()
        grad_norm = 0.0
        synced = False
        _skipped_steps = 0
        _t0 = time.time()
        _t_ref = _t_loop = _t_step = _t_sync = 0.0
        # ---- off-policy: split the per-GPU shard into mini-batches, each with
        # its own optimizer step. ppo_mini_batch_size (prompts) < train_batch_size
        # => multiple steps: the first is on-policy, the rest are slightly off-
        # policy (the importance ratio corrects for the weight drift). When
        # ppo_mini_batch_size == train_batch_size (or is unset), num_mini == 1
        # and the behaviour is identical to the old single-step update.
        try:
            train_bs = int(self.config.data.train_batch_size)
        except Exception:
            # ActorRolloutRefWorker's self.config is the actor_rollout_ref CONFIG
            # NODE (RayPPOTrainer passes config.actor_rollout_ref) — it has NO
            # ``data`` section, so the lookup throws and must NOT silently fall
            # back to ``b`` (the per-GPU shard) — that misreads train_bs as the
            # SHARD size (train_batch_size*rollout.n/world) instead of the global
            # prompt count, and (whenever shard > ppo_mini) wrongly triggers the
            # off-policy multi-step path (e.g. a perfectly on-policy 32==32
            # config was reported/executed as 2 mini-batches).
            # Reconstruct the GLOBAL train_batch_size from the shard instead:
            #   shard = train_batch_size * rollout.n / world_size.
            import torch.distributed as _dist

            n_rollout = int(getattr(self.config.rollout, "n", 1) or 1)
            world = (
                _dist.get_world_size()
                if _dist.is_available() and _dist.is_initialized()
                else 1
            )
            train_bs = max(1, int(round(b * world / max(1, n_rollout))))
        ppo_mini = int(getattr(self.config.actor, "ppo_mini_batch_size", train_bs)) or train_bs
        ppo_epochs = int(getattr(self.config.actor, "ppo_epochs", 1)) or 1
        num_mini = max(1, train_bs // max(1, ppo_mini))
        mini_bsz = max(1, b // num_mini)
        if num_mini > 1:
            logger.warning(
                "[off-policy] b=%d mini_bsz=%d num_mini=%d epochs=%d (ppo_mini=%d train_bs=%d)",
                b, mini_bsz, num_mini, ppo_epochs, ppo_mini, train_bs,
            )
        else:
            logger.warning(
                "[on-policy] b=%d mini_bsz=%d num_mini=1 epochs=%d "
                "(ppo_mini=%d train_bs=%d) — 1 optimizer step per rollout.",
                b, mini_bsz, ppo_epochs, ppo_mini, train_bs,
            )
        try:
            policy = self.rollout.policy  # collator (tokenizer / image processor)
            model = self._get_actor_model()  # FSDP-wrapped actor
            optimizer = self._get_actor_optimizer()
            _p = next(model.parameters(), None)
            if _p is None:
                raise RuntimeError(
                    "update_transfusion_actor: actor model has NO parameters -- "
                    "the checkpoint/model was not loaded into the actor engine "
                    "(parameters() is empty). Aborting the update instead of "
                    "silently training nothing."
                )
            device = _p.device
            # Frozen reference for the KL penalty: deep-copied from the rollout
            # policy at the FIRST update, i.e. BEFORE any actor->rollout weight
            # sync -- so it stays the initial (SFT) policy for the whole run.
            ref_model = self._get_ref_model() if max(
                cfg.get("kl_coef", 0.0),
                cfg.get("action_kl_coef", 0.0),
                cfg.get("cot_kl_coef", 0.0),
            ) > 0.0 else None
            # verl-style batched ref-logprob phase: compute the reference's x0
            # predictions + CoT log-probs for ALL samples ONCE (no grad, chunked at
            # log_prob_micro_batch_size_per_gpu) instead of re-running a full
            # reference forward inside every actor micro-batch. Kept OUT of the
            # grad-heavy loop below (lower peak memory) and batched (faster).
            logger.warning(
                "[mem] update_actor entry (pre-cache-flush): alloc=%.2fGB reserved=%.2fGB b=%d micro_bsz=%d",
                torch.cuda.memory_allocated(device) / 1e9,
                torch.cuda.memory_reserved(device) / 1e9, b, micro_bsz,
            )
            # Release the ~22-34GB of DEAD reserved cache left by the preceding
            # rollout decode / reward passes. Without this flush the actor's big
            # backward allocations (the B x T x vocab CoT logits block, ~12.6GB
            # at micro=16) miss on the first cudaMalloc EVERY step, and the run
            # only survives via PyTorch's warn-then-empty_cache-retry -> the
            # log floods with "memory allocation failed with OOM" W804 warnings.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x0_ref_all, cot_ref_all = self._precompute_ref_quantities(
                ref_model, ti_list, records, responses, x0_list, cfg, device,
                skip_cot=cfg.get("train_only_action", False),
            )
            _t_ref = time.time() - _t0
            logger.warning(
                "[mem] after ref precompute: alloc=%.2fGB peak=%.2fGB (t=%.1fs)",
                torch.cuda.memory_allocated(device) / 1e9,
                torch.cuda.max_memory_allocated(device) / 1e9, _t_ref,
            )
            # Flush the ref-forward's transient cache as well (x0_ref_all /
            # cot_ref_all are still ALLOCATED -- empty_cache never frees live
            # tensors, only unreferenced blocks) so the grad loop starts with
            # maximal free device memory.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _t_loop_start = time.time()
            model.train()

            for epoch_idx in range(ppo_epochs):
                for mb_start in range(0, b, mini_bsz):
                    mb_end = min(mb_start + mini_bsz, b)
                    mb_size = mb_end - mb_start
                    optimizer.zero_grad()
                    for start in range(mb_start, mb_end, micro_bsz):
                        end = min(start + micro_bsz, mb_end)
                        chunk_b = end - start
                        logger.warning(
                            "[mem] micro-batch [%d:%d] start: alloc=%.2fGB peak=%.2fGB",
                            start, end,
                            torch.cuda.memory_allocated(device) / 1e9,
                            torch.cuda.max_memory_allocated(device) / 1e9,
                        )
                        torch.cuda.reset_peak_memory_stats(device)
                        # Weight by mini-batch fraction so the accumulated gradient
                        # matches a mini-batch mean-loss backward (NOT the full batch).
                        weight = chunk_b / mb_size

                        batch = policy.collate(ti_list[start:end])
                        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                                 for k, v in batch.items()}
                        rollout = collate_flow_grpo_records(records[start:end], device=device)

                        # GRPO advantage is outcome-level (constant over a sample's valid
                        # tokens): reduce to one scalar per sample for both heads.
                        adv_chunk = advantages[start:end].to(device).float()
                        if response_mask is not None:
                            rm = response_mask[start:end].to(device).float()
                            adv_sample = (adv_chunk * rm).sum(1) / rm.sum(1).clamp(min=1.0)
                        else:
                            adv_sample = adv_chunk.mean(1) if adv_chunk.ndim > 1 else adv_chunk

                        # SPLIT GRPO advantage (see _attach_rm_scores): when the
                        # reward driver computed per-head group-relative advantages,
                        # use them so the action head sees ONLY the planning signal
                        # and the CoT head ONLY the prediction signal. Otherwise both
                        # heads share a single scalar -- which is either the GDPO
                        # reward-normalized advantage (rl_reward_norm) when present,
                        # or verl's own group-relative advantage (legacy behaviour).
                        adv_action = adv_sample
                        adv_cot = adv_sample
                        if "adv_action" in nt and "adv_cot" in nt:
                            try:
                                # np.array (not asarray) on purpose: nt[...] is a
                                # Ray object-store zero-copy view (read-only) and
                                # torch.from_numpy would trigger the
                                # "non-writable array" UserWarning. These are tiny
                                # (micro-batch-sized) vectors, the copy is free.
                                adv_action = torch.from_numpy(
                                    np.array(nt["adv_action"][start:end], dtype=np.float32)
                                ).to(device)
                                adv_cot = torch.from_numpy(
                                    np.array(nt["adv_cot"][start:end], dtype=np.float32)
                                ).to(device)
                            except Exception as e:  # pragma: no cover - never break the update
                                logger.warning("split-advantage slice failed, falling back: %s", e)
                                adv_action = adv_sample
                                adv_cot = adv_sample
                        elif "gdpo_adv" in nt:
                            # GDPO reward-normed scalar advantage (rl_reward_norm),
                            # shared by both heads when split advantage is off.
                            try:
                                adv_gdpo = torch.from_numpy(
                                    np.array(nt["gdpo_adv"][start:end], dtype=np.float32)
                                ).to(device)
                                adv_action = adv_gdpo
                                adv_cot = adv_gdpo
                            except Exception as e:  # pragma: no cover - never break the update
                                logger.warning("gdpo-advantage slice failed, falling back: %s", e)
                                adv_action = adv_sample
                                adv_cot = adv_sample

                        cot_ids = responses[start:end]
                        cot_old = old_log_probs[start:end] if old_log_probs is not None \
                            else torch.zeros_like(cot_ids, dtype=torch.float32)
                        cot_mask = response_mask[start:end] if response_mask is not None \
                            else torch.ones_like(cot_ids, dtype=torch.float32)
                        x0n = None
                        if x0_list:
                            x0n = torch.stack(
                                [torch.from_numpy(np.array(x0_list[i], dtype=np.float32))
                                 for i in range(start, end)], dim=0,
                            ).to(device).float()

                        # Slice the precomputed reference quantities for this micro-batch
                        # (x0_ref is [n_train, b, T, D] -> index the batch dim). When the
                        # batched precompute produced them, consume them (ref_model=None);
                        # if it was skipped/failed while the KL is on, fall back to the
                        # loss's own per-micro-batch reference recompute (pass ref_model).
                        x0_ref_chunk = x0_ref_all[:, start:end] if x0_ref_all is not None else None
                        cot_ref_chunk = cot_ref_all[start:end] if cot_ref_all is not None else None
                        have_precomputed_ref = x0_ref_all is not None or cot_ref_all is not None

                        loss, metrics = hybrid_grpo_loss_from_rollout(
                            model, batch, rollout, adv_sample,
                            cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
                            x0_normed=x0n, cfg=cfg,
                            ref_model=None if have_precomputed_ref else ref_model,
                            x0_ref=x0_ref_chunk, cot_ref_logp=cot_ref_chunk,
                            action_advantages=adv_action, cot_advantages=adv_cot,
                        )
                        (loss * weight).backward()
                        logger.warning(
                            "[mem] micro-batch [%d:%d] after backward: alloc=%.2fGB peak=%.2fGB",
                            start, end,
                            torch.cuda.memory_allocated(device) / 1e9,
                            torch.cuda.max_memory_allocated(device) / 1e9,
                        )
                        for k, v in metrics.items():
                            agg[k] = agg.get(k, 0.0) + float(v) * (chunk_b / b)

                    _t_step_start = time.time()
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    )
                    # Non-finite grad_norm (NaN/inf, e.g. a CoT ratio blow-up
                    # past the clamp) => SKIP the step on EVERY rank (all-reduce
                    # MAX flag) so the optimizer stays in sync.
                    grad_ok = grad_norm == grad_norm and grad_norm != float("inf")
                    import torch.distributed as _dist_guard
                    if _dist_guard.is_available() and _dist_guard.is_initialized():
                        try:
                            _flag = torch.tensor([0.0 if grad_ok else 1.0], device=device)
                            _dist_guard.all_reduce(_flag, op=_dist_guard.ReduceOp.MAX)
                            grad_ok = _flag.item() == 0.0
                        except Exception as _e:
                            logger.warning("grad-guard all_reduce failed: %s", _e)
                    if grad_ok:
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        _skipped_steps += 1
                        logger.error(
                            "update_transfusion_actor: non-finite grad_norm=%s — SKIPPING optimizer step (mini-batch %d:%d); the offending batch was already protected by the cot log-ratio clamp if the blow-up came from the CoT ratio.",
                            grad_norm, mb_start, mb_end,
                        )
                    _t_step += time.time() - _t_step_start
                    # ONE-TIME diagnostic (checkpoint forensics): the saved
                    # optim_world_size_*_rank_*.pt shards from previous runs had an
                    # EMPTY `state` dict (only param_groups) — meaning what got
                    # saved was never stepped. This probe logs, after the first
                    # optimizer step of the run, WHETHER the optimizer the update
                    # loop steps is the SAME object the checkpoint manager saves
                    # (engine.optimizer), what type it is, its param-group shape,
                    # and how much AdamW state it actually accumulated. If
                    # optim_state_len=0 here too, the step() path itself is wrong
                    # (fallback fresh AdamW or nothing-to-step); if it is >0 yet
                    # the saved shard still has empty state, the BUG is in the
                    # save path (ShardedOptimStateDictConfig / offload).
                    if not getattr(self, "_optim_probe_done", False):
                        self._optim_probe_done = True
                        _eng = getattr(self.actor, "engine", None)
                        _eng_opt = getattr(_eng, "optimizer", None)
                        _no_grad = sum(
                            1 for g_ in optimizer.param_groups for p_ in g_["params"]
                            if getattr(p_, "grad", None) is None
                        )
                        logger.warning(
                            "[optim-probe] stepped_optimizer type=%s id=%s "
                            "engine.optimizer type=%s id=%s same=%s "
                            "groups=%d group_sizes=%s state_len=%d params_without_grad=%d",
                            type(optimizer).__name__, id(optimizer),
                            type(_eng_opt).__name__, id(_eng_opt),
                            optimizer is _eng_opt,
                            len(optimizer.param_groups),
                            [len(g_["params"]) for g_ in optimizer.param_groups],
                            len(optimizer.state), _no_grad,
                        )
            _t_loop = time.time() - _t_loop_start
            _t_sync_start = time.time()
            # On-policy contract: the NEXT rollout must sample from the freshly
            # updated weights, so push them into the eager rollout copy now.
            synced = self._sync_actor_weights_to_rollout()
            _t_sync = time.time() - _t_sync_start

            # The _sync_actor_weights_to_rollout full-state gather (get_per_tensor_param)
            # and the grad loop leave the CUDA caching allocator holding a large
            # RESERVED pool that otherwise never returns (observed alloc~16GB but
            # reserved 45->72GB across steps on the 76GB cards -> OOM after a few
            # steps even though each single-step peak fits). Release the unused
            # reserved blocks back to the driver between steps: no-op when there is
            # genuinely-live work, and prevents the monotonic reserved climb.
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            # GPU memory exhaustion must be fatal (crash the run), not silently
            # skipped -- an actor-update step that never ran must not be masked
            # as a normal (near-zero-loss) training step.
            raise
        except Exception as e:
            # CUDA OOM surfaces here as a plain RuntimeError (e.g. Triton's
            # "Triton Error [CUDA]: out of memory" from the FLA gated-delta-rule
            # backward), NOT as torch.OutOfMemoryError. If it is swallowed like a
            # recoverable error, the FSDP module is left with its sharded
            # flat-parameters unre-materialised after the aborted backward, so
            # every LATER step dies on `next(model.parameters())` -> StopIteration
            # while training silently spins on error=1.0. Treat CUDA OOM as fatal
            # so we fail loudly at the step that actually ran out of memory.
            if _is_cuda_oom(e):
                logger.error(
                    "CUDA OOM during actor update (fatal): %s", e,
                )
                raise
            logger.error("update_transfusion_actor failed: %s", e, exc_info=True)
            agg["error"] = 1.0

        out = self._collect_actor_metrics(agg, grad_norm)
        out["rollout_synced"] = 1.0 if synced else 0.0
        out["skipped_steps"] = float(_skipped_steps)
        # Fine-grained update_actor breakdown (seconds): ref precompute / grad
        # loop / clip+step / actor->rollout weight sync / total.
        out["time_ref_precompute"] = _t_ref
        out["time_grad_loop"] = _t_loop
        out["time_optim_step"] = _t_step
        out["time_weight_sync"] = _t_sync
        out["time_total"] = time.time() - _t0
        return self._actor_metrics_proto(out)

    def _precompute_ref_quantities(self, ref_model, ti_list, records, responses, x0_list, cfg, device, *, skip_cot=False):
        """Batched, no-grad KL-reference quantities (verl-style ref-logprob phase).

        Computes the frozen reference's per-step x0 predictions (action head) and
        per-token CoT log-probs (CoT head) for ALL samples ONCE, chunked at
        ``rollout.log_prob_micro_batch_size_per_gpu`` -- rather than re-running a
        full reference forward inside every actor micro-batch. This mirrors verl's
        dedicated ``_compute_ref_log_prob`` phase (the reference is computed once
        per batch and reused): the expensive multimodal-context reference forward
        is batched (faster) and kept OUT of the grad-heavy actor loop (lower peak
        memory). The tiny outputs (``x0_ref [n_train, b, T, D]`` and
        ``cot_ref_logp [b, R]``) are then sliced per micro-batch and consumed by
        :func:`hybrid_grpo_loss_from_rollout`. Returns ``(None, None)`` when the KL
        penalty is off (``ref_model is None``) or there is nothing to score.
        """
        from recipe.transfusion.algorithms.diffusion_adapter import TransfusionDiffusionAdapter
        from recipe.transfusion.loss import collate_flow_grpo_records

        b = len(ti_list)
        if ref_model is None or b == 0 or not records:
            return None, None
        policy = self.rollout.policy
        timestep_fraction = float(cfg.get("timestep_fraction", 1.0))
        micro = self.config.rollout.log_prob_micro_batch_size_per_gpu or b
        ref_adapter = TransfusionDiffusionAdapter(ref_model)
        x0_chunks, cot_chunks = [], []
        try:
            with torch.no_grad():
                for start in range(0, b, micro):
                    end = min(start + micro, b)
                    batch = policy.collate(ti_list[start:end])
                    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                             for k, v in batch.items()}
                    rollout = collate_flow_grpo_records(records[start:end], device=device)
                    x_in = rollout["x_in"]
                    sigma = rollout["sigma"]
                    S = x_in.shape[0]
                    n_train = max(1, int(round(S * timestep_fraction)))
                    step_indices = list(range(n_train))
                    ref_ctx = ref_adapter.prepare_context(batch, grad=False)
                    x0_ref = ref_adapter.recompute_x0_preds(
                        ref_ctx, x_in, sigma, step_indices=step_indices,
                    ).detach()  # [n_train, chunk, T, D]
                    x0_chunks.append(x0_ref)
                    if responses is not None and not skip_cot:
                        # x0_normed the CoT was conditioned on (match the loss / rollout).
                        if x0_list:
                            x0n = torch.stack(
                                [torch.from_numpy(np.array(x0_list[i], dtype=np.float32))
                                 for i in range(start, end)], dim=0,
                            ).to(device).float()
                        else:
                            x0n = rollout["next_sample"][-1]
                        # Model-vintage compatibility: drop `state_features` when
                        # the installed cot_logprob does not accept it (removed
                        # in the cleaned AUGUstus_SFT).
                        import inspect as _inspect

                        _ref_kwargs = dict(
                            input_ids=batch["input_ids"],
                            actions_normed=x0n,
                            cot_ids=responses[start:end].to(device),
                            attention_mask=batch.get("attention_mask"),
                            pixel_values=batch.get("pixel_values"),
                            image_grid_thw=batch.get("image_grid_thw"),
                            pixel_values_videos=batch.get("pixel_values_videos"),
                            video_grid_thw=batch.get("video_grid_thw"),
                            second_per_grid_ts=batch.get("second_per_grid_ts"),
                            ctx_embeds=ref_ctx.get("ctx_embeds"),
                        )
                        if "state_features" in _inspect.signature(ref_model.cot_logprob).parameters:
                            _ref_kwargs["state_features"] = batch.get("state_features")
                        cot_ref, _ = ref_model.cot_logprob(**_ref_kwargs)
                        cot_chunks.append(cot_ref.detach())
        except torch.OutOfMemoryError:
            raise
        except Exception as e:
            # A failed reference precompute must not corrupt the update: fall back
            # to the loss's own per-micro-batch recompute by returning nothing.
            logger.warning(
                "ref precompute failed (%s); falling back to in-loss recompute.", e,
            )
            return None, None
        x0_ref_all = torch.cat(x0_chunks, dim=1) if x0_chunks else None
        cot_ref_all = torch.cat(cot_chunks, dim=0) if cot_chunks else None
        return x0_ref_all, cot_ref_all

    # ------------------------------------------------------------------
    # Actor -> rollout weight sync + frozen KL reference management.
    # ------------------------------------------------------------------
    def _get_ref_model(self):
        """Frozen reference policy for the KL penalty (lazy, created ONCE).

        Deep-copies the eager rollout policy model at the FIRST actor update --
        which runs BEFORE the first actor->rollout weight sync, so the copy holds
        the INITIAL (SFT) weights for the rest of the run. Returns ``None`` (and
        disables the KL penalty, loudly) if the copy fails.
        """
        if getattr(self, "_ref_model_failed", False):
            return None
        ref = getattr(self, "_ref_model", None)
        if ref is not None:
            return ref
        try:
            import copy as _copy

            src = self.rollout.policy.model
            ref = _copy.deepcopy(src).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            self._ref_model = ref
            logger.info("KL reference policy created (frozen copy of the initial rollout policy).")
            return ref
        except Exception as e:
            self._ref_model_failed = True
            logger.error(
                "KL reference policy could not be created (%s); the KL-to-reference "
                "penalty is DISABLED for this run.", e,
            )
            return None

    def _sync_actor_weights_to_rollout(self) -> bool:
        """Copy the trained (FSDP) actor weights into the eager rollout policy.

        Uses the engine's ``get_per_tensor_param`` (FULL state dict -- collective
        across the FSDP group, so EVERY rank must call this symmetrically) and
        copies tensors by name into the rollout model's parameters/buffers.
        Returns ``True`` when at least one tensor was copied. No-op when the
        rollout model IS the actor model (single-GPU fallback path).

        Tied-weight handling: the backbone is built with
        ``tie_word_embeddings=True``, so ``qwen_model.lm_head.weight`` IS
        ``qwen_model.model.language_model.embed_tokens.weight`` -- one parameter
        object registered under TWO names. The actor's full ``state_dict`` yields
        BOTH names, while the deduplicated ``named_parameters()`` would hide the
        ``lm_head.weight`` alias and make the name lookup below miss it (warning
        "1 tensor had no matching rollout destination", observed 2026-07-29), even
        though the SHARED storage was in fact updated via the embedding copy --
        harmless but noisy. ``remove_duplicate=False`` exposes the aliases so the
        lookup always succeeds, and each DISTINCT tensor is copied exactly once
        (tracked by object identity), so a tied alias never double-copies the
        shared storage.
        """
        engine = getattr(self.actor, "engine", None)
        if engine is None or not hasattr(engine, "get_per_tensor_param"):
            return False
        try:
            rollout_model = self.rollout.policy.model
        except Exception:
            return False
        if rollout_model is getattr(engine, "module", None):
            return False
        per_tensor_param, _peft = engine.get_per_tensor_param()
        named = dict(rollout_model.named_parameters(remove_duplicate=False))
        named.update(dict(rollout_model.named_buffers()))
        copied = 0
        copied_ids: set = set()
        missing_names: list[str] = []
        with torch.no_grad():
            for name, full in per_tensor_param:
                dst = named.get(name)
                if dst is None:
                    missing_names.append(name)
                    continue
                if id(dst) in copied_ids:
                    continue  # tied alias: shared storage already synced above
                dst.copy_(full.to(device=dst.device, dtype=dst.dtype, non_blocking=True))
                copied_ids.add(id(dst))
                copied += 1
        if missing_names and not getattr(self, "_sync_missing_logged", False):
            self._sync_missing_logged = True
            logger.warning(
                "actor->rollout weight sync: %d tensors had no matching rollout "
                "destination (copied %d). Missing names: %s",
                len(missing_names), copied, missing_names[:20],
            )
        return copied > 0

    @staticmethod
    def _collect_actor_metrics(agg: dict, grad_norm: float) -> dict:
        """Canonical scalar metrics for the actor update (stable keys across paths).

        Building the SAME key set on every code path (normal / skipped / errored)
        keeps the ``DP_COMPUTE_PROTO`` gather concat-compatible across ranks even
        when a rank receives an empty shard.
        """
        return {
            "loss": agg.get("hybrid/loss", 0.0),
            "action_loss": agg.get("hybrid/action_loss", 0.0),
            "cot_loss": agg.get("hybrid/cot_loss", 0.0),
            "action_ratio": agg.get("jit_grpo/ratio_mean", 0.0),
            "cot_ratio": agg.get("cot_grpo/ratio_mean", 0.0),
            "action_clipfrac": agg.get("jit_grpo/clipfrac", 0.0),
            "cot_clipfrac": agg.get("cot_grpo/clipfrac", 0.0),
            "cot_entropy": agg.get("cot_grpo/entropy", 0.0),
            "action_entropy": agg.get("hybrid/action_entropy", 0.0),
            "cot_approx_kl": agg.get("cot_grpo/approx_kl", 0.0),
            "action_ref_kl": agg.get("hybrid/action_kl", 0.0),
            "cot_ref_kl": agg.get("hybrid/cot_kl", 0.0),
            "adv_mean": agg.get("jit_grpo/adv_mean", 0.0),
            "grad_norm": float(grad_norm),
            "error": agg.get("error", 0.0),
            "cot_clamped_frac": agg.get("cot_grpo/clamped_frac", 0.0),
            "cot_nonfinite_masked": agg.get("cot_grpo/nonfinite_masked", 0.0),
        }

    @staticmethod
    def _actor_metrics_proto(metrics: dict) -> DataProto:
        """Wrap scalar actor metrics as ``[1]`` tensors for the DP_COMPUTE gather."""
        return DataProto.from_dict(tensors={
            k: torch.tensor([float(v)], dtype=torch.float32) for k, v in metrics.items()
        })

    def _get_actor_model(self):
        """Return the (FSDP-wrapped) Transfusion model from the actor engine."""
        engine = getattr(self.actor, "engine", None)
        if engine is not None and hasattr(engine, "module"):
            return engine.module
        # Fallback: use the rollout model (no FSDP, single-GPU)
        return self.rollout.policy.model

    def _get_actor_optimizer(self):
        """Return the actor optimizer from the FSDP engine."""
        engine = getattr(self.actor, "engine", None)
        if engine is not None and hasattr(engine, "optimizer"):
            return engine.optimizer
        # Fallback: create a simple optimizer
        model = self._get_actor_model()
        return torch.optim.AdamW(model.parameters(), lr=1e-6)


# ---------------------------------------------------------------------------
# Shims that stand in for the async rollout infrastructure in sync mode.
# ---------------------------------------------------------------------------
def _noop(*args, **kwargs):
    return None


def _group_zscore(values) -> np.ndarray:
    """Group-relative z-score of a 1-D array, mirroring verl's GRPO convention:
    ``(r - mean_g) / (std_g + eps)``. A constant group (std==0) or a single-member
    group yields all-zero advantages (no relative signal to learn from)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size <= 1:
        return np.zeros_like(v)
    m = v.mean()
    s = v.std()
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(v)
    return (v - m) / s


def _grouped_zscore_weighted(terms, n, group_of):
    """Combined advantage where several reward components are EACH z-scored
    within their group first, then weighted-summed.

    This is the core of the GDPO reward normalization (arXiv 2601.05242, Eq. 4-5)
    when several component rewards contribute to one advantage: each component is
    put on a common (unit-variance) scale within its GRPO group before weighting,
    so high-variance components cannot drown lower-variance ones, and the weights
    REALLY control the balance.

    Args:
        terms:     list of ``(values: np.ndarray[len n], weight: float)``.
        n:         number of total samples.
        group_of:  ``{group_key: [indices]}`` map (must exactly partition 0..n-1).
    Returns:
        ``np.ndarray[n]`` ``sum_k weight_k * zscore_group(r_k)`` per sample.
    """
    out = np.zeros(n, dtype=np.float64)
    for _gk, members in group_of.items():
        acc = np.zeros(len(members), dtype=np.float64)
        for values, weight in terms:
            arr = np.asarray(values, dtype=np.float64)
            acc = acc + float(weight) * _group_zscore(arr[members])
        out[members] = acc
    return out


def _batch_whiten(values) -> np.ndarray:
    """Batch-wide advantage normalization (GDPO Eq. 6): ``(v - mean) / (std + eps)``
    over ALL samples in the batch. Keeps the magnitude stable as the number of
    reward components grows. Constant input -> all zeros."""
    v = np.asarray(values, dtype=np.float64)
    if v.size <= 1:
        return np.zeros_like(v)
    m = v.mean()
    s = v.std()
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(v)
    return (v - m) / s


def _default_rollout_n() -> int:
    """Fallback GRPO group size for grouping when a batch carries no per-sample
    ``uid``: prompts are laid out contiguously on the driver in repeats of
    ``rollout.n``, so samples ``[i * n, (i+1) * n)`` share one prompt-group."""
    return 8


class _SyncLLMServerManager:
    """No-op stand-in for ``LLMServerManager`` (no LLM servers in sync mode)."""

    def get_client(self):
        return None

    def get_replicas(self):
        return []

    def __getattr__(self, name):
        # start_profile / stop_profile / sleep / wake / ... -> no-op.
        return _noop


class _SyncRolloutManager:
    """Stand-in for ``AgentLoopManager``: routes generation to the worker group.

    In the async path the agent loop also computes the reward during generation
    and attaches ``rm_scores`` (verl's ``extract_reward`` merely unpacks that). The
    sync worker rollout only produces trajectories / CoT, so we compute the reward
    here (mirroring ``NaiveRewardManager.run_single``) and attach ``rm_scores`` so
    ``fit()`` / ``_validate()`` stay unchanged.
    """

    def __init__(self, trainer):
        self._trainer = trainer
        self._reward_fn = None
        self._reward_fn_loaded = False

    def _reward_function(self):
        if not self._reward_fn_loaded:
            self._reward_fn_loaded = True
            try:
                from verl.trainer.ppo.reward import get_custom_reward_fn

                self._reward_fn = get_custom_reward_fn(self._trainer.config)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("Sync reward: could not load custom reward function: %s", e)
                self._reward_fn = None
        return self._reward_fn

    def generate_sequences(self, batch, **kwargs):
        out = self._trainer.actor_rollout_wg.generate_sequences(batch)
        # fit() reads gen_output.meta_info["timing"]; the sync worker rollout does
        # not populate it, so guarantee the key exists to keep fit() unchanged.
        if getattr(out, "meta_info", None) is None:
            out.meta_info = {}
        out.meta_info.setdefault("timing", {})
        self._attach_rm_scores(out)
        return out

    def _attach_rm_scores(self, batch):
        """Compute the custom reward per sample and attach a token-level
        ``rm_scores`` tensor (score placed on each sample's last valid token)."""
        import numpy as np
        import torch

        if getattr(batch, "batch", None) is None or "rm_scores" in batch.batch:
            return
        responses = batch.batch.get("responses")
        if responses is None:
            return
        reward_fn = self._reward_function()
        tokenizer = getattr(self._trainer, "tokenizer", None)
        n, r = responses.shape[0], responses.shape[1]
        rm = torch.zeros((n, r), dtype=torch.float32)
        resp_mask = batch.batch.get("response_mask")
        nt = batch.non_tensor_batch
        # verl stamps every gen batch with these (ray_trainer.py's fit()/_validate());
        # use them so the best-effort sample logger can tell a training-step rollout
        # apart from an actual validation pass instead of guessing from sample counts
        # (see recipe.transfusion.sample_logging + docs/VAL_LOGGING.md).
        meta_info = getattr(batch, "meta_info", None) or {}
        is_validate = bool(meta_info.get("validate", False))
        global_step = meta_info.get("global_steps")

        # ---- Batch-precompute PDMS for this WHOLE shard in ONE multiprocessed
        # call. PDMS is expensive per sample (lzma-decompress a metric-cache
        # pickle from disk + PDM simulate/score, ~0.5-0.6s each). score_sample()'s
        # on-demand fallback computes it ONE SAMPLE AT A TIME when not pre-supplied
        # -- that path never benefited from compute_pdm_scores_batch's
        # ProcessPoolExecutor parallelism (len(args_list) was always 1). Doing it
        # once here for the whole shard turned a ~2h validation pass into minutes
        # (2026-07-07; see docs/BATCH_SIZE.md and repo memory).
        precomputed_pdms = [None] * n
        precomputed_epdms = [None] * n
        gkeys: list = [None] * n
        if "pred_trajectory" in nt and "reward_model" in nt:
            try:
                from recipe.transfusion import transfusion_reward as _tr
                from recipe.transfusion.reward_function import (
                    resolve_metric_cache_path,
                    resolve_metric_cache_v2_path,
                    strip_weights_for_split,
                )

                gts = [nt["reward_model"][i].get("ground_truth", {}) or {} for i in range(n)]
                metric_cache_path = resolve_metric_cache_path(
                    gts[0].get("metric_cache_path") if gts else None, {}
                )
                valid_idx, valid_traj, valid_fid = [], [], []
                for i in range(n):
                    # The batch PDMS MUST score EXACTLY the trajectory the
                    # reward uses, otherwise the PDMS component of the score
                    # can disagree with what compute_total_reward later
                    # computes from the same sample (the classic "batch PDMS
                    # is wrong" symptom). Reuse the reward's own normalisation
                    # (_default_trajectory) so an invalid/collapsed action
                    # (None, NaN/inf, wrong shape) is defaulted to the same
                    # all-zero [8,3] trajectory the reward would use, instead
                    # of being stacked raw and crashing the whole batch or
                    # silently scoring a different shape.
                    pt_raw = nt["pred_trajectory"][i]
                    fid = gts[i].get("frame_id") or ""
                    if pt_raw is None:
                        continue
                    if not fid:
                        continue
                    pt = _tr._default_trajectory(pt_raw).astype(np.float32)
                    # PDMS needs [T,3] (x, y, heading); ensure the heading
                    # axis exists so a 2D trajectory can't make np.stack
                    # below dimension-mismatch and nuke the whole batch.
                    if pt.ndim == 2 and pt.shape[1] < 3:
                        pt = np.pad(pt, ((0, 0), (0, 3 - pt.shape[1])))
                    valid_idx.append(i)
                    valid_traj.append(pt)
                    valid_fid.append(fid)
                if valid_traj and metric_cache_path:
                    _t_pdms = time.time()
                    pdms_values = _tr.pdms_trajectory_reward(
                        np.stack(valid_traj), valid_fid, metric_cache_path,
                        num_workers=min(len(valid_traj), 128),
                    )
                    logger.warning(
                        "[time] reward pdms batch: %.1fs (%d samples)",
                        time.time() - _t_pdms, len(valid_traj),
                    )
                    for idx, val in zip(valid_idx, pdms_values):
                        precomputed_pdms[idx] = val
                # ---- Batch-precompute the OFFICIAL NAVSIM v2 EPDMS the
                # same way (per-frame, with_gt_prev_ec in training), but
                # ONLY when this split actually computes EPDMS at all
                # (weights.epdms != 0 / val_epdms; see tr.epdms_enabled).
                if valid_traj:
                    from omegaconf import OmegaConf
                    try:
                        _rk = self._trainer.config.reward.custom_reward_function.reward_kwargs
                        _rk = OmegaConf.to_container(_rk, resolve=True) or {}
                    except Exception:
                        _rk = {}
                    _raw_w = {
                        **(_rk.get("weights") or {}),
                        **(gts[0].get("weights") or {}),
                    }
                    _w = _tr.RewardWeights.from_dict(
                        strip_weights_for_split(_raw_w, is_validate=is_validate)
                    )
                    if _tr.epdms_enabled(_w, is_validate=is_validate):
                        metric_cache_v2 = resolve_metric_cache_v2_path(
                            gts[0].get("metric_cache_v2_path") if gts else None, _rk
                        )
                        if metric_cache_v2:
                            from recipe.transfusion import epdms_eval
                            _t_ep = time.time()
                            ep_rows = epdms_eval.epdms_score_per_frame(
                                np.stack(valid_traj), valid_fid, metric_cache_v2,
                                num_workers=min(len(valid_traj), 128),
                                with_gt_prev_ec=not is_validate,
                            )
                            logger.warning(
                                "[time] reward epdms batch: %.1fs (%d samples, cache=%s)",
                                time.time() - _t_ep, len(valid_traj), metric_cache_v2,
                            )
                            for idx, row in zip(valid_idx, ep_rows):
                                if row is None:
                                    continue
                                precomputed_epdms[idx] = row
                        else:
                            logger.warning(
                                "Sync reward: EPDMS enabled but no v2 metric cache (data.metric_cache_*_v2 / reward_kwargs); skipping EPDMS."
                            )
            except Exception as e:  # pragma: no cover - never break the reward loop
                logger.warning(
                    "Sync reward: batched PDMS/EPDMS precompute failed (falling back to per-sample PDMS): %s", e
                )

        # ---- Precompute the batch-relative LEFT-padding length for prompt
        # logging. ``TransfusionCollator`` (Transfusion/transfusion/dataset.py)
        # left-pads every sample's input_ids to the LONGEST sample in its batch
        # with pad_id before the model ever sees it -- that's the ACTUAL tensor
        # `policy.collate()` builds for generation. Each
        # transfusion_inputs[i]["input_ids"] here is the RAW per-sample
        # tokenization (dataset __getitem__ calls the processor with a batch of
        # 1, so there is nothing to pad against), so without reproducing that
        # same padding, the logged prompt is missing the leading <|endoftext|>
        # padding SFT's own logging shows for shorter-than-longest samples.
        prompt_max_len = 0
        pad_id = tokenizer.pad_token_id if tokenizer is not None else None
        if "transfusion_inputs" in nt:
            try:
                for i in range(n):
                    ti = nt["transfusion_inputs"][i]
                    ids = ti.get("input_ids") if isinstance(ti, dict) else None
                    if ids is not None:
                        prompt_max_len = max(prompt_max_len, len(ids))
            except Exception:
                prompt_max_len = 0

        extra: dict[str, list] = {}
        _t_loop = time.time()
        for i in range(n):
            vlen = int(resp_mask[i].sum().item()) if resp_mask is not None else r
            vlen = max(1, min(vlen, r))
            score = 0.0
            result = None
            prompt_len = None
            if reward_fn is not None:
                try:
                    ids = responses[i][:vlen]
                    solution = tokenizer.decode(ids, skip_special_tokens=True) if tokenizer is not None else ""
                    gt = nt["reward_model"][i]["ground_truth"] if "reward_model" in nt else {}
                    # The trajectory rollout keeps ``pred_trajectory`` as its own
                    # non_tensor key (not in extra_info, to keep union happy);
                    # inject it into extra_info for the reward function here.
                    extra_info = dict(nt["extra_info"][i]) if "extra_info" in nt else {}
                    # GRPO group id for the in-group trajectory overlay plots
                    # (sample_logging): the `uid` of the originating prompt;
                    # fall back to the contiguous repeat layout i // rollout.n
                    # when no uid is attached.
                    gkey = nt["uid"][i] if "uid" in nt else None
                    if gkey is None:
                        try:
                            rollout_n = int(self._trainer.config.actor_rollout_ref.rollout.n)
                        except Exception:
                            rollout_n = 1
                        gkey = i // max(1, rollout_n)
                    extra_info["group_key"] = gkey
                    gkeys[i] = gkey
                    if "pred_trajectory" in nt:
                        extra_info["pred_trajectory"] = nt["pred_trajectory"][i]
                    if "_raw_video_frames" in nt:
                        # Only present for validation samples (see
                        # TransfusionRLDataset.keep_raw_frames); lets the reward
                        # function's best-effort sample logger show the last frame.
                        extra_info["raw_frames"] = nt["_raw_video_frames"][i]
                    extra_info["is_validate"] = is_validate
                    extra_info["global_step"] = global_step
                    if precomputed_pdms[i] is not None:
                        extra_info["pdms_components"] = precomputed_pdms[i]
                    if precomputed_epdms[i] is not None:
                        extra_info["epdms_components"] = precomputed_epdms[i]

                    # Input prompt text + length -- verl's own batch["prompts"] is
                    # a 1-token placeholder (see rollout.py), the REAL model input
                    # lives in transfusion_inputs. Decoded here once so the sample
                    # logger can show it (was missing entirely) and so prompt_len/
                    # cot_len averages surface in val-aux metrics next to ADE/FDE/
                    # PDMS/reward (same reward_extra_keys mechanism, below).
                    # skip_special_tokens=False (unlike the CoT `solution` decode
                    # above) is intentional: the point of logging the prompt is to
                    # see the ACTUAL structure fed to the model -- chat-template
                    # markers (<|im_start|>/<|im_end|>), left-padding
                    # (<|endoftext|>), the per-frame "<X seconds>" timestamps, and
                    # every <|video_pad|> placeholder -- stripping them would hide
                    # exactly what you'd want to debug/verify. Re-applies the same
                    # batch-relative LEFT-padding TransfusionCollator uses (see
                    # prompt_max_len above) so this matches what the model
                    # actually saw, not the un-padded per-sample dataset output.
                    if "transfusion_inputs" in nt:
                        try:
                            ti = nt["transfusion_inputs"][i]
                            prompt_ids = ti.get("input_ids") if isinstance(ti, dict) else None
                            if prompt_ids is not None:
                                prompt_len = int(len(prompt_ids))
                                if tokenizer is not None:
                                    ids_list = (
                                        prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
                                    )
                                    pad_amount = max(0, prompt_max_len - prompt_len)
                                    if pad_amount > 0 and pad_id is not None:
                                        ids_list = [pad_id] * pad_amount + ids_list
                                    extra_info["prompt_text"] = tokenizer.decode(
                                        ids_list, skip_special_tokens=False
                                    )
                        except Exception:
                            pass

                    result = reward_fn(
                        data_source=nt["data_source"][i] if "data_source" in nt else None,
                        solution_str=solution,
                        ground_truth=gt,
                        extra_info=extra_info,
                    )
                    score = float(result["score"]) if isinstance(result, dict) else float(result)
                except Exception as e:  # pragma: no cover - never break the loop
                    logger.warning("Sync reward: sample %d failed: %s", i, e)
                    result, score = None, 0.0
            rm[i, vlen - 1] = score
            if isinstance(result, dict):
                for k, v in result.items():
                    extra.setdefault(k, [0.0] * n)[i] = v
            extra.setdefault("cot_len", [0.0] * n)[i] = float(vlen)
            if prompt_len is not None:
                extra.setdefault("prompt_len", [0.0] * n)[i] = float(prompt_len)
        batch.batch["rm_scores"] = rm
        logger.warning("[time] reward per-sample loop: %.1fs (%d samples)", time.time() - _t_loop, n)
        if extra:
            batch.meta_info["reward_extra_keys"] = list(extra.keys())
            for k, v in extra.items():
                nt[k] = np.array(v, dtype=object)

        # ---- SPLIT GRPO ADVANTAGE (per-head group-relative normalization) ----
        # By default the whole batch shares ONE scalar advantage (computed by
        # verl's compute_advantage on ``rm_scores``) which is then fed to BOTH the
        # action (trajectory) and CoT heads. That misattributes the reward: the
        # noisier / higher-variance component dominates both heads' within-group
        # signal (soft_L2 varies ~2 orders more than PDMS -> the action head ends
        # up imitating the GT instead of climbing PDMS; trajectory outcomes also
        # flood the CoT head). When ``rl_split_advantage`` is enabled we instead
        # compute a SEPARATE per-head reward that only the head can influence and
        # z-score IT within its GRPO group:
        #     A_action_i = zscore_group(reward_action_i)   (planning: soft_L2+PDMS)
        #     A_cot_i    = zscore_group(reward_cot_i)      (prediction+format/verbosity)
        # ``reward_action`` / ``reward_cot`` were baked into the reward dict by
        # compute_total_reward and surfaced here through ``extra``. The advantage
        # tensors are stored on ``nt`` (NOT in ``meta_info['reward_extra_keys']``,
        # so they never show up as train_reward/* metrics).
        _cfg_ra = None
        try:
            _cfg_ra = self._trainer.config.actor_rollout_ref.rollout
        except Exception:
            _cfg_ra = None
        _split = False
        _gnorm = False   # GDPO reward_norm
        if _cfg_ra is not None and not is_validate:

            def _on(v):
                if isinstance(v, str):
                    return v.strip().lower() not in ("0", "false", "no", "off")
                return bool(v)

            _split = _on(getattr(_cfg_ra, "rl_split_advantage", False)
                         if not hasattr(_cfg_ra, "get") else _cfg_ra.get("rl_split_advantage", False))
            _gnorm = _on(getattr(_cfg_ra, "rl_reward_norm", False)
                         if not hasattr(_cfg_ra, "get") else _cfg_ra.get("rl_reward_norm", False))

        # map sample -> group
        group_of = {}
        for i in range(n):
            gk = gkeys[i] if i < len(gkeys) and gkeys[i] is not None else i // max(1, _default_rollout_n())
            group_of.setdefault(gk, []).append(i)

        # ---- SPLIT GRPO ADVANTAGE (per-head group-relative normalization) ----
        # Each head gets its OWN group-relative advantage from only the reward it
        # can influence: the action head from the PLANNING block, the CoT head from
        # the PREDICTION (+ format/verbosity penalties). This stops the two heads'
        # signals polluting each other (see docs/SPLIT_ADVANTAGE.md). Note it does
        # NOT by itself rebalance soft_L2 vs PDMS *inside* the action head -- that
        # is the job of ``rl_reward_norm`` (GDPO) below.
        if _split and "reward_action" in extra and "reward_cot" in extra:
            for k in ("reward_action", "reward_cot"):
                if k not in nt:
                    nt[k] = np.array(extra[k], dtype=object)
            raw_action = np.asarray(extra["reward_action"], dtype=np.float64)
            raw_cot = np.asarray(extra["reward_cot"], dtype=np.float64)
            adv_action = np.zeros(n, dtype=np.float64)
            adv_cot = np.zeros(n, dtype=np.float64)
            for gk, members in group_of.items():
                adv_action[members] = _group_zscore(raw_action[members])
                adv_cot[members] = _group_zscore(raw_cot[members])
            nt["adv_action"] = adv_action.astype(np.float32)
            nt["adv_cot"] = adv_cot.astype(np.float32)

        # ---- GDPO REWARD NORM (rl_reward_norm) ----
        # Group reward-Decoupled Normalization (arXiv 2601.05242): instead of
        # normalizing the TOTAL reward once (which lets a high-variance component
        # dominate and collapses the fine-grained advantage structure), normalize
        # EACH reward component on its OWN within the group first, weight them,
        # then batch-whiten the sum. Component set here = the individual reward
        # signals: prediction (CoT), soft_L2, PDMS. This is what makes the
        # coefficient the REAL lever between soft_L2 and PDMS inside the action
        # head (and fixes the ~100:1 variance flood). Default false.
        if _gnorm:
            _gdpo_terms = []
            _GD_COMP = (
                ("prediction_reward", "coef_prediction"),
                ("soft_l2_reward", "coef_soft_l2"),
                ("pdms_reward", "coef_pdms"),
            )
            for _rk, _ck in _GD_COMP:
                if _rk in extra:
                    _coef = float(extra[_ck][0]) if (_ck in extra and extra.get(_ck)) else 1.0
                    _gdpo_terms.append((np.asarray(extra[_rk], dtype=np.float64), _coef))
            if _gdpo_terms:
                # pre_bn = sum_k coef_k * group_zscore(r_k)   (decoupled normalization)
                _pre = _grouped_zscore_weighted(_gdpo_terms, n, group_of)
                # batch-wise advantage normalization (GDPO Eq. 6)
                nt["gdpo_adv"] = _batch_whiten(_pre).astype(np.float32)

        # Flush this batch's accumulated sample-logger window/means now for
        # TRAINING (one call == one full training step). For VALIDATION, do NOT
        # flush here -- verl chunks the val split into MANY of these calls
        # (data.val_batch_size), and flushing per-chunk would repeatedly
        # overwrite the same step's val_samples/* files + val_reward/* scalars
        # with just that one chunk's data instead of the whole split's (this was
        # a regression from capping/chunking val_batch_size, 2026-07-07).
        # TransfusionRayPPOTrainer (best_ckpt.py) flushes "val" exactly once,
        # right after the FULL validation pass (every chunk) completes.
        if not is_validate:
            try:
                from recipe.transfusion.sample_logging import (
                    flush_metric_means,
                    get_global_sample_logger,
                    should_log_train_samples,
                )

                flush_metric_means("train")
                # Driver-side SwanLab mirror of the train-step artefacts just
                # written to disk (train_samples/*.txt|png, incl. the GRPO group
                # trajectory overlays group_*_trajs.png). The live sl.log inside
                # log_text/log_figure can silently no-op (e.g. swanlab.Image on a
                # matplotlib Figure), so re-publish the on-disk artefacts with the
                # SAME keys -- whatever was generated is uploaded, mirroring the
                # proven val_samples mirror in best_ckpt.TransfusionRayPPOTrainer.
                if should_log_train_samples(int(global_step) if global_step is not None else None):
                    get_global_sample_logger().mirror_step_dir_to_swanlab(
                        int(global_step), prefix="train_samples"
                    )
            except Exception:
                pass


    def __getattr__(self, name):
        return _noop


class _SyncCheckpointManager:
    """No-op stand-in for the async checkpoint/weight-sync manager.

    The REAL actor->rollout weight sync happens inside the worker instead
    (:meth:`TransfusionActorRolloutRefWorker._sync_actor_weights_to_rollout`,
    called after every actor update), so every async replica-lifecycle call
    (``sleep_replicas`` / ``update_weights`` / ``wake_up`` / ...) is a no-op here.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _noop


# ---------------------------------------------------------------------------
# Trainer mixin: replace the async infra in init_workers with the shims above.
# MRO must place this BEFORE RayPPOTrainer.
# ---------------------------------------------------------------------------
class SyncRolloutTrainerMixin:
    """Route rollout generation through the sync worker rollout instead of the
    async LLM-server path (this verl version has no sync trainer path)."""

    def init_workers(self):
        from unittest.mock import patch

        import verl.checkpoint_engine as _ckpt_mod
        import verl.experimental.agent_loop as _agent_mod
        import verl.trainer.ppo.ray_trainer as _rt

        trainer = self

        class _LLMServerShim:
            @classmethod
            def create(cls, *args, **kwargs):
                return _SyncLLMServerManager()

        class _AgentLoopShim:
            @classmethod
            def create(cls, *args, **kwargs):
                return _SyncRolloutManager(trainer)

        # ``init_workers`` references ``LLMServerManager`` as a module global and
        # imports ``AgentLoopManager`` / ``CheckpointEngineManager`` locally from
        # their source modules, so patch each at its lookup site for the duration
        # of the parent call. Everything else (worker groups, actor/ref init,
        # reward loop manager) is built by the real parent implementation.
        with (
            patch.object(_rt, "LLMServerManager", _LLMServerShim),
            patch.object(_agent_mod, "AgentLoopManager", _AgentLoopShim),
            patch.object(_ckpt_mod, "CheckpointEngineManager", _SyncCheckpointManager),
        ):
            super().init_workers()

        logger.info(
            "Transfusion sync-rollout: async LLM-server / AgentLoop / checkpoint "
            "managers replaced with sync shims; generation routed through the "
            "worker-group generate_sequences. Actor->rollout weight sync runs "
            "inside the worker after every actor update."
        )

    # ------------------------------------------------------------------
    # Override verl's HF-causal-LM-specific log-prob / actor-update methods.
    #
    # The stock ``_compute_old_log_prob`` / ``_update_actor`` convert the batch
    # to a TensorDict and delegate to the FSDP engine, which assumes
    # ``output.logits`` (a standard HF causal-LM output). The Transfusion model
    # returns a ``TransfusionOutput`` (diffusion + LM loss, no ``logits``), so
    # those paths crash. These overrides route to the Transfusion-native worker
    # proxies defined on :class:`TransfusionActorRolloutRefWorker`.
    # ------------------------------------------------------------------
    def _compute_old_log_prob(self, batch):
        """Behaviour-policy per-token CoT log-prob + entropy for the PPO ratio.

        FAST PATH: the rollout sampler records the raw-distribution per-token
        log-probs / entropies WHILE generating the CoT (``record_logp``), and
        attaches them to the batch as ``old_log_probs`` / ``entropys`` — the
        exact quantities the teacher-forced recompute produces, for free. When
        present, skip the worker recompute (the whole timing_s/old_log_prob
        phase) and hand them to verl directly. FALLBACK: route to
        ``compute_transfusion_cot_logp`` (``DP_COMPUTE_PROTO``), which scores
        the sampled CoT under the rollout policy conditioned on the sampled
        trajectory. The action head's behaviour log-prob is the SDE record
        captured during rollout in both cases.
        """
        keys = batch.batch.keys() if getattr(batch, "batch", None) is not None else []
        if "old_log_probs" in keys and "entropys" in keys:
            out = DataProto.from_dict(tensors={
                "old_log_probs": batch.batch["old_log_probs"],
                "entropys": batch.batch["entropys"],
            })
            # Remove them from the main batch so the later ``batch.union(out)``
            # in fit() does not collide on duplicate keys.
            batch.batch.pop("old_log_probs")
            batch.batch.pop("entropys")
            return out, 0.0
        # train_only_action: no CoT was generated, skip the recompute and
        # return zero old_log_probs / entropys matching the (all-pad) responses.
        # NOTE: this runs on the TRAINER, where self.config is the FULL Hydra
        # config — the rollout knobs live under ``actor_rollout_ref.rollout``;
        # there is NO top-level ``rollout`` key (OmegaConf struct mode raises
        # ConfigAttributeError on it). getattr-chains are safe because
        # ConfigAttributeError subclasses AttributeError (returns the default).
        _arr = getattr(self.config, "actor_rollout_ref", None)
        _rollout_cfg = getattr(_arr, "rollout", None) if _arr is not None else None
        if _rollout_cfg is not None and bool(getattr(_rollout_cfg, "train_only_action", False)):
            responses = batch.batch.get("responses")
            if responses is not None:
                b, r = responses.shape
                return DataProto.from_dict(tensors={
                    "old_log_probs": torch.zeros((b, r), dtype=torch.float32),
                    "entropys": torch.zeros((b, r), dtype=torch.float32),
                }), 0.0
        return self.actor_rollout_wg.compute_transfusion_cot_logp(batch), 0.0

    def _update_actor(self, batch):
        """Update the Transfusion actor via the combined-GRPO worker proxy.

        Calls ``TransfusionActorRolloutRefWorker.update_transfusion_actor``
        (registered with ``DP_COMPUTE_PROTO``), which computes the closed-form
        action ratio + token-probability CoT ratio and applies one clipped
        optimizer step. The worker returns per-rank scalar metric tensors; they
        are averaged and re-keyed under ``actor/``.
        """
        actor_output = self.actor_rollout_wg.update_transfusion_actor(batch)

        metrics = {}
        if getattr(actor_output, "batch", None) is not None:
            for k in actor_output.batch.keys():
                metrics[f"actor/{k}"] = float(actor_output.batch[k].float().mean().item())
        metrics.setdefault("actor/mfu", 0.0)
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})


__all__ = [
    "TransfusionActorRolloutRefWorker",
    "SyncRolloutTrainerMixin",
]
