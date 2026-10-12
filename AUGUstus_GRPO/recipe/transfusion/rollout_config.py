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

"""RolloutConfig subclass for the (synchronous) Transfusion diffusion rollout.

verl's stock :class:`~verl.workers.config.rollout.RolloutConfig` is built for the
async LLM-server rollouts (vLLM / SGLang / TRT-LLM ``ServerAdapter``) and, as of
this verl version, is strict in two ways that are incompatible with the
Transfusion recipe:

1. It rejects any field it does not declare, so the diffusion-RL selector fields
   that :class:`recipe.transfusion.rollout.TransfusionRollout` reads
   (``rl_num_steps`` / ``rl_noise_level``) raise
   ``TypeError: RolloutConfig.__init__() got an unexpected keyword argument``.
2. Its ``__post_init__`` hard-raises on ``mode='sync'`` ("Rollout mode 'sync'
   has been removed"). The Transfusion trajectory rollout is genuinely
   synchronous — it runs ``generate_trajectory`` directly (``BaseRollout``'s sync
   ``generate_sequences`` path) rather than an async token server — and registers
   itself under ``("transfusion", "sync")``.

This subclass declares the three extra fields and permits ``mode='sync'`` while
preserving every other RolloutConfig validation. It is wired in via
``actor_rollout_ref.rollout._target_`` in the recipe configs, so no verl core
file is edited (consistent with the rest of the recipe).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from verl.workers.config.rollout import RolloutConfig


@dataclass
class TransfusionRolloutConfig(RolloutConfig):
    """``RolloutConfig`` extended for the synchronous Transfusion rollout."""

    # Diffusion-RL algorithm selector. This release ships the ORIGINAL
    # token-level GRPO only (the hybrid action+CoT head); the sibling yaml
    # configs still carry the ``rl_algorithm: null`` knob (and the
    # TRANSFUSION_RL_ALGORITHM env fallback) for compatibility — any non-None
    # value raises at rollout build time.
    rl_algorithm: Optional[str] = None
    # Denoising steps per trajectory for TRAINING rollouts (the SDE sampler
    # horizon of the GRPO rollout).
    rl_num_steps: int = 10
    # Denoising steps for IN-TRAINING VALIDATION rollouts (verl's ``_validate``
    # sets ``meta_info['validate']=True``). ``None`` -> fall back to
    # ``rl_num_steps``.
    rl_val_num_steps: Optional[int] = None
    # Flow-GRPO SDE stochasticity (0 -> deterministic ODE collection). This is
    # the SCALAR HYPER-PARAMETER of ``rl_noise_schedule`` below and scales the
    # per-step std LINEARLY in every schedule: ``a`` for flow_grpo, ``eta`` for
    # dancegrpo, ``a`` for lambda_t (g = a*sqrt(t); a 1-step rollout has std = a).
    # A SCALAR (all dims share it) or a PER-DIMENSION LIST ``[nl_x, nl_y,
    # nl_heading]`` (each action dim gets its own std scale).
    rl_noise_level: Any = 0.7
    # Reverse-time SDE diffusion coefficient g(t) SCHEDULE for TRAINING rollouts:
    #   ``flow_grpo`` -> g = a*sqrt(t/(1-t))   (diverges at t=1; sigma_max guard)
    #   ``dancegrpo`` -> g = eta               (constant, finite everywhere)
    #   ``lambda_t``  -> g = a*sqrt(t)         (finite at t=1, ->0 at t=0)
    # ``rl_noise_level`` is its scalar hyper-parameter. Env fallback
    # ``TRANSFUSION_RL_NOISE_SCHEDULE``.
    rl_noise_schedule: str = "flow_grpo"
    # Optional SEPARATE CoT sampling knobs for the in-training VALIDATION
    # rollouts (``None`` -> reuse the training-rollout ``temperature`` / ``top_p``
    # / ``top_k``). The SFT-style validation path (validation.py) forwards them
    # through the shard payload; without them the worker falls back to
    # $TRANSFUSION_CONFIG's ``cot.cot_*`` supervised-eval defaults, which are
    # less diverse than the training sampler.
    rl_val_cot_temperature: Optional[float] = None
    rl_val_cot_top_p: Optional[float] = None
    rl_val_cot_top_k: Optional[int] = None
    # Optional cap on the number of CoT CONTENT tokens for the IN-TRAINING
    # VALIDATION rollouts. ``None`` -> default to the RL response budget (i.e.
    # ``data.max_response_length``, 256 by default), past which the model would
    # otherwise fall back to its native ``max_cot_tokens`` (512) and pad the whole
    # window with junk after ``</prediction>`` — the observed CoT collapse.
    rl_val_cot_max_len: Optional[int] = None

    # ---- combined-GRPO actor-update knobs (default token-level path) ----
    # Fraction of the recorded SDE steps (the highest-noise prefix) the actor
    # update trains the closed-form action ratio on (1.0 = all steps).
    rl_timestep_fraction: float = 1.0
    # Relative weights of the two GRPO heads in the combined actor loss: the
    # closed-form trajectory (action) ratio vs. the token-probability CoT ratio.
    rl_action_coef: float = 1.0
    rl_cot_coef: float = 1.0
    # KL-to-REFERENCE penalty coefficient (0 = off). Applied per head with the
    # same machinery as the ratios (closed-form Gaussian KL for the trajectory,
    # k3 token-probability KL for the CoT) against a frozen copy of the initial
    # policy. Recipe-level on purpose: verl's actor.use_kl_loss would build the
    # stock HF-logits reference worker, which Transfusion does not support.
    rl_kl_coef: float = 0.0
    # Optional SEPARATE per-head KL coefficients (``None`` -> that head reuses
    # the shared ``rl_kl_coef``). Useful to keep the trajectory head free while
    # reigning in CoT drift (or vice versa) without forking the shared knob:
    #   rl_action_kl_coef: 0.0    -> no trajectory KL even if rl_kl_coef > 0
    #   rl_cot_kl_coef:    0.01  -> CoT KL on, trajectory KL off
    # Both default to ``None`` (= old single-knob behaviour, unchanged results).
    rl_action_kl_coef: Optional[float] = None
    rl_cot_kl_coef: Optional[float] = None
    # Estimator for the DIAGNOSTIC CoT new-vs-old approximate KL metric
    # (cot_grpo/approx_kl -> actor/cot_approx_kl): "k1" = mean(old - new), the
    # PPO convention — cheap but can go negative under finite-sample noise;
    # "k3" = exp(gap) - 1 - gap with gap = old - new (verl's low_var_kl),
    # pointwise non-negative. Metrics only; the loss is unaffected either way.
    rl_cot_kl_estimator: str = "k1"

    # ---- PPO clip ranges (SEPARATE per head) ----
    # clip = [1 - epsilon_low, 1 + epsilon_high]. The action (trajectory) and
    # CoT (token) heads get their OWN lower / upper epsilons, overriding verl's
    # shared actor.clip_ratio / clip_ratio_high for the hybrid-GRPO loss; unset
    # (None) -> fall back to the actor defaults (old behaviour).
    rl_action_clip_low: Optional[float] = None
    rl_action_clip_high: Optional[float] = None
    rl_cot_clip_low: Optional[float] = None
    rl_cot_clip_high: Optional[float] = None
    # Hard cap on the per-token CoT importance log-ratio |log(pi_new/pi_old)|
    # inside the CoT head: values beyond the cap are clamped (not the whole
    # token masked — only the ratio saturates), which guards the head against
    # pathological importance weights when a token flips from ~0 to ~1
    # probability between the rollout and the update (rare but catastrophic
    # for the gradient). 5.0 is loose enough to never bind in healthy runs.
    rl_cot_max_log_ratio: float = 5.0
    # Skip the CoT head loss entirely and train ONLY the action (trajectory)
    # head. The CoT tokens still get ROLLED OUT (the model still produces its
    # reasoning; the ratios are still recorded for diagnostics) but they
    # contribute nothing to the actor gradient. Useful to isolate the effect
    # of the trajectory head or to freeze CoT behaviour while the action head
    # adapts. Off by default (False = legacy both-heads behaviour).
    train_only_action: bool = False

    # ------------------------------------------------------------------
    # NOTE ON THE NOISE / KL KNOB EVOLUTION (for future readers):
    #
    #   * ``rl_noise_level``            — the base scalar (or per-dim list) that
    #                                    every schedule multiplies.
    #
    # The KL family mirrors this pattern on the loss side:
    #
    #   * ``rl_kl_coef``                — the shared knob (both heads).
    #   * ``rl_action_kl_coef`` /       — optional per-head overrides; None
    #     ``rl_cot_kl_coef``              falls back to the shared value.
    #   * ``rl_cot_max_log_ratio``      — per-token importance clamp for the
    #                                    CoT head only (loss-side guard; the
    #                                    action head's closed-form Gaussian
    #                                    ratios do not need it).
    #   * ``train_only_action``         — CoT-head kill switch for ablations.
    #
    # Nothing here is read by verl core; ``TransfusionRollout`` /
    # ``TransfusionPolicy`` / the sync driver consume these via getattr-style
    # access with the same defaults, so REMOVING a knob from a config is
    # always safe and old checkpoints/configs remain loadable.
    #
    # When adding a NEW knob: keep it Optional with a None/False default,
    # document the env-var fallback if one exists, and make sure the rollout
    # and the sync driver BOTH tolerate its absence (getattr / .get with the
    # same default) so mixed-version checkpoints never break mid-run.
    #
    # ------------------------------------------------------------------

    # ---- per-head / GDPO advantage switches (see docs/SPLIT_ADVANTAGE.md) ----
    # When true, each hybrid-GRPO head gets its OWN group-relative advantage
    # derived from only the sub-reward it can influence (action head -> the
    # planning block soft_L2 + PDMS; CoT head -> the prediction block + format
    # penalties) instead of a single shared scalar computed from the total score.
    # Default false = legacy single-scalar behaviour. Consumed driver-side in
    # sync_driver.py (also via getattr/.get, so both are safe to omit).
    rl_split_advantage: bool = False
    # GDPO reward norm (group reward-Decoupled normalization, arXiv 2601.05242):
    # normalize EACH reward component (prediction / soft_L2 / PDMS) within its
    # GRPO group first, weight by the coef_* knobs, then batch-whiten the sum, so
    # coef_* becomes the real lever between soft_L2 and PDMS inside the action
    # head. Independent of rl_split_advantage; default false.
    rl_reward_norm: bool = False

    # ---- rollout generation micro-batch (OOM guard) ----
    # The synchronous Transfusion rollout has NO internal micro-batching —
    # ``model.generate_trajectory`` runs on the whole per-GPU shard at once.
    # With train_batch_size=128 * rollout.n=8 / 8 GPUs = 128 samples/GPU that
    # OOMs on the 2B multimodal model. Set this to chunk the generation INTO
    # smaller forward passes (e.g. 32) and concatenate the results, trading a
    # small latency overhead for peak-memory relief. None = no chunking (old
    # behaviour; safe when train_batch_size is small).
    rl_gen_micro_batch_size: Optional[int] = None

    def __post_init__(self):
        # The Transfusion rollout is synchronous by design, but the upstream
        # __post_init__ raises on mode='sync'. Run all the OTHER RolloutConfig
        # validations with mode temporarily presented as 'async', then restore
        # the real value so get_rollout_class(name, 'sync') resolves the sync
        # TransfusionRollout. (BaseConfig is frozen -> object.__setattr__.)
        if self.mode == "sync":
            object.__setattr__(self, "mode", "async")
            try:
                super().__post_init__()
            finally:
                object.__setattr__(self, "mode", "sync")
        else:
            super().__post_init__()
