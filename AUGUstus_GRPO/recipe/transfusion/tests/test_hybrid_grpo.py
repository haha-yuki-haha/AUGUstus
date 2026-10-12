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

"""End-to-end tests for the hybrid GRPO actor loss (closed-form action + CoT tokens).

Exercises :func:`recipe.transfusion.loss.hybrid_grpo_loss_from_rollout` and the
``rl_score_fn`` forward hook against a tiny mock model (no GPU / verl /
Transfusion backbone needed), checking:

* the FSDP-safe ``model(rl_score_fn=...)`` closure returns the differentiable
  action x0 predictions + CoT log-probs and the loss math runs outside it;
* on-policy (behaviour == current) => both the action ratio and the CoT ratio are
  exactly 1 and the clipped surrogate is ``-advantage`` (i.e. real GRPO, not the
  old zero-logp stub);
* gradients flow into BOTH heads (the denoiser params AND the CoT head);
* the reported ``actor/entropy`` proxy (CoT token entropy) is non-zero.

Run::

    conda activate augustus_grpo
    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_hybrid_grpo.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion.algorithms.cot_grpo import cot_token_grpo_loss  # noqa: E402
from recipe.transfusion.algorithms.diffusion_adapter import (  # noqa: E402
    TransfusionDiffusionAdapter,
)
from recipe.transfusion.tests._mocks import _MockModel, _mock_batch  # noqa: E402

torch.manual_seed(0)

_VOCAB = 64


class _MockHybridModel(_MockModel):
    """``_MockModel`` + the CoT log-prob method and the FSDP-safe forward hook."""

    use_action_first = True
    action_first_generate_cot = True

    def __init__(self, traj_token=8, action_dim=3, hidden=16):
        super().__init__(traj_token, action_dim, hidden)
        # A learnable CoT head so the token-probability loss has a gradient path;
        # logits depend on the (sampled) action so the CoT gradient also reaches
        # the trajectory representation, mirroring action_first's action->CoT conditioning.
        self.cot_head = torch.nn.Parameter(torch.randn(action_dim, _VOCAB) * 0.05)

    def cot_logprob(self, input_ids, actions_normed, cot_ids, ctx_embeds=None, **kw):
        B, C = cot_ids.shape
        feat = actions_normed.float().mean(dim=1)  # [B, D]
        logits = (feat @ self.cot_head).unsqueeze(1).expand(B, C, _VOCAB)  # [B, C, V]
        logp_all = torch.log_softmax(logits, dim=-1)
        ids = cot_ids.long().clamp(min=0, max=_VOCAB - 1).unsqueeze(-1)
        token_logp = logp_all.gather(-1, ids).squeeze(-1)  # [B, C]
        token_entropy = -(logp_all.exp() * logp_all).sum(-1)  # [B, C]
        return token_logp, token_entropy

    def forward(self, input_ids=None, rl_score_fn=None, **kwargs):
        # Mirrors TransfusionQwen3p5.forward's FSDP-safe RL scoring hook.
        if rl_score_fn is not None:
            return rl_score_fn(self)
        raise NotImplementedError("mock forward only supports rl_score_fn")


def _rollout(B=4, steps=5, noise=0.7, schedule="lambda_t"):
    model = _MockHybridModel()
    adapter = TransfusionDiffusionAdapter(model)
    batch = _mock_batch(B=B)
    ctx = adapter.prepare_context(batch, grad=False)
    record = adapter.sample_sde_with_logprob(
        ctx, num_steps=steps, noise_level=noise, noise_schedule=schedule,
    )
    return model, batch, record


def test_hybrid_onpolicy_ratios_are_one_and_grad_flows_both_heads():
    from recipe.transfusion.loss import hybrid_grpo_loss_from_rollout

    B = 4
    model, batch, record = _rollout(B=B)
    x0_normed = record["next_sample"][-1].detach()  # clean sampled trajectory
    cot_ids = torch.randint(0, _VOCAB, (B, 6))
    cot_mask = torch.ones(B, 6)
    cot_mask[:, -2:] = 0  # some right-padding

    # Behaviour CoT log-prob under the SAME (unchanged) weights -> on-policy.
    with torch.no_grad():
        cot_old, _ = model.cot_logprob(batch["input_ids"], x0_normed, cot_ids)

    advantages = torch.randn(B)
    loss, m = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed,
        cfg={"clip_range": 0.2, "timestep_fraction": 1.0,
             "action_coef": 1.0, "cot_coef": 1.0, "entropy_coef": 0.0},
    )
    # On-policy: both importance ratios are exactly 1.
    assert abs(m["jit_grpo/ratio_mean"] - 1.0) < 1e-4, m["jit_grpo/ratio_mean"]
    assert abs(m["cot_grpo/ratio_mean"] - 1.0) < 1e-4, m["cot_grpo/ratio_mean"]
    # Entropy proxy (goes to actor/entropy) is a real, non-zero quantity now.
    assert m["cot_grpo/entropy"] > 0.0
    assert torch.isfinite(loss)

    loss.backward()
    # Gradient reaches BOTH the denoiser (action) params and the CoT head.
    assert model.w.grad is not None and torch.isfinite(model.w.grad).all()
    assert float(model.w.grad.abs().sum()) > 0
    assert model.cot_head.grad is not None and torch.isfinite(model.cot_head.grad).all()
    assert float(model.cot_head.grad.abs().sum()) > 0


def test_hybrid_closure_runs_inside_forward_hook():
    """The differentiable quantities must come back THROUGH model(rl_score_fn=...)."""
    from recipe.transfusion.loss import hybrid_grpo_loss_from_rollout

    B = 3
    model, batch, record = _rollout(B=B, steps=4)
    calls = {"n": 0}
    orig_forward = model.forward

    def _counting_forward(*a, **k):
        if k.get("rl_score_fn") is not None:
            calls["n"] += 1
        return orig_forward(*a, **k)

    model.forward = _counting_forward
    cot_ids = torch.randint(0, _VOCAB, (B, 5))
    with torch.no_grad():
        cot_old, _ = model.cot_logprob(batch["input_ids"], record["next_sample"][-1], cot_ids)
    loss, _ = hybrid_grpo_loss_from_rollout(
        model, batch, record, torch.randn(B),
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=torch.ones(B, 5),
    )
    assert calls["n"] == 1, "action + CoT scoring must run in ONE forward hook call"
    loss.backward()
    assert model.cot_head.grad is not None


def test_cot_ratio_moves_when_weights_change():
    """After a CoT-head perturbation the token ratio diverges from 1 (real ratio)."""
    B = 4
    model, batch, record = _rollout(B=B)
    x0 = record["next_sample"][-1].detach()
    cot_ids = torch.randint(0, _VOCAB, (B, 5))
    mask = torch.ones(B, 5)
    with torch.no_grad():
        old, _ = model.cot_logprob(batch["input_ids"], x0, cot_ids)
        model.cot_head.add_(torch.randn_like(model.cot_head))  # non-uniform update
        new, ent = model.cot_logprob(batch["input_ids"], x0, cot_ids)
    _loss, m = cot_token_grpo_loss(new, old, torch.randn(B), mask, entropy=ent, clip_range=0.2)
    assert abs(m["cot_grpo/ratio_mean"] - 1.0) > 1e-3


def test_cot_kl_estimator_k1_can_be_negative():
    """k1 estimator: mean(old - new) can go NEGATIVE when new_logp > old_logp."""
    B, R = 2, 4
    mask = torch.ones(B, R)
    # new > old at every token => old - new < 0 => k1 < 0
    new_logp = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    old_logp = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]])
    _loss, m = cot_token_grpo_loss(
        new_logp, old_logp, torch.randn(B), mask, clip_range=0.2, kl_estimator="k1",
    )
    assert m["cot_grpo/approx_kl"] < 0.0, m["cot_grpo/approx_kl"]


def test_cot_kl_estimator_k3_non_negative():
    """k3 estimator: exp(gap) - 1 - gap is pointwise >= 0, so the mean is too."""
    B, R = 3, 6
    mask = torch.ones(B, R)
    torch.manual_seed(42)
    new_logp = torch.randn(B, R)
    old_logp = torch.randn(B, R)
    _loss, m = cot_token_grpo_loss(
        new_logp, old_logp, torch.randn(B), mask, clip_range=0.2, kl_estimator="k3",
    )
    assert m["cot_grpo/approx_kl"] >= 0.0, m["cot_grpo/approx_kl"]
    # k3 with zero gap (on-policy) should be ~0
    _loss2, m2 = cot_token_grpo_loss(
        old_logp, old_logp, torch.randn(B), mask, clip_range=0.2, kl_estimator="k3",
    )
    assert abs(m2["cot_grpo/approx_kl"]) < 1e-5, m2["cot_grpo/approx_kl"]


def test_cot_kl_estimator_invalid_raises():
    """An unknown kl_estimator value must raise ValueError, not silently fall back."""
    B, R = 2, 3
    mask = torch.ones(B, R)
    new_logp = torch.zeros(B, R)
    old_logp = torch.zeros(B, R)
    raised = False
    try:
        cot_token_grpo_loss(
            new_logp, old_logp, torch.randn(B), mask, clip_range=0.2, kl_estimator="k9",
        )
    except ValueError:
        raised = True
    assert raised, "expected ValueError for kl_estimator='k9'"


def test_hybrid_cot_kl_estimator_invalid_cfg_raises():
    """hybrid_grpo_loss_from_rollout must reject an invalid cot_kl_estimator cfg.

    The validation runs BEFORE any model forward / rollout access, so dummy
    objects suffice (no _rollout needed — the mock model's SDE sampler is
    broken in the current env, see pre-existing test failures)."""
    from recipe.transfusion.loss import hybrid_grpo_loss_from_rollout

    raised = False
    try:
        hybrid_grpo_loss_from_rollout(
            model=None, batch={}, rollout={}, advantages=torch.zeros(1),
            cot_ids=torch.zeros(1, 1, dtype=torch.long),
            cot_old_logp=torch.zeros(1, 1), cot_mask=torch.ones(1, 1),
            cfg={"clip_range": 0.2, "cot_kl_estimator": "bogus"},
        )
    except ValueError:
        raised = True
    assert raised, "expected ValueError for cot_kl_estimator='bogus' in hybrid cfg"


def test_hybrid_ref_kl_zero_for_identical_ref_positive_for_perturbed():
    """ref_model wiring: KL == 0 when ref == policy; > 0 after a perturbation.

    The action head uses the exact closed-form Gaussian KL from the two x0
    predictions; the CoT head uses the k3 token-probability estimator — both
    driven by cfg['kl_coef'] via the frozen ref_model."""
    import copy

    from recipe.transfusion.loss import hybrid_grpo_loss_from_rollout

    B = 4
    model, batch, record = _rollout(B=B)
    x0_normed = record["next_sample"][-1].detach()
    cot_ids = torch.randint(0, _VOCAB, (B, 6))
    cot_mask = torch.ones(B, 6)
    with torch.no_grad():
        cot_old, _ = model.cot_logprob(batch["input_ids"], x0_normed, cot_ids)
    advantages = torch.randn(B)
    cfg = {"clip_range": 0.2, "kl_coef": 0.01}

    # (1) ref == policy -> both KL terms are (numerically) zero.
    ref_same = copy.deepcopy(model)
    _l, m_same = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed, cfg=cfg, ref_model=ref_same,
    )
    assert abs(m_same["hybrid/action_kl"]) < 1e-8, m_same["hybrid/action_kl"]
    assert abs(m_same["hybrid/cot_kl"]) < 1e-6, m_same["hybrid/cot_kl"]

    # (2) perturbed ref -> strictly positive KL on both heads, and grad flows.
    ref_pert = copy.deepcopy(model)
    with torch.no_grad():
        ref_pert.w.add_(0.5 * torch.randn_like(ref_pert.w))
        ref_pert.cot_head.add_(torch.randn_like(ref_pert.cot_head))
    loss, m_pert = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed, cfg=cfg, ref_model=ref_pert,
    )
    assert m_pert["hybrid/action_kl"] > 0.0
    assert m_pert["hybrid/cot_kl"] > 0.0
    loss.backward()
    assert model.w.grad is not None and float(model.w.grad.abs().sum()) > 0

    # (3) kl_coef=0 -> ref model is ignored entirely (identical loss w/ and w/o).
    l_no_ref, m_no = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed, cfg={"clip_range": 0.2, "kl_coef": 0.0}, ref_model=ref_pert,
    )
    assert m_no["hybrid/action_kl"] == 0.0 and m_no["hybrid/cot_kl"] == 0.0


def test_hybrid_precomputed_ref_matches_internal_recompute():
    """Batched-precompute ref path == in-loss recompute (the speed change is exact).

    The worker (``sync_driver._precompute_ref_quantities``) computes the frozen
    reference's per-step x0 predictions + CoT log-probs ONCE per batch (verl-style
    ref-logprob phase) and passes them into ``hybrid_grpo_loss_from_rollout`` as
    ``x0_ref`` / ``cot_ref_logp`` instead of recomputing a full reference forward
    inside every micro-batch. This must be numerically identical to the recompute."""
    import copy

    from recipe.transfusion.algorithms.diffusion_adapter import TransfusionDiffusionAdapter
    from recipe.transfusion.loss import hybrid_grpo_loss_from_rollout

    B = 4
    model, batch, record = _rollout(B=B)
    x0_normed = record["next_sample"][-1].detach()
    cot_ids = torch.randint(0, _VOCAB, (B, 6))
    cot_mask = torch.ones(B, 6)
    cot_mask[:, -2:] = 0
    with torch.no_grad():
        cot_old, _ = model.cot_logprob(batch["input_ids"], x0_normed, cot_ids)
    advantages = torch.randn(B)
    cfg = {"clip_range": 0.2, "kl_coef": 0.05, "timestep_fraction": 1.0}

    # Perturbed frozen reference so BOTH KL terms are strictly positive.
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.w.add_(0.3 * torch.randn_like(ref.w))
        ref.cot_head.add_(0.5 * torch.randn_like(ref.cot_head))

    # Path A: the loss recomputes the reference internally (ref_model given).
    loss_a, m_a = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed, cfg=cfg, ref_model=ref,
    )

    # Path B: precompute x0_ref / cot_ref_logp (as the worker does) and pass them.
    ref_adapter = TransfusionDiffusionAdapter(ref)
    with torch.no_grad():
        ref_ctx = ref_adapter.prepare_context(batch, grad=False)
        S = record["x_in"].shape[0]
        x0_ref = ref_adapter.recompute_x0_preds(
            ref_ctx, record["x_in"], record["sigma"], step_indices=list(range(S)),
        ).detach()
        cot_ref, _ = ref.cot_logprob(
            batch["input_ids"], x0_normed, cot_ids, ctx_embeds=ref_ctx.get("ctx_embeds"),
        )
        cot_ref = cot_ref.detach()
    loss_b, m_b = hybrid_grpo_loss_from_rollout(
        model, batch, record, advantages,
        cot_ids=cot_ids, cot_old_logp=cot_old, cot_mask=cot_mask,
        x0_normed=x0_normed, cfg=cfg, ref_model=None,
        x0_ref=x0_ref, cot_ref_logp=cot_ref,
    )

    assert torch.allclose(loss_a, loss_b, atol=1e-6), (float(loss_a), float(loss_b))
    assert abs(m_a["hybrid/action_kl"] - m_b["hybrid/action_kl"]) < 1e-6
    assert abs(m_a["hybrid/cot_kl"] - m_b["hybrid/cot_kl"]) < 1e-6
    assert m_b["hybrid/action_kl"] > 0.0 and m_b["hybrid/cot_kl"] > 0.0


def test_sample_grpo_uses_action_first_flags():
    """Collect CoT and its probabilities using the action-first flags."""
    from itertools import product
    from types import SimpleNamespace

    from recipe.transfusion.transfusion_policy import TransfusionPolicy

    batch = {"input_ids": torch.zeros(2, 3, dtype=torch.long)}
    sampled = torch.full((2, 8, 3), 2.0)
    ids = torch.tensor([[1, 2], [3, 4]])
    logps = torch.full((2, 2), -0.5)
    entropies = torch.full((2, 2), 0.7)
    ctx = {
        "ctx_embeds": torch.zeros(2, 3, 4),
        "prefix_pos": torch.zeros(3, 2, 3, dtype=torch.long),
        "prefix_adarms_cond": None,
        "B": 2,
    }

    for generate_cot, use_action_first, action_first_generate_cot in product(
        (False, True), repeat=3,
    ):
        calls = []

        def generate_after_action(*args, **kwargs):
            calls.append(kwargs)
            assert torch.equal(args[1], sampled.clamp(-1.0, 1.0))
            assert kwargs["record_logp"] is True
            assert kwargs["max_cot_tokens"] == 3
            return ids, logps, entropies

        policy = TransfusionPolicy.__new__(TransfusionPolicy)
        policy.model = SimpleNamespace(
            use_action_first=use_action_first,
            action_first_generate_cot=action_first_generate_cot,
            _generate_cot_after_action=generate_after_action,
        )
        policy.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "sampled CoT"),
        )
        policy.collate = lambda inputs: batch
        policy._adapter = SimpleNamespace(
            prepare_context=lambda batch, **kwargs: ctx,
            sample_sde_with_logprob=lambda ctx, **kwargs: {"x0_normed": sampled},
            denormalize=lambda values: values,
        )

        result = policy.sample_grpo(
            [{}, {}], generate_cot=generate_cot, cot_max_tokens=3, device="cpu",
        )
        assert torch.equal(result["x0_normed"], sampled.clamp(-1.0, 1.0))
        if generate_cot and use_action_first and action_first_generate_cot:
            assert len(calls) == 1
            assert result["cot_ids"] == ids.tolist()
            assert result["cot_texts"] == ["sampled CoT", "sampled CoT"]
            assert torch.equal(result["cot_logps"], logps)
            assert torch.equal(result["cot_entropies"], entropies)
        else:
            assert not calls
            assert result["cot_ids"] == [[], []]
            assert result["cot_texts"] == ["", ""]
            assert result["cot_logps"] is None
            assert result["cot_entropies"] is None


def test_config_overrides_keep_action_first_flags():
    """Preserve the canonical CoT switches when merging configuration overrides."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from recipe.transfusion.transfusion_policy import load_transfusion_config

    with TemporaryDirectory(prefix="augustus-config-test-") as directory:
        config = Path(directory) / "config.yaml"
        config.write_text(
            "cot:\n  use_action_first: true\n  action_first_generate_cot: false\n",
        )
        loaded = load_transfusion_config(
            str(config), {"cot": {"action_first_generate_cot": True}},
        )
        assert loaded["cot"]["use_action_first"] is True
        assert loaded["cot"]["action_first_generate_cot"] is True


if __name__ == "__main__":
    test_hybrid_onpolicy_ratios_are_one_and_grad_flows_both_heads()
    test_hybrid_closure_runs_inside_forward_hook()
    test_cot_ratio_moves_when_weights_change()
    test_cot_kl_estimator_k1_can_be_negative()
    test_cot_kl_estimator_k3_non_negative()
    test_cot_kl_estimator_invalid_raises()
    test_hybrid_cot_kl_estimator_invalid_cfg_raises()
    test_hybrid_ref_kl_zero_for_identical_ref_positive_for_perturbed()
    test_hybrid_precomputed_ref_matches_internal_recompute()
    test_sample_grpo_uses_action_first_flags()
    test_config_overrides_keep_action_first_flags()
    print("All hybrid GRPO tests passed.")
