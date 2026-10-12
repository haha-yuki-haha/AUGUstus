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

"""Unit tests for the configurable denoising-step counts in TransfusionRollout.

Covers the three places the step count must be configurable:

* TRAINING rollouts          -> ``rl_num_steps``     (all three algorithms),
* IN-TRAINING validation     -> ``rl_val_num_steps`` (verl's ``_validate`` tags
  the gen batch with ``meta_info['validate']=True``; falls back to
  ``rl_num_steps`` when unset).

(Standalone evaluation outside RL training is configured in the Transfusion
repo: ``test_qwen3p5.py --num_inference_steps`` / ``NUM_INFERENCE_STEPS`` in
``scripts/test_checkpoint.sh`` — not exercised here.)

Run::

    conda activate augustus_grpo
    cd AUGUstus_GRPO
    python -m pytest recipe/transfusion/tests/test_rollout_num_steps.py -q
    # or:  python recipe/transfusion/tests/test_rollout_num_steps.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from verl import DataProto  # noqa: E402

from recipe.transfusion.rollout import TransfusionRollout  # noqa: E402


class _FakePolicy:
    """Minimal policy stub recording the ``num_steps`` it was called with.

    The default TRAINING path samples via ``sample_grpo`` (stochastic SDE +
    per-step record for the closed-form action ratio); VALIDATION still uses the
    deterministic ``rollout``. Both record into ``seen_num_steps`` so the existing
    step-count assertions hold regardless of which path a batch takes.
    """

    processor = None

    def __init__(self):
        self.seen_num_steps = []
        self.seen_cot_sampler = []

    def rollout(self, transfusion_inputs, *, generate_cot=True, return_normed=False,
                num_steps=None, **kwargs):
        self.seen_num_steps.append(num_steps)
        self.seen_cot_sampler.append(dict(kwargs))
        b = len(transfusion_inputs)
        return {
            "trajectories": np.zeros((b, 8, 3), dtype=np.float32),
            "cot_texts": [""] * b,
            "cot_ids": [[1, 2, 3]] * b,
        }

    def sample_grpo(self, transfusion_inputs, *, num_steps=None, noise_level=0.7,
                    noise_schedule="flow_grpo", generate_cot=True, **kwargs):
        import torch

        self.seen_num_steps.append(num_steps)
        b = len(transfusion_inputs)
        S, T, D = int(num_steps or 4), 8, 3
        record = {
            "x_in": torch.zeros(S, b, T, D), "next_sample": torch.zeros(S, b, T, D),
            "sigma": torch.zeros(S, b), "sigma_prev": torch.zeros(S, b),
            "log_probs": torch.zeros(S, b), "x0_old": torch.zeros(S, b, T, D),
            "noise_level": noise_level, "noise_schedule": noise_schedule, "sigma_max": 1.0,
        }
        return {
            "trajectories": np.zeros((b, 8, 3), dtype=np.float32),
            "x0_normed": torch.zeros(b, T, D),
            "record": record,
            "cot_texts": [""] * b,
            "cot_ids": [[1, 2, 3]] * b,
        }


def _gen_batch(n=2, validate=False) -> DataProto:
    return DataProto(
        non_tensor_batch={
            "transfusion_inputs": np.array([{"dummy": i} for i in range(n)], dtype=object),
        },
        meta_info={"validate": True} if validate else {},
    )


def _make_rollout(policy, **kwargs):
    return TransfusionRollout(config=None, model_config=None, device_mesh=None,
                              policy=policy, **kwargs)


def test_train_batch_uses_rl_num_steps():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, rl_num_steps=10, rl_val_num_steps=4)
    rollout.generate_sequences(_gen_batch(validate=False))
    assert policy.seen_num_steps == [10]


def test_val_batch_uses_rl_val_num_steps_when_set():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, rl_num_steps=10, rl_val_num_steps=4)
    rollout.generate_sequences(_gen_batch(validate=True))
    assert policy.seen_num_steps == [4]


def test_val_batch_falls_back_to_rl_num_steps_when_unset():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, rl_num_steps=10)  # rl_val_num_steps default None
    rollout.generate_sequences(_gen_batch(validate=True))
    assert policy.seen_num_steps == [10]


def test_train_and_val_batches_intermixed():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, rl_num_steps=10, rl_val_num_steps=6)
    rollout.generate_sequences(_gen_batch(validate=False))
    rollout.generate_sequences(_gen_batch(validate=True))
    rollout.generate_sequences(_gen_batch(validate=False))
    assert policy.seen_num_steps == [10, 6, 10]


def test_rl_val_num_steps_from_env(monkeypatch=None):
    """Env-var fallback (TRANSFUSION_RL_VAL_NUM_STEPS) works when config is absent."""
    os.environ["TRANSFUSION_RL_VAL_NUM_STEPS"] = "7"
    try:
        policy = _FakePolicy()
        rollout = _make_rollout(policy, rl_num_steps=10)
        assert rollout._rl_val_num_steps == 7
        rollout.generate_sequences(_gen_batch(validate=True))
        assert policy.seen_num_steps == [7]
    finally:
        os.environ.pop("TRANSFUSION_RL_VAL_NUM_STEPS", None)


def test_empty_or_null_env_val_num_steps_means_fallback():
    """A null-ish env value must not crash the int() parse -> fallback to train steps."""
    os.environ["TRANSFUSION_RL_VAL_NUM_STEPS"] = "null"
    try:
        policy = _FakePolicy()
        rollout = _make_rollout(policy, rl_num_steps=10)
        assert rollout._rl_val_num_steps is None
        rollout.generate_sequences(_gen_batch(validate=True))
        assert policy.seen_num_steps == [10]
    finally:
        os.environ.pop("TRANSFUSION_RL_VAL_NUM_STEPS", None)


# ---------------------------------------------------------------------------
# Noise level + g(t) schedule (single shared sampler for train and val).
# ---------------------------------------------------------------------------
def test_noise_schedule_alias_normalized_at_construction():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, rl_noise_schedule="Flow-GRPO")
    assert rollout._rl_noise_schedule == "flow_grpo"


def test_invalid_noise_schedule_raises_at_construction():
    policy = _FakePolicy()
    try:
        _make_rollout(policy, rl_noise_schedule="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for an unknown schedule")


# ---------------------------------------------------------------------------
# CoT sampler: validation uses rl_val_cot_* when set, else the training knobs.
# ---------------------------------------------------------------------------
def test_train_val_cot_sampler_resolution():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, temperature=1.0, top_p=0.95, top_k=100,
                            rl_val_cot_temperature=1.2, rl_val_cot_top_p=0.9, rl_val_cot_top_k=50)
    train, val = _gen_batch(validate=False), _gen_batch(validate=True)
    assert rollout._cot_sampler_for(train) == {"cot_temperature": 1.0, "cot_top_p": 0.95, "cot_top_k": 100}
    assert rollout._cot_sampler_for(val) == {"cot_temperature": 1.2, "cot_top_p": 0.9, "cot_top_k": 50}


def test_val_cot_sampler_falls_back_to_train_knobs():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, temperature=1.0, top_p=0.95, top_k=100)
    sampler = rollout._cot_sampler_for(_gen_batch(validate=True))
    assert sampler == {"cot_temperature": 1.0, "cot_top_p": 0.95, "cot_top_k": 100}


def test_val_batch_forwards_val_cot_sampler_to_policy():
    policy = _FakePolicy()
    rollout = _make_rollout(policy, temperature=1.0, top_p=0.95, top_k=100,
                            rl_val_cot_temperature=0.2, rl_val_cot_top_p=0.8, rl_val_cot_top_k=10)
    rollout.generate_sequences(_gen_batch(validate=True))
    assert policy.seen_cot_sampler[-1] == {"cot_temperature": 0.2, "cot_top_p": 0.8, "cot_top_k": 10}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} rollout num-steps tests passed.")
