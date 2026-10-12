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

"""RL algorithm kernels for the Transfusion trajectory diffusion policy.

* :mod:`.fm_sde_logprob` — the FM-SDE rollout kernel (exact per-step Gaussian
  log-probabilities under ``x_t = (1 - t) * x_0 + t * noise``, ``t = 0`` clean /
  ``t = 1`` noise; lambda_t g(t) schedule) used by the data-collection rollout.
* :mod:`.jit_grpo` — closed-form x0-prediction GRPO: exact per-step log-ratio +
  Gaussian KL for the action head, and the per-token probability-ratio GRPO loss
  for the CoT head (:func:`jit_grpo_policy_loss` / :func:`cot_token_grpo_loss` is
  re-exported from :mod:`.cot_grpo`).
* :mod:`.cot_grpo` — the CoT head's per-token GRPO loss (k1 / k3 KL estimators).
* :mod:`.diffusion_adapter` — binds the kernels to a concrete
  ``TransfusionQwen3p5`` model (prefix building + differentiable per-step
  denoiser).

The modules depend only on ``torch`` (no verl / Transfusion imports) so the
maths is unit-testable in isolation.
"""

from __future__ import annotations

__all__ = [
    "fm_sde_logprob",
    "jit_grpo",
    "cot_grpo",
    "diffusion_adapter",
]
