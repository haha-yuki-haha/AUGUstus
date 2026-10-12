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

"""verl <-> Transfusion reinforcement-learning integration (recipe).

This recipe adapts the upstream verl PPO/GRPO machinery to fine-tune the
``Transfusion`` Qwen3.5 diffusion driving planner. Only the *data*, *model* and
*reward* surfaces are swapped — every reinforcement-learning hyper-parameter in
``verl/trainer/config`` is left untouched (see ``docs/ADAPTATION.md``).

Public surface:
    * :mod:`recipe.transfusion.transfusion_reward` — framework-free reward math
      (L2 / PDMS / CoT bbox-IoU + future-position correctness).
    * :mod:`recipe.transfusion.reward_function`    — ``compute_score`` hook for
      ``reward.custom_reward_function``.
    * :mod:`recipe.transfusion.reward_manager`     — trajectory-aware
      ``TransfusionRewardManager`` for ``reward.reward_manager``.
    * :mod:`recipe.transfusion.dataset`            — ``TransfusionRLDataset`` for
      ``data.custom_cls``.
    * :mod:`recipe.transfusion.transfusion_policy` — builds ``TransfusionQwen3p5``.
    * :mod:`recipe.transfusion.rollout`            — diffusion trajectory rollout.
"""
