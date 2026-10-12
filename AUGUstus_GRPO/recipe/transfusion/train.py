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

"""Transfusion RL entry point — verl's ``main_ppo`` + best-checkpoint-by-metric.

This is a *thin* wrapper around ``verl.trainer.main_ppo`` that injects the
best-checkpoint-aware trainer (:mod:`recipe.transfusion.best_ckpt`) through verl's
official extension hook ``run_ppo(config, task_runner_class=...)`` — **no verl
core file is edited**. Everything else (Hydra config composition, Ray init, worker
setup, the training loop) is verl's, unchanged.

It is a drop-in replacement for ``python -m verl.trainer.main_ppo``; the recipe
run scripts call it instead so that ``trainer.best_metrics`` takes effect. With
``best_metrics`` empty it behaves exactly like the stock entry.

    python -m recipe.transfusion.train \
      --config-dir recipe/transfusion/config \
      --config-name transfusion_navsim_qwen3p5_flow_grpo
"""

from __future__ import annotations

import os

import hydra
import ray
from omegaconf import OmegaConf

import verl.trainer  # for the verl config dir
from recipe.transfusion.ckpt_layout import apply_checkpoint_layout
from verl.trainer.main_ppo import TaskRunner, run_ppo

# verl's config dir (holds ``ppo_trainer.yaml`` referenced by the recipe defaults).
_VERL_CONFIG_DIR = os.path.join(os.path.dirname(verl.trainer.__file__), "config")


class TransfusionTaskRunner(TaskRunner):
    """``TaskRunner`` that swaps in the best-checkpoint-aware ``RayPPOTrainer``.

    ``TaskRunner.run`` builds ``RayPPOTrainer`` from its own module globals
    (``verl.trainer.main_ppo.RayPPOTrainer``). We re-bind that name to our
    subclass *inside this Ray actor process* right before delegating, so the
    swap is local and needs no core edit. Idempotent and safe when
    ``best_metrics`` is empty (the subclass is then a no-op).
    """

    def run(self, config):
        import verl.trainer.main_ppo as vmain
        from recipe.transfusion.best_ckpt import build_trainer_class

        vmain.RayPPOTrainer = build_trainer_class()
        return super().run(config)

    def init_resource_pool_mgr(self, config):
        # One fused Transfusion actor-rollout-ref worker per GPU: each worker must
        # claim a WHOLE GPU (max_colocate_count=1) so Ray is forced to spread the
        # N*G workers across ALL nodes instead of packing them all onto the first
        # node. verl's default max_colocate_count=3 makes every worker request only
        # 1/3 GPU, so on a 2x8 (16 GPU) cluster all 16 actor workers can be packed
        # onto ONE node's 8 GPUs -> two ranks share the same physical GPU -> NCCL
        # "Duplicate GPU detected: rank N and rank N+8 on CUDA device ...".
        mgr = super().init_resource_pool_mgr(config)
        # TRANSFUSION_MAX_COLOCATE: how many worker-processes each Ray "unit" (and
        # hence the number/size of placement-group bundles) colocates on one GPU.
        #   =1  -> every worker claims a WHOLE GPU (16 separate 1-GPU bundles) -- the
        #         historical default forcing Ray to spread workers across all nodes.
        #   >1  -> verl's default colocation groups several workers per GPU into
        #         FEWER, SMALLER bundles (a much lighter placement group to schedule).
        # Tuning this changes how the multi-node placement group is formed, which is
        # the stage where 2-node init currently dies on the worker-role node.
        _mcc = os.environ.get("TRANSFUSION_MAX_COLOCATE", "1")
        try:
            _mcc = int(_mcc)
        except Exception:
            _mcc = 1
        try:
            mgr.max_colocate_count = _mcc
        except Exception as e:  # never break the trainer if this is unsettable
            print(f"[recipe.transfusion] could not set max_colocate_count={_mcc}: {e}")
        return mgr

    def add_actor_rollout_worker(self, config):
        # Swap the actor-rollout worker for the subclass that exposes a SYNC
        # ``generate_sequences`` proxy on the WorkerGroup, so the trainer can drive
        # the Transfusion diffusion rollout directly (this verl version's async
        # LLM-server path cannot host a custom rollout). ``add_actor_rollout_worker``
        # imports ``ActorRolloutRefWorker`` locally, so patching the module symbol
        # for the duration of the parent call is enough.
        import verl.workers.engine_workers as _ew

        from recipe.transfusion.sync_driver import TransfusionActorRolloutRefWorker

        _orig = _ew.ActorRolloutRefWorker
        _ew.ActorRolloutRefWorker = TransfusionActorRolloutRefWorker
        try:
            return super().add_actor_rollout_worker(config)
        finally:
            _ew.ActorRolloutRefWorker = _orig


@hydra.main(config_path=_VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    # Mirror verl.trainer.main_ppo.main's pre-processing (guarded so a verl
    # refactor of these helpers cannot break the entry).
    try:
        from verl.utils.device import auto_set_device

        auto_set_device(config)
    except Exception:
        pass
    try:
        from verl.experimental.reward_loop import migrate_legacy_reward_impl

        config = migrate_legacy_reward_impl(config)
    except Exception:
        pass

    # Transfusion-style run layout (see recipe.transfusion.ckpt_layout): append a
    # <YYYYMMDD_HHMMSS> timestamp subdir to trainer.default_local_dir for a FRESH run
    # (or re-point it at the resumed run's dir and set resume_mode/resume_from_path
    # when trainer.resume.enabled), and map trainer.save_optimizer onto the actor's
    # checkpoint.save_contents. Must run BEFORE any ${trainer.default_local_dir}
    # interpolation is resolved (the env_vars block below, rollout_data_dir,
    # SWANLAB_LOG_DIR, TRANSFUSION_OUTPUT_DIR, best-ckpt snapshots all follow it).
    apply_checkpoint_layout(config)

    # verl's run_ppo forwards ray_kwargs.ray_init.runtime_env.env_vars to ray.init()
    # via OmegaConf.to_container(..., resolve=False), so any interpolation left in
    # that block (${actor_rollout_ref.model.path}, ${trainer.default_local_dir},
    # ${oc.env:...}) would reach the Ray workers as a LITERAL "${...}" string and
    # break the model / stats / logging paths. Resolve it here — while the node is
    # still attached to the full config root — so workers get concrete string values.
    env_vars_node = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars")
    if env_vars_node is not None:
        resolved_env_vars = {
            k: ("" if v is None else str(v))
            for k, v in OmegaConf.to_container(env_vars_node, resolve=True).items()
        }
        config.ray_kwargs.ray_init.runtime_env.env_vars = resolved_env_vars

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TransfusionTaskRunner))


if __name__ == "__main__":
    main()
