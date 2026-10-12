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

"""Custom training engine that builds the Transfusion model inside verl.

verl resolves the actor/ref training engine through ``EngineRegistry`` keyed by
``actor_rollout_ref.model.model_type``. The stock ``language_model`` engine
builds an HF ``AutoModelForCausalLM`` in ``FSDPEngine._build_module``. This
module registers a new ``model_type = "transfusion_diffusion"`` whose only
difference is that ``_build_module`` constructs ``TransfusionQwen3p5`` (via the
shared :func:`recipe.transfusion.transfusion_policy.build_transfusion_model`),
so the FSDP wrapping, optimizer, gradient-clipping and checkpointing provided by
the base engine are reused unchanged.

Select it from the recipe config with::

    actor_rollout_ref:
      model:
        model_type: transfusion_diffusion
        path: /path/to/Qwen3.5-2B          # backbone (tokenizer / hf_config)
        trust_remote_code: true

and point the build at the Transfusion YAML via ``$TRANSFUSION_CONFIG``.

IMPORTANT — what is and isn't done here (see ``docs/ADAPTATION.md``):
  * ``_build_module`` (model construction) is fully overridden — this is the
    "swap the model" step the integration is centred on.
  * The forward / log-prob / loss path inherited from ``FSDPEngineWithLMHead``
    assumes HF ``logits``. Transfusion returns a ``TransfusionOutput`` (diffusion
    + LM loss), so the actor loss must be supplied through the worker's
    ``set_loss_fn`` injection point (continuous-action policy gradient). That
    loss wiring is the remaining cluster-side task and is intentionally left as a
    clearly-marked hook rather than a silent stub.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_transfusion_engine() -> bool:
    """Register the ``transfusion_diffusion`` FSDP engine. Idempotent.

    Returns ``True`` on success. Safe to import on machines without the full
    verl engine stack — failures are logged and swallowed so config resolution
    never hard-crashes just from importing the recipe.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import torch
        import warnings

        from verl.workers.engine.base import EngineRegistry
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
        from verl.utils.fsdp_utils import get_init_weight_context_manager
        from verl.utils.torch_dtypes import PrecisionType
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Transfusion engine not registered (verl engine import failed: %s)", e)
        return False

    @EngineRegistry.register(model_type="transfusion_diffusion", backend=["fsdp", "fsdp2"], device=["cuda"])
    class TransfusionFSDPEngine(FSDPEngineWithLMHead):
        """FSDP engine whose module is ``TransfusionQwen3p5`` instead of an HF LM."""

        def _build_module(self):
            from recipe.transfusion.transfusion_policy import (
                build_transfusion_model,
                load_transfusion_config,
            )

            config_path = os.environ.get("TRANSFUSION_CONFIG")
            if not config_path:
                raise ValueError(
                    "model_type=transfusion_diffusion requires $TRANSFUSION_CONFIG "
                    "(path to configs/navsim_qwen3p5.yaml)."
                )
            ckpt_path = os.environ.get("TRANSFUSION_CKPT")  # optional RL warm-start

            torch_dtype = self.engine_config.model_dtype
            if torch_dtype is None:
                torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

            init_context = get_init_weight_context_manager(
                use_meta_tensor=False, mesh=self.device_mesh
            )
            with init_context(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg = load_transfusion_config(config_path)
                module = build_transfusion_model(cfg, ckpt_path=ckpt_path)
                if torch_dtype is not None:
                    module = module.to(torch_dtype)

            # Mirror SFT (train_qwen3p5.build_model): per-layer gradient
            # checkpointing on the Qwen3.5 backbone. Without it the FLA
            # (gated-delta) chunk kernels retain every layer's per-chunk state
            # for backward — tens of GB at RL batch sizes (OOM at micro_bsz>=4);
            # with it, activations are recomputed in backward and the actor
            # update fits SFT-scale micro-batches (SFT enables this by default).
            # Only the TRAINING engine's module is touched — the eager rollout
            # policy / KL reference keep checkpointing off (inference path).
            if getattr(self.model_config, "enable_gradient_checkpointing", False):
                module.qwen_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Gradient checkpointing enabled on the Transfusion backbone (actor engine).")

            # Stash the processor so rollout/reward can reuse the exact tokenizer.
            self._transfusion_processor = getattr(module, "processor", None)
            self.module = module
            return module

    _REGISTERED = True
    logger.info("Registered Transfusion FSDP engine (model_type='transfusion_diffusion').")
    return True


# Register on import so `import recipe.transfusion.model_engine` is enough.
register_transfusion_engine()

__all__ = ["register_transfusion_engine"]
