"""
AUGUstus inference / evaluation script for Qwen3.5.

Usage
-----
# single sample
python test_qwen3p5.py --checkpoint checkpoints/best \
    --image scene.jpg --prompt "Drive forward"

# batch evaluation
python test_qwen3p5.py --checkpoint checkpoints/best \
    --test_data data/test.json --batch_eval --visualize

# multi-GPU evaluation
accelerate launch --num_processes 4 test_qwen3p5.py \
    --checkpoint checkpoints/best --test_data data/test.json --batch_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.dataset import TransfusionCollator, TransfusionDataset, TransfusionHFCacheDataset
from model.model_qwen3p5 import TransfusionQwen3p5
from model.utils import (
    compute_action_metrics,
    compute_pdm_scores_batch,
    count_parameters,
    fmt,
    load_config,
    load_or_compute_action_norm_stats,
    set_seed,
)

logger = logging.getLogger(__name__)


def resolve_short_prediction_vocab(cfg):
    """Resolve the short_prediction vocab JSON path.

    Priority: explicit cot.short_prediction_vocab > sibling of the train cache
    dir (``<data_dir>/short_prediction_vocab.json`` written by
    4_3_v2_short_prediction.py). Only relevant for cot_components=["short_prediction"].
    """
    cot_cfg = cfg.get("cot", {}) or {}
    explicit = cot_cfg.get("short_prediction_vocab")
    if explicit:
        return explicit
    train_path = (cfg.get("data", {}) or {}).get("train_data_path")
    if train_path:
        return str(Path(train_path).parent / "short_prediction_vocab.json")
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", default=None)
    p.add_argument("--video", default=None)
    p.add_argument("--prompt", default="Predict the trajectory.")
    p.add_argument("--test_data", default=None)
    p.add_argument("--batch_eval", action="store_true")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--cfg_scale", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--visualize", action="store_true")
    # ----- Flow Matching SDE sampler (instead of the default ODE) -----
    p.add_argument("--fm_sde", action="store_true",
                   help="Flow matching only: use the stochastic reverse-time SDE "
                        "sampler instead of the deterministic Euler ODE.")
    p.add_argument("--sde_noise_scale", type=float, default=1.0,
                   help="Stochasticity multiplier for the SiT FM SDE sampler "
                        "(0 -> ODE; larger -> more injected noise). Ignored "
                        "when --sde_kernel=flow_grpo.")
    # ----- GRPO TRAINING-rollout kernel (train/inference parity) -----
    p.add_argument("--sde_kernel", type=str, default="sit",
                   choices=["sit", "flow_grpo"],
                   help="SDE kernel used with --fm_sde: 'sit' (SiT / stochastic-"
                        "interpolants, driven by --sde_noise_scale) or "
                        "'flow_grpo' (the GRPO TRAINING rollout kernel: per-step "
                        "Gaussian noise injected at EVERY step incl. the last, "
                        "output clamped to [-1, 1] — mirrors sample_grpo).")
    p.add_argument("--sde_noise_level", type=str, default="0.7",
                   help="flow_grpo kernel: the reverse-SDE g(t) schedule's "
                        "std-scale hyper-parameter. JSON scalar ('0.05') or "
                        "per-dim list ('[0.05, 0.05, 0.02]'); 0 -> ODE.")
    p.add_argument("--sde_smooth_noise", action="store_true",
                   help="flow_grpo kernel: colour the sampled SDE noise along "
                        "the waypoint axis (RBF Cholesky, unit marginal "
                        "variance) like the RL rollout's rl_noise_smooth.")
    p.add_argument("--sde_smooth_length_scale", type=float, default=2.0,
                   help="flow_grpo kernel + smooth noise: correlation length "
                        "(in waypoints) for the position dims (x, y).")
    p.add_argument("--sde_smooth_heading_length_scale", type=float, default=1.0,
                   help="flow_grpo kernel + smooth noise: correlation length "
                        "for the heading dim (<=0 -> i.i.d. heading noise).")
    # ----- Per-step denoising input/output dump -----
    p.add_argument("--save_steps", action="store_true",
                   help="Save every denoising step's inputs/outputs to "
                        "<output_dir>/denoise_steps/ for offline statistics.")
    # ----- Eval-time CoT generation switch -----
    p.add_argument("--generate_cot", choices=["auto", "1", "0"], default="auto",
                   help="CoT generation switch: 'auto' (default) follows the config "
                        "(cot.enable_cot AND cot.eval_generate_cot); '1' forces CoT "
                        "generation ON (CoT-trained models only); '0' forces "
                        "action-only (skips the autoregressive CoT decode, faster).")
    return p.parse_args()


def _parse_sde_noise_level(val: str):
    """Parse --sde_noise_level: JSON scalar ('0.05') or per-dim list ('[0.05, 0.05, 0.02]')."""
    import json as _json
    try:
        parsed = _json.loads(val)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"--sde_noise_level must be a JSON number or list of numbers, got {val!r}"
        ) from exc
    if isinstance(parsed, (int, float)):
        return float(parsed)
    if isinstance(parsed, list) and parsed and all(isinstance(v, (int, float)) for v in parsed):
        return [float(v) for v in parsed]
    raise argparse.ArgumentTypeError(
        f"--sde_noise_level must be a JSON number or list of numbers, got {val!r}"
    )


def load_model(cfg, ckpt_path, device="cuda"):
    mc, tc, dc = cfg["model"], cfg["trajectory"], cfg["diffusion"]
    cot_cfg = cfg.get("cot", {})
    norm_cfg = cfg.get("normalization", {})

    def _strip_state_dict_prefixes(state_dict, prefixes=("module.", "model.")):
        """Strip common prefixes from state_dict keys (e.g. 'module.', 'model.').

        Repeated prefixes like 'module.model.' are removed iteratively.
        """
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                        changed = True
            new_sd[new_k] = v
        return new_sd

    # Use ONLY the config passed via --config. We intentionally do NOT read any
    # config.yaml that may sit next to the checkpoint, so all parameters
    # (num_inference_steps, diffusion/model/trajectory/data/normalization, ...)
    # come from --config exactly as given. mc/tc/dc/cot_cfg/norm_cfg
    # above are already derived from that config.
    cp = Path(ckpt_path)

    # Always initialize the Qwen backbone from the configured HF model name.
    # Some training checkpoints save the entire Transfusion model (with
    # keys like 'model.action_decoder...') which will confuse
    # `from_pretrained(checkpoint_dir)` and cause partial HF loading
    # warnings. Use the original `qwen_model_name` and then load the
    # full checkpoint into our assembled model below.
    model_name_or_path = mc["qwen_model_name"]
    logger.info(f"Initializing backbone from: {model_name_or_path}")

    # Load action normalization stats (compute only if file missing)
    action_stats = None
    if norm_cfg.get("action_norm", True):
        action_stats = load_or_compute_action_norm_stats(
            stats_file=norm_cfg.get("stats_file"),
            train_data_path=cfg["data"]["train_data_path"],
            action_dim=tc["action_dim"],
            action_repr=tc.get("action_repr", "absolute"),
        )

    model = TransfusionQwen3p5(
        qwen_model_name=model_name_or_path,
        traj_token=tc["traj_token"],
        action_dim=tc["action_dim"],
        num_inference_steps=dc["num_inference_steps"],
        timestep_shift=dc.get("timestep_shift", dc.get("shift", 1.0)),
        lambda_diffusion=dc["lambda_diffusion"],
        cfg_scale=dc["cfg_scale"],
        freeze_vision_encoder=True,
        freeze_llm=True,
        max_cot_tokens=cot_cfg.get("max_cot_tokens", 256),
        action_norm=norm_cfg.get("action_norm", True),
        action_stats=action_stats,
        dim_weights=norm_cfg.get("dim_weights"),
        action_repr=tc.get("action_repr", "absolute"),
        action_offset=tc.get("action_offset"),
        ada_layer_mode=dc.get("ada_layer_mode", "full"),
        action_loss_weight=dc.get("action_loss_weight", 1.0),
        recon_loss_weight=dc.get("recon_loss_weight", 1.0),
        clip_sample_range=dc.get("clip_sample_range", None),
        special_FFN=mc.get("special_FFN", False),
        special_QKVO=mc.get("special_QKVO"),
        # ----- short_prediction CoT (compact <|sp_...|> tokens) -----
        cot_components=cot_cfg.get("cot_components", None),
        short_prediction_vocab=resolve_short_prediction_vocab(cfg),
        # ----- action-first: action -> CoT -----
        use_action_first=cot_cfg.get("use_action_first", False),
        # action-first eval-time CoT switch: True=action+CoT (default); False=action only.
        action_first_generate_cot=cot_cfg.get("action_first_generate_cot", True),
    )
    cp = Path(ckpt_path)
    sd = None
    # Try single-file checkpoints first
    for name in ("pytorch_model.bin", "model.safetensors"):
        fp = cp / name
        if fp.exists():
            logger.info(f"Loading checkpoint: {fp} ({fp.stat().st_size / 1e9:.1f} GB)")
            if name.endswith(".bin"):
                sd = torch.load(fp, map_location="cpu", weights_only=False)
            else:
                from safetensors.torch import load_file
                sd = load_file(fp)
            break

    # Try sharded safetensors in pytorch_model/ subdirectory or directly in cp
    if sd is None:
        pm_dir = cp / "pytorch_model"
        shards = sorted(pm_dir.glob("model-*.safetensors")) if pm_dir.is_dir() else []
        if not shards:
            shards = sorted(cp.glob("model-*.safetensors"))
        if shards:
            from safetensors.torch import load_file
            logger.info(f"Loading {len(shards)} sharded safetensors from {shards[0].parent}")
            sd = {}
            for shard in shards:
                sd.update(load_file(str(shard)))

    if sd is not None:
        # Some checkpoints save model weights under a top-level prefix like
        # 'model.' or training with DDP adds 'module.'. Strip those so
        # `load_state_dict` matches the model's parameter names.
        sd = _strip_state_dict_prefixes(sd)
        # Cast fp32 checkpoint to bf16 to halve memory and speed up loading
        for k in sd:
            if sd[k].dtype == torch.float32:
                sd[k] = sd[k].to(torch.bfloat16)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        del sd  # free ~9GB immediately
        logger.info(f"Checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        if missing:
            logger.warning(f"Missing keys (first 10): {missing[:10]}")
    elif cp.suffix == ".pt":
        sd = torch.load(cp, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = _strip_state_dict_prefixes(sd)
        model.load_state_dict(sd, strict=False)
    else:
        logger.warning("No standard ckpt found; attempting accelerate load_state")

    logger.info(f"Moving model to {device} ...")
    model = model.to(dtype=torch.bfloat16, device=device).eval()
    logger.info("Model ready.")
    return model


def single_inference(model, image_path, video_path, prompt, cfg_scale, device,
                     fm_sde=False, sde_noise_scale=1.0,
                     sde_kernel="sit", sde_noise_level=None,
                     sde_smooth_noise=None):
    proc = model.processor
    content = []
    images, videos = None, None
    if video_path:
        from model.dataset import load_video_frames
        frames = load_video_frames(video_path)
        content.append({"type": "video", "video": frames})
        videos = [frames]
    elif image_path:
        img = Image.open(image_path).convert("RGB")
        content.append({"type": "image", "image": img})
        images = [img]
    content.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if videos:
        inp = proc(text=[text], videos=videos, padding=True, return_tensors="pt")
    elif images:
        inp = proc(text=[text], images=images, padding=True, return_tensors="pt")
    else:
        inp = proc(text=[text], padding=True, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}

    traj, _cot_ids = model.generate_trajectory(
        input_ids=inp["input_ids"],
        attention_mask=inp.get("attention_mask"),
        pixel_values=inp.get("pixel_values"),
        image_grid_thw=inp.get("image_grid_thw"),
        pixel_values_videos=inp.get("pixel_values_videos"),
        video_grid_thw=inp.get("video_grid_thw"),
        cfg_scale=cfg_scale,
        fm_sde=fm_sde,
        sde_noise_scale=sde_noise_scale,
        sde_kernel=sde_kernel,
        sde_noise_level=sde_noise_level,
        sde_smooth_noise=sde_smooth_noise,
    )
    # Round trajectory to 2 decimals so single-inference output matches the
    # batch-eval convention (2-decimal-rounded trajectory everywhere).
    return np.round(traj.cpu().numpy()[0], 2)


def batch_evaluate(model, test_path, cfg, accelerator, batch_size, max_samples, output_dir,
                   fm_sde=False, sde_noise_scale=1.0,
                   sde_kernel="sit", sde_noise_level=None,
                   sde_smooth_noise=None, save_steps=False, args_generate_cot="auto"):
    # Unwrap DDP / DeepSpeed wrapper so we can access .processor etc.
    raw_model = getattr(model, 'module', model)
    tc, dc = cfg["trajectory"], cfg["data"]
    data_format = dc.get("data_format", "json")
    cot_cfg = cfg.get("cot", {})

    # Configure processor for pre-sampled video frames (MUST match train_qwen3p5.py):
    # - do_sample_frames=False -> use the already-decoded frames as-is; otherwise the
    #   Qwen3.5 video processor re-samples them using a default fps (24), changing the
    #   frame count vs training and creating a train/eval mismatch.
    # - Suppress the (cosmetic) fps warning emitted when pre-sampled frames lack metadata.
    if getattr(raw_model.processor, "video_processor", None) is not None:
        raw_model.processor.video_processor.do_sample_frames = False
        import transformers.models.qwen3_vl.processing_qwen3_vl as _qwen3_proc_mod
        _qwen3_proc_mod.logger.warning_once = lambda *a, **kw: None

    if data_format == "hf_cache":
        ds = TransfusionHFCacheDataset(
            cache_dir=test_path,
            processor=raw_model.processor,
            prompt_file=dc["prompt_file"],
            traj_token=tc["traj_token"],
            action_dim=tc["action_dim"],
            is_training=False,
            cot_components=cot_cfg.get("cot_components", None),
            video_fps=dc.get("video_fps", 2.0),
        )
    else:
        ds = TransfusionDataset(
            test_path,
            processor=raw_model.processor,
            traj_token=tc["traj_token"],
            action_dim=tc["action_dim"],
            video_max_frames=dc["video_max_frames"],
            video_fps=dc["video_fps"],
            is_training=False,
        )
    if max_samples and max_samples < len(ds):
        if data_format == "hf_cache":
            ds.hf_dataset = ds.hf_dataset.select(range(max_samples))
        else:
            ds.data = ds.data[:max_samples]

    enable_cot = cfg.get("cot", {}).get("enable_cot", False)
    # Eval-time CoT toggle, controlled ONLY from the config (cot.eval_generate_cot).
    # Default True => generate the action AND the post-hoc CoT (eval CoT stays
    # visible, e.g. during training). Set cot.eval_generate_cot: false in the YAML
    # for fast real testing: only the trajectory is produced and the slow
    # autoregressive CoT decode is skipped. Gated by enable_cot so a model without
    # CoT never tries to generate one.
    eval_generate_cot = bool(cfg.get("cot", {}).get("eval_generate_cot", True))
    generate_cot = enable_cot and eval_generate_cot
    # CLI override (--generate_cot 1|0); 'auto' keeps the config behaviour.
    if args_generate_cot == "0":
        generate_cot = False
    elif args_generate_cot == "1":
        if not enable_cot:
            logger.warning("--generate_cot 1 ignored: the model was not trained "
                           "with CoT (cot.enable_cot=false in the config).")
        else:
            generate_cot = True

    if accelerator.is_local_main_process:
        if fm_sde and sde_kernel == "flow_grpo":
            logger.info(
                "FM sampler: SDE kernel=flow_grpo (GRPO TRAINING-rollout parity) "
                f"noise_level={sde_noise_level}, "
                f"smooth_noise={sde_smooth_noise} — noise injected at EVERY step, "
                "final chain clamped to [-1, 1]"
            )
        elif fm_sde:
            logger.info(f"FM sampler: SDE (sit, sde_noise_scale={sde_noise_scale})")
        else:
            logger.info("FM sampler: ODE (Euler)")
        if save_steps:
            logger.info("Per-step denoising dump ENABLED -> <output_dir>/denoise_steps/")
        if enable_cot and not generate_cot:
            logger.info("CoT generation DISABLED (action-only, faster eval) via "
                        "cot.eval_generate_cot=false or --generate_cot 0.")
        elif enable_cot and generate_cot and args_generate_cot == "1":
            logger.info("CoT generation FORCED ON via --generate_cot 1.")
        elif not enable_cot:
            logger.info("Model without CoT (cot.enable_cot=false): action-only eval.")

    collator = TransfusionCollator(raw_model.processor, tc["traj_token"], tc["action_dim"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                    collate_fn=collator, pin_memory=True)
    dl = accelerator.prepare(dl)

    preds, gts = [], []
    # Keep both post-denorm variants from the SAME model sample so ST-P3 can
    # compare offset OFF vs ON without running the VLM/diffusion twice.
    preds_no_offset, preds_with_offset = [], []
    preds_normed = []
    all_cot_texts = []  # decoded CoT strings per sample
    # Token-length diagnostics: input prompt length AND generated CoT (output)
    # length, so we can quantify how much short_prediction shrinks generation.
    prompt_len_all = []
    cot_len_all = []
    meta_keys = ["_frame_id", "_scene_token", "_driving_command",
                 "_ego_dynamic_state", "_historical_x_y_angle", "_historical_vx_vy",
                 "_reasoning_trace"]
    meta_all = {k: [] for k in meta_keys}
    tokenizer = raw_model.processor.tokenizer if hasattr(raw_model, 'processor') else None
    pad_id = (tokenizer.pad_token_id if tokenizer is not None else 0) or 0
    cot_cfg = cfg.get("cot", {})
    cot_temperature = cot_cfg.get("cot_temperature", 0.7)
    cot_top_p = cot_cfg.get("cot_top_p", 0.9)
    cot_top_k = cot_cfg.get("cot_top_k", 50)
    # Per-step denoising dump: one buffer per denoising step, lazily sized on
    # the first batch. Each buffer accumulates gathered (global) tensors.
    step_buffers = None
    model.eval()
    batch_idx = 0
    for batch in tqdm(dl, desc=f"Evaluating ({len(ds)} samples)", disable=not accelerator.is_local_main_process):
        # Fresh recorder per batch; the model appends one entry per denoising step.
        step_recorder = [] if save_steps else None
        result = raw_model.generate_trajectory(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            return_normed=True,
            cfg_scale=cfg["diffusion"]["cfg_scale"],
            generate_cot=generate_cot,
            cot_temperature=cot_temperature,
            cot_top_p=cot_top_p,
            cot_top_k=cot_top_k,
            fm_sde=fm_sde,
            sde_noise_scale=sde_noise_scale,
            sde_kernel=sde_kernel,
            sde_noise_level=sde_noise_level,
            sde_smooth_noise=sde_smooth_noise,
            step_recorder=step_recorder,
        )
        # `generate_trajectory` may return (traj, cot_ids) or
        # (traj, cot_ids, normed_traj) or (traj, cot_ids, normed_traj, denoised)
        if isinstance(result, tuple) or isinstance(result, list):
            if len(result) >= 3:
                pred, _cot_ids, pred_normed = result[0], result[1], result[2]
            else:
                pred, _cot_ids = result[0], result[1]
                pred_normed = None
        else:
            # Unexpected shape: wrap as single-element
            pred = result
            _cot_ids = None
            pred_normed = None

        # Build offset-OFF and offset-ON trajectories from this SAME sample.
        # `pred` returned by the model already follows trajectory.action_offset.enabled.
        # Undo it first when enabled, then build the ON variant from the configured
        # [lateral(right+), longitudinal(forward+)] offset. This happens in the
        # source action space AFTER denormalization (x=forward, y=left), exactly
        # matching model.apply_action_offset: x += longitudinal, y -= lateral.
        ao_cfg = tc.get("action_offset") or {}
        ao_values = ao_cfg.get("offset") or [0.0, 0.0]
        if len(ao_values) != 2:
            raise ValueError(f"trajectory.action_offset.offset must be [lateral, longitudinal], got {ao_values!r}")
        offset_vec = torch.zeros(pred.shape[-1], device=pred.device, dtype=pred.dtype)
        offset_vec[0] = float(ao_values[1])
        if pred.shape[-1] > 1:
            offset_vec[1] = -float(ao_values[0])
        pred_no_offset = pred - offset_vec if ao_cfg.get("enabled", False) else pred.clone()
        pred_with_offset = pred_no_offset + offset_vec
        pred = pred_with_offset if ao_cfg.get("enabled", False) else pred_no_offset

        # Round each independently, just as two separate source-code inference
        # runs (offset OFF / ON) would be rounded.
        if torch.is_tensor(pred):
            pred = torch.round(pred * 100.0) / 100.0
            pred_no_offset = torch.round(pred_no_offset * 100.0) / 100.0
            pred_with_offset = torch.round(pred_with_offset * 100.0) / 100.0
        if torch.is_tensor(pred_normed):
            pred_normed = torch.round(pred_normed * 100.0) / 100.0

        # Decode CoT tokens to text
        batch_cot_texts = []
        if _cot_ids is not None and tokenizer is not None:
            for i in range(_cot_ids.shape[0]):
                ids = _cot_ids[i]
                # Filter out padding (0) and special tokens
                ids = ids[ids > 0]
                text = tokenizer.decode(ids, skip_special_tokens=True).strip()
                batch_cot_texts.append(text)
        else:
            batch_cot_texts = [""] * pred.shape[0]

        # Save and log first batch locally (before gathering across devices)
        if batch_idx == 0 and accelerator.is_local_main_process:
            first_dir = Path(output_dir) / "first_batch"
            first_dir.mkdir(parents=True, exist_ok=True)
            try:
                np.save(first_dir / "preds_local.npy", pred.cpu().numpy())
            except Exception:
                logger.exception("Failed saving first-batch preds")
            if pred_normed is not None:
                try:
                    np.save(first_dir / "preds_normed_local.npy", pred_normed.cpu().numpy())
                except Exception:
                    logger.exception("Failed saving first-batch normed preds")
            try:
                np.save(first_dir / "gts_local.npy", batch["actions"].cpu().numpy())
            except Exception:
                logger.exception("Failed saving first-batch gts")
            # Also save JSON and simple trajectory plots per-sample
            try:
                preds_np = pred.cpu().numpy()
                gts_np = batch["actions"].cpu().numpy()
                preds_normed_np = pred_normed.cpu().numpy() if pred_normed is not None else None
                json_list = []
                B = preds_np.shape[0]
                for i in range(B):
                    item = {
                        "pred": preds_np[i].tolist(),
                        "gt": gts_np[i].tolist(),
                    }
                    if preds_normed_np is not None:
                        item["pred_normed"] = preds_normed_np[i].tolist()
                    if i < len(batch_cot_texts) and batch_cot_texts[i]:
                        item["cot"] = batch_cot_texts[i]
                    # include some meta if available (best-effort)
                    for mk in ("_frame_id", "_scene_token", "_driving_command"):
                        if mk in batch and i < len(batch[mk]):
                            item[mk] = batch[mk][i]
                    json_list.append(item)
                    # plot trajectories (pred vs gt)
                    try:
                        plot_trajectory(preds_np[i], gts_np[i], save_path=str(first_dir / f"traj_{i:04d}.png"),
                                        title=f"Sample {i}")
                    except Exception:
                        logger.exception("Failed plotting trajectory for sample %d", i)
                with open(first_dir / "first_batch.json", "w", encoding="utf-8") as fj:
                    json.dump(json_list, fj, indent=2, ensure_ascii=False)
            except Exception:
                logger.exception("Failed saving first-batch JSON/plots")
            # Log small summaries
            logger.info(f"First-batch saved to {first_dir} | preds shape: {getattr(pred, 'shape', None)} | gts shape: {getattr(batch.get('actions'), 'shape', None)}")
        preds.append(accelerator.gather(pred).cpu())
        preds_no_offset.append(accelerator.gather(pred_no_offset).cpu())
        preds_with_offset.append(accelerator.gather(pred_with_offset).cpu())
        preds_normed.append(accelerator.gather(pred_normed).cpu())
        gts.append(accelerator.gather(batch["actions"]).cpu())
        # ---- per-step denoising dump: gather so rows align with preds ----
        if save_steps and step_recorder:
            if step_buffers is None:
                _step_keys = [k for k, v in step_recorder[0].items() if torch.is_tensor(v)]
                step_buffers = [{k: [] for k in _step_keys}
                                for _ in range(len(step_recorder))]
            for si, rec in enumerate(step_recorder):
                for k in step_buffers[si]:
                    step_buffers[si][k].append(accelerator.gather(rec[k]).cpu())
        # ---- token-length diagnostics (input prompt + generated CoT) ----
        B_local = pred.shape[0]
        attn_b = batch.get("attention_mask")
        if attn_b is not None:
            in_lens_b = attn_b.sum(dim=-1).float()
        else:
            in_lens_b = torch.full((B_local,), float(batch["input_ids"].shape[1]),
                                   device=pred.device)
        if _cot_ids is not None:
            out_lens_b = (_cot_ids != pad_id).sum(dim=-1).float().to(pred.device)
        else:
            out_lens_b = torch.zeros(B_local, device=pred.device)
        prompt_len_all.append(accelerator.gather(in_lens_b).cpu())
        cot_len_all.append(accelerator.gather(out_lens_b).cpu())
        # Gather string meta per-batch to match tensor gathering order (batch-major)
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            for mk in meta_keys:
                if mk in batch:
                    gathered = [None] * accelerator.num_processes
                    dist.all_gather_object(gathered, batch[mk])
                    for g in gathered:
                        meta_all[mk].extend(g)
            gathered_cot = [None] * accelerator.num_processes
            dist.all_gather_object(gathered_cot, batch_cot_texts)
            for g in gathered_cot:
                all_cot_texts.extend(g)
        else:
            all_cot_texts.extend(batch_cot_texts)
            for mk in meta_keys:
                if mk in batch:
                    meta_all[mk].extend(batch[mk])
        batch_idx += 1

    preds_np = torch.cat(preds).numpy()
    preds_no_offset_np = torch.cat(preds_no_offset).numpy()
    preds_with_offset_np = torch.cat(preds_with_offset).numpy()
    preds_normed_np = torch.cat(preds_normed).numpy()
    gts_np = torch.cat(gts).numpy()

    # Trim padding added by DistributedSampler (rounds up to num_processes)
    real_n = len(ds)
    if preds_np.shape[0] > real_n:
        logger.info(f"Trimming gathered results from {preds_np.shape[0]} to {real_n} (DistributedSampler padding)")
        preds_np = preds_np[:real_n]
        preds_no_offset_np = preds_no_offset_np[:real_n]
        preds_with_offset_np = preds_with_offset_np[:real_n]
        preds_normed_np = preds_normed_np[:real_n]
        gts_np = gts_np[:real_n]
        all_cot_texts = all_cot_texts[:real_n]
        for mk in meta_keys:
            meta_all[mk] = meta_all[mk][:real_n]

    # ---- Save per-step denoising inputs/outputs (one .npz per step) ----
    # Rows are aligned with predictions.npy / ground_truths.npy / _frame_id.
    if step_buffers is not None and accelerator.is_main_process:
        steps_dir = Path(output_dir) / "denoise_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        last_shapes = {}
        for si, buf in enumerate(step_buffers):
            arrs = {}
            for k, chunks in buf.items():
                a = torch.cat(chunks).numpy()
                if a.shape[0] > real_n:
                    a = a[:real_n]
                arrs[k] = a
            np.savez(steps_dir / f"step_{si:02d}.npz", **arrs)
            last_shapes = {k: list(v.shape) for k, v in arrs.items()}
        sampler_name = "sde" if fm_sde else "ode"
        if fm_sde:
            sampler_name = f"sde_{sde_kernel}"
        meta_info = {
            "num_steps": len(step_buffers),
            "num_samples": int(real_n),
            "diffusion_type": "flow_matching",
            "sampler": sampler_name,
            "sde_noise_scale": float(sde_noise_scale) if fm_sde else 0.0,
            "sde_kernel": sde_kernel if fm_sde else None,
            "sde_noise_level": sde_noise_level if (fm_sde and sde_kernel == "flow_grpo") else None,
            "sde_smooth_noise": sde_smooth_noise if (fm_sde and sde_kernel == "flow_grpo") else None,
            "num_inference_steps": cfg["diffusion"].get("num_inference_steps"),
            "prediction_type": "x0",
            "arrays": list(step_buffers[0].keys()),
            "array_shapes": last_shapes,
            "ordering": "row i aligns with predictions.npy / ground_truths.npy / _frame_id[i]",
            "notes": "Normalized action space. x_in = state entering the step; "
                     "x_out = state after the step; model_out = raw network output; "
                     "v_pred = velocity (flow matching only); t = step time/timestep.",
        }
        with open(steps_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)
        logger.info(f"Saved per-step denoising dump to {steps_dir} "
                    f"({len(step_buffers)} steps x {real_n} samples, sampler={sampler_name})")

    metrics = compute_action_metrics(
        torch.from_numpy(preds_np), torch.from_numpy(gts_np),
        action_dim=tc["action_dim"],
    )
    metrics["num_samples"] = preds_np.shape[0]

    # ---- token-length diagnostics ----
    if prompt_len_all:
        all_in = torch.cat(prompt_len_all)[:real_n]
        if all_in.numel() > 0:
            metrics["prompt_len_mean"] = all_in.mean().item()
            metrics["prompt_len_max"] = all_in.max().item()
    if cot_len_all:
        all_out = torch.cat(cot_len_all)[:real_n]
        if all_out.numel() > 0:
            metrics["cot_len_mean"] = all_out.mean().item()
            metrics["cot_len_max"] = all_out.max().item()

    # PDMS evaluation. Every rank holds the SAME full preds_np/frame_ids_all
    # (gathered above). Standalone test needs the COMPLETE per-sample PDMS list
    # on rank 0 for the JSON dump, so -- unlike training-time evaluate(), which
    # only needs aggregate metrics and therefore shards disjointly -- all ranks
    # score the full set: their ProcessPoolExecutors collectively warm the
    # shared OS page cache over the NAS metric cache (each frame's pickle is
    # fetched from NAS ~once, then reused across ranks), so this stays fast while
    # giving rank 0 a full per_sample list. main() only consumes rank 0's result.
    frame_ids_all = meta_all.get("_frame_id", [])
    test_metric_cache = cfg.get("data", {}).get("metric_cache_test")
    pdms_per_sample = None
    if frame_ids_all:
        pdms_metrics = compute_pdm_scores_batch(
            preds_np, frame_ids_all,
            metric_cache_path=test_metric_cache,
            return_per_sample=True,
            show_progress=accelerator.is_main_process,
        )
        pdms_per_sample = pdms_metrics.pop("per_sample", None)
        metrics.update(pdms_metrics)

    return (metrics, preds_np, gts_np, preds_normed_np, meta_all,
            all_cot_texts, pdms_per_sample,
            preds_no_offset_np, preds_with_offset_np)


def plot_trajectory(pred, gt=None, save_path="traj.png", title="Trajectory"): 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(pred[:, 0], pred[:, 1], "b-o", ms=5, lw=2, label="Predicted")
    ax.scatter(*pred[0, :2], c="blue", s=80, zorder=5, marker="s")
    ax.scatter(*pred[-1, :2], c="blue", s=80, zorder=5, marker="^")
    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], "r--o", ms=5, lw=2, label="GT")
        ax.scatter(*gt[0, :2], c="red", s=80, zorder=5, marker="s")
        ax.scatter(*gt[-1, :2], c="red", s=80, zorder=5, marker="^")
    ax.set_xlabel("dim-0")
    ax.set_ylabel("dim-1")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    set_seed(args.seed)
    if args.num_inference_steps is not None:
        cfg["diffusion"]["num_inference_steps"] = args.num_inference_steps

    # ---- Resolve the GRPO training-rollout SDE knobs ----
    sde_noise_level = _parse_sde_noise_level(args.sde_noise_level)
    sde_smooth_noise = None
    if args.sde_smooth_noise:
        sde_smooth_noise = {
            "length_scale": args.sde_smooth_length_scale,
            "heading_length_scale": args.sde_smooth_heading_length_scale,
            "n_position_dims": 2,
            "anchor": None,
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.batch_eval and args.test_data:
        accel = Accelerator()
        model = load_model(cfg, args.checkpoint, str(accel.device))
        model = accel.prepare(model)
        pi = count_parameters(model)
        logger.info(f"Model {fmt(pi['total'])} params, traj_token={cfg['trajectory']['traj_token']}")
        logger.info(f"Diffusion: flow_matching (x0), "
                     f"num_inference_steps={cfg['diffusion']['num_inference_steps']}, "
                     f"timestep_shift={cfg['diffusion'].get('timestep_shift', None)}, "
                     f"cfg_scale={cfg['diffusion']['cfg_scale']}, "
                     f"clip_sample_range={cfg['diffusion'].get('clip_sample_range', None)}, ")

        (metrics, preds, gts, preds_normed, meta, all_cot_texts,
         pdms_per_sample, preds_no_offset, preds_with_offset) = batch_evaluate(
            model, args.test_data, cfg, accel, args.batch_size, args.max_samples, out,
            fm_sde=args.fm_sde, sde_noise_scale=args.sde_noise_scale,
            sde_kernel=args.sde_kernel, sde_noise_level=sde_noise_level,
            sde_smooth_noise=sde_smooth_noise, save_steps=args.save_steps,
            args_generate_cot=args.generate_cot,
        )
        # All ranks gathered identical prediction/meta arrays; rank 0 reports.
        accel.wait_for_everyone()
        if accel.is_main_process:
            logger.info("=" * 50)
            logger.info(f"  Samples : {metrics['num_samples']}")
            logger.info(f"  ADE     : {metrics['ADE']:.4f}")
            logger.info(f"  FDE     : {metrics['FDE']:.4f}")
            if "PDMS" in metrics:
                logger.info(f"  PDMS    : {metrics['PDMS']:.4f}")
                logger.info(f"  NC      : {metrics['NC']:.4f}")
                logger.info(f"  DAC     : {metrics['DAC']:.4f}")
                logger.info(f"  DDC     : {metrics['DDC']:.4f}")
                logger.info(f"  EP      : {metrics['EP']:.4f}")
                logger.info(f"  TTC     : {metrics['TTC']:.4f}")
                logger.info(f"  C       : {metrics['C']:.4f}")
            logger.info("=" * 50)
            eval_summary = {k: float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in metrics.items()}
            with open(out / "eval_results.json", "w") as f:
                json.dump(eval_summary, f, indent=2)
            np.save(out / "predictions.npy", preds)
            np.save(out / "predictions_no_offset.npy", preds_no_offset)
            np.save(out / "predictions_with_offset.npy", preds_with_offset)
            np.save(out / "ground_truths.npy", gts)

            # ---- Save detailed per-sample results ----
            time_labels = [f"{(t+1)*0.5:.1f}s" for t in range(cfg["trajectory"]["traj_token"])]
            action_dim = cfg["trajectory"]["action_dim"]

            def _traj_to_str(arr, labels):
                """Format trajectory array [T, D] as a readable json-code-block string."""
                items = []
                for t in range(len(arr)):
                    xyz = [round(float(arr[t, 0]), 2),
                           round(float(arr[t, 1]), 2),
                           round(float(arr[t, 2]), 2) if action_dim >= 3 else 0.0]
                    items.append(f'\t{{"x_y_radian": {xyz}, "label": "{labels[t]}"}}')
                return "```json\n[\n" + ",\n".join(items) + "\n]\n```"

            detailed_results = []
            for i in range(len(preds)):
                pred_i = preds[i]   # [T, D] denormalized
                gt_i = gts[i]       # [T, D]
                normed_i = preds_normed[i]  # [T, D] in [-1, 1]
                fid = meta["_frame_id"][i] if i < len(meta["_frame_id"]) else ""
                scene_token = meta["_scene_token"][i] if i < len(meta["_scene_token"]) else ""
                driving_cmd = meta["_driving_command"][i] if i < len(meta["_driving_command"]) else ""
                ego_state = meta["_ego_dynamic_state"][i] if i < len(meta["_ego_dynamic_state"]) else ""
                hist_traj = meta["_historical_x_y_angle"][i] if i < len(meta["_historical_x_y_angle"]) else ""
                hist_vel = meta["_historical_vx_vy"][i] if i < len(meta["_historical_vx_vy"]) else ""

                # Per-step displacement error (L2 on x,y)
                d = min(action_dim, 2)
                disp = np.linalg.norm(pred_i[:, :d] - gt_i[:, :d], axis=-1)  # [T]
                ade_i = float(disp.mean())
                fde_i = float(disp[-1])

                sample_entry = {
                    "frame_id": fid,
                    "scene_token": scene_token,
                    "driving_command": driving_cmd,
                    "ego_dynamic_state": ego_state,
                    "historical_x_y_angle": hist_traj,
                    "historical_vx_vy": hist_vel,
                    "ADE": round(ade_i, 4),
                    "FDE": round(fde_i, 4),
                    "gt_planning": _traj_to_str(gt_i, time_labels),
                    # Primary prediction follows action_offset.enabled.
                    "pred_planning_denormed": _traj_to_str(pred_i, time_labels),
                    # Both variants come from the same denoising sample, with
                    # the configured offset applied after denormalization.
                    "pred_planning_no_offset": _traj_to_str(preds_no_offset[i], time_labels),
                    "pred_planning_with_offset": _traj_to_str(preds_with_offset[i], time_labels),
                    "pred_planning_normed": _traj_to_str(normed_i, time_labels),
                    "per_step_error": str([round(float(disp[t]), 4) for t in range(len(disp))]),
                }
                # GT reasoning trace
                gt_reasoning = meta["_reasoning_trace"][i] if i < len(meta.get("_reasoning_trace", [])) else ""
                if gt_reasoning:
                    sample_entry["gt_reasoning_trace"] = gt_reasoning
                # Predicted CoT
                if i < len(all_cot_texts) and all_cot_texts[i]:
                    sample_entry["pred_cot"] = all_cot_texts[i]
                # Per-sample PDMS metrics
                if pdms_per_sample is not None and i < len(pdms_per_sample) and pdms_per_sample[i] is not None:
                    sample_entry.update(pdms_per_sample[i])
                detailed_results.append(sample_entry)

            # Sort by FDE descending so worst cases are on top
            detailed_results.sort(key=lambda x: x["FDE"], reverse=True)

            detailed_path = out / "eval_detailed.json"
            detailed_summary = {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in metrics.items()
            }
            with open(detailed_path, "w") as f:
                json.dump({
                    "summary": detailed_summary,
                    "samples": detailed_results,
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed per-sample results saved to {detailed_path}")

            if args.visualize:
                vis = out / "vis"
                vis.mkdir(exist_ok=True)
                for i in range(min(20, len(preds))):
                    plot_trajectory(preds[i], gts[i], str(vis / f"traj_{i:04d}.png"),
                                    title=f"Sample {i}")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(cfg, args.checkpoint, device)
        pred = single_inference(
            model, args.image, args.video, args.prompt,
            args.cfg_scale, device,
            fm_sde=args.fm_sde, sde_noise_scale=args.sde_noise_scale,
            sde_kernel=args.sde_kernel, sde_noise_level=sde_noise_level,
            sde_smooth_noise=sde_smooth_noise,
        )
        logger.info(f"Predicted trajectory ({pred.shape}):\n{pred}")
        np.save(out / "prediction.npy", pred)
        if args.visualize:
            plot_trajectory(pred, save_path=str(out / "trajectory.png"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
