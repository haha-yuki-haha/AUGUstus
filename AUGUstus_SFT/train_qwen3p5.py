"""
AUGUstus SFT training script for Qwen3.5 — multi-GPU / multi-node via HuggingFace Accelerate.

Supports multiple logging backends: wandb, swanlab, tensorboard, console.

Launch examples
---------------
# single GPU
python train_qwen3p5.py --config configs/navsim_qwen3p5.yaml

# 8-GPU single node
accelerate launch --num_processes 8 train_qwen3p5.py --config configs/navsim_qwen3p5.yaml

# Resume from checkpoint
python train_qwen3p5.py --config configs/navsim_qwen3p5.yaml --resume_from checkpoints/checkpoint-5000

# DeepSpeed ZeRO-2
accelerate launch --config_file configs/accelerate_deepspeed.yaml \
    train_qwen3p5.py --config configs/navsim_qwen3p5.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Reduce CUDA memory fragmentation (helps prevent OOM after many steps)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed as acc_set_seed
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.dataset import TransfusionCollator, TransfusionDataset, TransfusionHFCacheDataset
from model.logger import (
    TrainingLogger,
    log_data_samples,
    log_eval_samples,
)
from model.model_qwen3p5 import TransfusionQwen3p5
from model.utils import (
    EMA,
    compute_action_metrics,
    compute_pdm_scores_batch,
    count_parameters,
    fmt,
    get_cosine_schedule_with_warmup,
    load_config,
    load_or_compute_action_norm_stats,
    set_seed,
)

logger = get_logger(__name__, log_level="INFO")


# ============================================================
# CLI
# ============================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--resume_from", type=str, default=None)
    return p.parse_args()


def resolve_short_prediction_vocab(cfg: dict):
    """Resolve the short_prediction vocab JSON path.

    Priority: explicit cot.short_prediction_vocab > sibling of the train cache
    dir (``<data_dir>/short_prediction_vocab.json`` as written by
    4_3_v2_short_prediction.py). Only relevant when cot_components uses
    "short_prediction".
    """
    cot_cfg = cfg.get("cot", {}) or {}
    explicit = cot_cfg.get("short_prediction_vocab")
    if explicit:
        return explicit
    train_path = (cfg.get("data", {}) or {}).get("train_data_path")
    if train_path:
        return str(Path(train_path).parent / "short_prediction_vocab.json")
    return None


def build_model(cfg: dict):
    mc, tc, dc = cfg["model"], cfg["trajectory"], cfg["diffusion"]
    cot_cfg = cfg.get("cot", {})
    norm_cfg = cfg.get("normalization", {})

    # Load or compute action normalization statistics
    action_stats = None
    if norm_cfg.get("action_norm", True):
        action_stats = load_or_compute_action_norm_stats(
            stats_file=norm_cfg.get("stats_file"),
            train_data_path=cfg["data"]["train_data_path"],
            action_dim=tc["action_dim"],
            action_repr=tc.get("action_repr", "absolute"),
        )

    model_kwargs = dict(
        qwen_model_name=mc["qwen_model_name"],
        traj_token=tc["traj_token"],
        action_dim=tc["action_dim"],
        num_inference_steps=dc["num_inference_steps"],
        timestep_shift=dc.get("timestep_shift", dc.get("shift", 1.0)),
        lambda_diffusion=dc["lambda_diffusion"],
        lambda_lm=dc.get("lambda_lm", 1.0),
        cfg_scale=dc["cfg_scale"],
        freeze_vision_encoder=mc["freeze_vision_encoder"],
        freeze_llm=mc["freeze_llm"],
        max_cot_tokens=cot_cfg.get("max_cot_tokens", 256),
        action_norm=norm_cfg.get("action_norm", True),
        action_stats=action_stats,
        dim_weights=norm_cfg.get("dim_weights"),
        action_repr=tc.get("action_repr", "absolute"),
        action_offset=tc.get("action_offset"),
        ada_layer_mode=dc.get("ada_layer_mode", "full"),
        # Backbone weight-source switch: when True (set by the RL recipe's
        # build_transfusion_model whenever a full Transfusion ckpt is about to
        # be loaded on top) the HF backbone is built from config only — its
        # 6.3GB pretrained checkpoint file is never read (cold shared-FS read
        # ~2.5 min), since every backbone weight comes from the Transfusion
        # checkpoint load that follows. Default False = SFT behaviour unchanged.
        skip_backbone_weight_load=mc.get("skip_backbone_weight_load", False),
        action_loss_weight=dc.get("action_loss_weight", 1.0),
        recon_loss_weight=dc.get("recon_loss_weight", 1.0),
        goal_point_weight=dc.get("goal_point_weight", 0.0),
        clip_sample_range=dc.get("clip_sample_range", None),
        special_FFN=mc.get("special_FFN", False),
        special_QKVO=mc.get("special_QKVO", None),
        # ----- short_prediction CoT (compact <|sp_...|> tokens) -----
        cot_components=cot_cfg.get("cot_components", None),
        short_prediction_vocab=resolve_short_prediction_vocab(cfg),
        # ----- action-first: action -> CoT -----
        use_action_first=cot_cfg.get("use_action_first", False),
        # action-first eval-time CoT switch: True=action+CoT (default); False=action only.
        action_first_generate_cot=cot_cfg.get("action_first_generate_cot", True),
    )

    return TransfusionQwen3p5(**model_kwargs)


def build_loaders(cfg: dict, processor):
    dc, tc = cfg["data"], cfg["trajectory"]
    data_format = dc.get("data_format", "json")
    cot_cfg = cfg.get("cot", {})
    enable_cot = cot_cfg.get("enable_cot", False)
    cot_components = cot_cfg.get("cot_components", None)

    if data_format == "hf_cache":
        # HuggingFace Dataset cache mode (pre-cached video tensors)
        kw = dict(
            processor=processor,
            prompt_file=dc["prompt_file"],
            traj_token=tc["traj_token"],
            action_dim=tc["action_dim"],
            enable_cot=enable_cot,
            cot_components=cot_components,
            video_fps=dc.get("video_fps", 2.0),
        )
        train_ds = TransfusionHFCacheDataset(
            dc["train_data_path"], is_training=True, **kw,
        )
        val_ds = None
        if dc.get("val_data_path") and Path(dc["val_data_path"]).exists():
            val_ds = TransfusionHFCacheDataset(
                dc["val_data_path"], is_training=False, **kw,
            )
    else:
        # JSON file mode (load videos on-the-fly)
        kw = dict(
            processor=processor,
            traj_token=tc["traj_token"],
            action_dim=tc["action_dim"],
            video_max_frames=dc["video_max_frames"],
            video_fps=dc["video_fps"],
        )
        train_ds = TransfusionDataset(dc["train_data_path"], is_training=True, **kw)
        val_ds = None
        if dc.get("val_data_path") and Path(dc["val_data_path"]).exists():
            val_ds = TransfusionDataset(dc["val_data_path"], is_training=False, **kw)

    collator = TransfusionCollator(processor, tc["traj_token"], tc["action_dim"])
    tcfg = cfg["training"]
    train_dl = DataLoader(
        train_ds,
        batch_size=tcfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=tcfg["dataloader_num_workers"],
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=tcfg["per_device_eval_batch_size"],
            shuffle=False,
            num_workers=tcfg["dataloader_num_workers"],
            collate_fn=collator,
            pin_memory=True,
        )
    return train_dl, val_dl


def _log_val_debug(model, batch, pred_denorm, normed_pred, denoised_hidden, step, num_samples):
    """Log detailed inference debug info matching training debug output."""
    actions = batch["actions"]  # GT raw actions [B, T, D]
    B = min(num_samples, actions.shape[0])

    for idx in range(B):
        gt_raw = actions[idx].cpu()  # [T, D]
        pred_raw = pred_denorm[idx].cpu()  # [T, D]

        # Normalize GT for comparison with normed prediction
        gt_norm = model.normalize_actions(actions[idx:idx+1].float()).cpu()[0]  # [T, D]
        pred_norm = normed_pred[idx].cpu() if normed_pred is not None else None  # [T, D]

        def _fmt_traj(t, max_d=None):
            lines = []
            for row in t:
                vals = row[:max_d].tolist() if max_d else row.tolist()
                lines.append("[" + ", ".join(f"{v:.4f}" for v in vals) + "]")
            return "\n      ".join(lines)

        logger.info(f"[step {step}] val_debug/sample_{idx}:", main_process_only=True)
        logger.info(f"  gt_actions_norm: {_fmt_traj(gt_norm)}", main_process_only=True)
        logger.info(f"  gt_actions_denorm: {_fmt_traj(gt_raw)}", main_process_only=True)
        if pred_norm is not None:
            logger.info(f"  pred_actions_norm: {_fmt_traj(pred_norm)}", main_process_only=True)
        logger.info(f"  pred_actions_denorm: {_fmt_traj(pred_raw)}", main_process_only=True)

        # Denoised hidden state stats (first 3 dims, like training debug)
        if denoised_hidden is not None:
            h = denoised_hidden[idx].cpu()  # [T, H]
            logger.info(f"  denoised_hidden[:, :3]: {_fmt_traj(h, max_d=3)}", main_process_only=True)
            logger.info(f"  denoised_hidden stats: min={h.min():.4f} max={h.max():.4f} "
                        f"mean={h.mean():.4f} std={h.std():.4f}", main_process_only=True)

        # Encoder-decoder reconstruction of GT (if latent mode)
        if hasattr(model, 'latent_action_proj') and model.noise_space == "latent":
            gt_f = model.normalize_actions(actions[idx:idx+1].float().to(pred_denorm.device))
            proj_dtype = next(model.latent_action_proj.parameters()).dtype
            clean_emb = model.latent_action_proj(gt_f.to(proj_dtype))
            decoder_dtype = next(model.action_decoder.parameters()).dtype
            recon = model.action_decoder(clean_emb.to(decoder_dtype)).float()
            recon_denorm = model.denormalize_actions(recon.clamp(-1, 1)).cpu()[0]
            recon_norm = recon.clamp(-1, 1).cpu()[0]
            logger.info(f"  recon_actions_norm: {_fmt_traj(recon_norm)}", main_process_only=True)
            logger.info(f"  recon_actions_denorm: {_fmt_traj(recon_denorm)}", main_process_only=True)
            # Also log clean embedding stats
            logger.info(f"  clean_embed stats: min={clean_emb.min():.4f} max={clean_emb.max():.4f} "
                        f"mean={clean_emb.mean():.4f} std={clean_emb.std():.4f}",
                        main_process_only=True)


@torch.no_grad()
def evaluate(
    model,
    val_dl,
    accelerator,
    tlogger: Optional[TrainingLogger],
    processor,
    step: int,
    log_num_samples: int = 10,
    enable_cot: bool = False,
    metric_cache_path: Optional[str] = None,
):
    """Run evaluation on ALL validation data, compute metrics including PDMS."""
    model.eval()
    preds_all, gts_all = [], []
    frame_ids_all = []
    logged_samples = False
    n = 0
    # Track generated CoT (output) token length -> complements input prompt_len.
    pad_id = processor.tokenizer.pad_token_id if processor is not None else 0
    pad_id = pad_id or 0
    cot_len_all = []

    for i, batch in enumerate(tqdm(val_dl, desc="Validating", disable=not accelerator.is_local_main_process)):
        result = model.generate_trajectory(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            pixel_values_videos=batch.get("pixel_values_videos"),
            video_grid_thw=batch.get("video_grid_thw"),
            generate_cot=enable_cot,
            return_normed=True,
        )
        # returns (trajectory, cot_ids, normed_traj, denoised_hidden)
        pred, cot_ids, normed_pred, denoised_hidden = result

        # Round the predicted trajectory to 2 decimals at the SOURCE so that every
        # downstream use (ADE/FDE + PDMS metrics, gathered tensors, per-sample
        # sample/debug logging) consistently uses the 2-decimal-rounded result.
        # Matches the test-time convention in test_qwen3p5.py.
        if torch.is_tensor(pred):
            pred = torch.round(pred * 100.0) / 100.0
        if torch.is_tensor(normed_pred):
            normed_pred = torch.round(normed_pred * 100.0) / 100.0

        # ---- gather across ranks (auto-removes DistributedSampler padding) ----
        # Without this the aggregated metrics only reflect rank-0's shard
        # (e.g. 1520 samples instead of 12146 with 8 GPUs).
        pred_g = accelerator.gather_for_metrics(pred)
        gts_g = accelerator.gather_for_metrics(batch["actions"])
        preds_all.append(pred_g.cpu())
        gts_all.append(gts_g.cpu())
        # Generated CoT length per sample (excludes trailing pad after EOT).
        if cot_ids is not None:
            cot_lens_b = (cot_ids != pad_id).sum(dim=-1).float()  # [B]
            cot_lens_g = accelerator.gather_for_metrics(cot_lens_b)
            cot_len_all.append(cot_lens_g.cpu())
        if "_frame_id" in batch:
            fid_g = accelerator.gather_for_metrics(batch["_frame_id"])
            if isinstance(fid_g, (list, tuple)):
                frame_ids_all.extend(list(fid_g))
            else:
                frame_ids_all.extend(list(fid_g))
        n += 1

        # Log first batch samples
        if not logged_samples and tlogger is not None and accelerator.is_main_process:
            log_eval_samples(
                tlogger, batch, pred, cot_ids, processor,
                step=step, num_samples=log_num_samples,
            )
            # Log detailed debug info for inference diagnosis
            unwrapped = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else model
            _log_val_debug(unwrapped, batch, pred, normed_pred, denoised_hidden, step, log_num_samples)
            logged_samples = True

    metrics = {}
    if preds_all:
        all_preds = torch.cat(preds_all)
        all_gts = torch.cat(gts_all)
        # Safety trim in case the gather returned slightly more than dataset size
        ds_len = len(val_dl.dataset) if hasattr(val_dl, "dataset") else all_preds.shape[0]
        all_preds = all_preds[:ds_len]
        all_gts = all_gts[:ds_len]
        if frame_ids_all:
            frame_ids_all = frame_ids_all[:ds_len]
        if accelerator.is_main_process:
            logger.info(
                f"Aggregated eval samples: preds={all_preds.shape[0]} "
                f"frame_ids={len(frame_ids_all)} dataset_len={ds_len}",
                main_process_only=True,
            )
        metrics.update(compute_action_metrics(all_preds, all_gts))

        # Generated CoT (output) token length -> logged as val/cot_len_mean.
        if cot_len_all:
            all_cot_lens = torch.cat(cot_len_all)[:ds_len]
            if all_cot_lens.numel() > 0:
                metrics["cot_len_mean"] = all_cot_lens.mean().item()
                metrics["cot_len_max"] = all_cot_lens.max().item()

        # PDMS evaluation -- split DISJOINTLY across ranks for speed.
        # Every rank holds the SAME full, de-duplicated all_preds/frame_ids_all
        # (produced by gather_for_metrics above). PDMS scoring is the eval
        # bottleneck: each frame needs an lzma-decompress of a NAS-hosted
        # metric-cache pickle (~0.5s) + a closed-loop simulation. Scoring the
        # WHOLE set on EVERY rank wastes world_size x the CPU; scoring it all on
        # rank 0 ALONE serialises onto one rank's num_workers with all-cold NAS
        # reads (~30 min for 12k frames). Instead each rank scores only the
        # DISJOINT slice frame[rank::world]: the slices partition the full set so
        # every frame is scored EXACTLY ONCE with world_size * num_workers
        # effective parallelism (e.g. 8 ranks x 8 workers = 64 -> ~2 min). The
        # per-metric SUMS + success count are then all-reduced so every rank
        # recovers the exact full-set averages. Only rank 0 draws the tqdm bar.
        # NOTE: all ranks must reach the all_reduce below, so the guard uses only
        # values that are identical on every rank (frame_ids_all/metric_cache_path).
        if frame_ids_all and metric_cache_path:
            _world = max(1, accelerator.num_processes)
            _rank = accelerator.process_index
            _shard_preds = all_preds[_rank::_world].numpy()
            _shard_fids = frame_ids_all[_rank::_world]
            logger.info(
                f"Computing PDMS for {len(frame_ids_all)} samples "
                f"(~{len(_shard_fids)}/rank x {_world} ranks) ...",
                main_process_only=True,
            )
            _shard_res = compute_pdm_scores_batch(
                _shard_preds, _shard_fids,
                metric_cache_path=metric_cache_path,
                return_count=True,
                show_progress=accelerator.is_main_process,
            )
            # Reduce per-metric SUMS (avg * count) + count -> exact weighted
            # full-set mean. float64 keeps the division precise.
            _pdms_keys = ["PDMS", "NC", "DAC", "DDC", "EP", "TTC", "C"]
            _cnt = float(_shard_res.get("count", 0))
            _vec = torch.tensor(
                [_shard_res.get(k, 0.0) * _cnt for k in _pdms_keys] + [_cnt],
                dtype=torch.float64, device=accelerator.device,
            )
            if torch.distributed.is_initialized() and _world > 1:
                torch.distributed.all_reduce(_vec)
            _total = _vec[-1].item()
            if _total > 0:
                for _i, _k in enumerate(_pdms_keys):
                    metrics[_k] = _vec[_i].item() / _total

    model.train()
    return metrics


# ============================================================
# Checkpoint resume helper
# ============================================================


def try_resume(accelerator, args, tcfg) -> int:
    """
    Resume from checkpoint if --resume_from is specified.

    Looks for ``training_state.json`` inside the checkpoint dir to
    restore ``global_step`` and ``epoch``.  Returns the step to resume from.
    """
    if args.resume_from is None:
        return 0

    ckpt_path = Path(args.resume_from)
    if not ckpt_path.exists():
        logger.warning(f"Checkpoint path {ckpt_path} does not exist, training from scratch")
        return 0

    logger.info(f"Resuming from {ckpt_path}", main_process_only=True)
    # Fail fast with a clear, actionable message if this checkpoint is missing
    # any rank's ZeRO optimizer shard (e.g. an old multi-node run scattered its
    # shards across sibling <timestamp> dirs). Otherwise DeepSpeed raises a
    # cryptic FileNotFoundError deep inside load_checkpoint.
    ws = _expected_world_size()
    if ws and not _checkpoint_is_complete(ckpt_path, ws):
        present = sorted(_zero_shard_ranks(ckpt_path))
        missing = sorted(set(range(ws)) - set(present))
        raise RuntimeError(
            f"Refusing to resume: checkpoint {ckpt_path} is INCOMPLETE -- found ZeRO "
            f"optimizer shards for ranks {present} but this {ws}-rank job needs "
            f"0..{ws - 1} (missing {missing}). An earlier multi-node run likely "
            f"scattered its shards across sibling <timestamp> dirs. Point resume_path "
            f"at a COMPLETE checkpoint (checkpoint-<N>/pytorch_model must contain "
            f"bf16_zero_pp_rank_0..{ws - 1}_mp_rank_00_optim_states.pt) or use an "
            f"earlier intact checkpoint."
        )
    accelerator.load_state(str(ckpt_path))

    # Try to read saved training state
    state_file = ckpt_path / "training_state.json"
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        return state.get("global_step", 0)
    return 0


def _save_model_configs(checkpoint_dir: Path, model, processor):
    """Save model config, tokenizer and processor files into checkpoint dir.

    Mirrors what HuggingFace Trainer saves so the checkpoint is self-contained
    and can be loaded with ``from_pretrained`` without the original model path.
    """
    # Unwrap DDP / FSDP / DeepSpeed wrappers
    hf_model = model
    while hasattr(hf_model, "module"):
        hf_model = hf_model.module
    # Save underlying HF model config (config.json, generation_config.json)
    backbone = getattr(hf_model, "qwen_model", hf_model)
    if hasattr(backbone, "config"):
        backbone.config.save_pretrained(checkpoint_dir)
    if hasattr(backbone, "generation_config"):
        backbone.generation_config.save_pretrained(checkpoint_dir)
    # Save processor (preprocessor_config.json + chat_template.json)
    if processor is not None:
        processor.save_pretrained(checkpoint_dir)
    # Save tokenizer (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, etc.)
    if processor is not None and hasattr(processor, "tokenizer"):
        processor.tokenizer.save_pretrained(checkpoint_dir)


def save_checkpoint(accelerator, tcfg, global_step: int, tag: str = "",
                    ema: Optional[EMA] = None, model=None, processor=None,
                    save_optimizer: bool = True):
    """Save checkpoint with training state metadata and optional EMA state.

    Args:
        save_optimizer: when False, only save model weights (much faster, no
            optimizer / scheduler / sampler / rng state).  Use this for
            ``best`` and ``final`` tags that are only ever used for inference,
            not for resuming training.  Saves ~30s vs ~2min on shared storage
            for a 3B model with DeepSpeed ZeRO-2 optimizer state.
    """
    name = tag if tag else f"checkpoint-{global_step}"
    p = Path(tcfg["output_dir"]) / name
    if accelerator.is_main_process:
        p.mkdir(parents=True, exist_ok=True)
    if save_optimizer:
        # All ranks must participate in save_state (DeepSpeed shards optimizer across ranks)
        accelerator.save_state(str(p))
        # Additionally save standalone model weights so the checkpoint can
        # be loaded directly via `from_pretrained(<ckpt>/pytorch_model)`
        # without needing to convert DeepSpeed ZeRO shards. Cheap (~6 GB bf16
        # vs ~36 GB optimizer state) and matches what `best` saves.
        if model is not None:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, str(p / "pytorch_model"))
    else:
        # Lightweight save: model weights only (~6 GB bf16 vs ~36 GB full state).
        # All ranks must call save_model so DeepSpeed can coordinate even though
        # ZeRO-2 keeps bf16 weights replicated (only optimizer state is sharded).
        accelerator.wait_for_everyone()
        if model is None:
            raise ValueError("save_checkpoint(save_optimizer=False) requires `model`.")
        accelerator.save_model(model, str(p / "pytorch_model"))
    if not accelerator.is_main_process:
        return
    # Save extra training state for resume
    with open(p / "training_state.json", "w") as f:
        json.dump({"global_step": global_step}, f)
    # Save EMA state
    if ema is not None:
        torch.save(ema.state_dict(), p / "ema_state.pt")
    # Copy config yaml into checkpoint dir for reproducibility
    src_cfg = Path(tcfg["output_dir"]) / "config.yaml"
    if src_cfg.exists():
        shutil.copy2(src_cfg, p / "config.yaml")
    # Save model config, tokenizer & processor for self-contained checkpoint
    if model is not None:
        _save_model_configs(p, model, processor)
        # Also drop the same config files into the pytorch_model/ subdir so
        # that subdir alone is a complete `from_pretrained`-loadable bundle.
        pm_dir = p / "pytorch_model"
        if pm_dir.exists():
            _save_model_configs(pm_dir, model, processor)
    logger.info(f"Saved checkpoint: {p}", main_process_only=True)


def _expected_world_size() -> Optional[int]:
    """Total rank count for this job, from the launcher env (accelerate/torchrun
    set ``WORLD_SIZE``). Used to tell whether a DeepSpeed checkpoint has a
    COMPLETE set of per-rank ZeRO shards. Returns None if unknown.
    """
    for key in ("WORLD_SIZE", "NUM_PROCESSES"):
        v = os.environ.get(key)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return None


def _zero_shard_ranks(ckpt_dir) -> set:
    """Set of rank indices whose ZeRO optimizer shard is present in
    ``<ckpt_dir>/pytorch_model/``.
    """
    pm = Path(ckpt_dir) / "pytorch_model"
    ranks = set()
    if pm.is_dir():
        for f in pm.glob("bf16_zero_pp_rank_*_mp_rank_00_optim_states.pt"):
            try:
                ranks.add(int(f.name[len("bf16_zero_pp_rank_"):].split("_", 1)[0]))
            except (ValueError, IndexError):
                pass
    return ranks


def _checkpoint_is_complete(ckpt_dir, world_size: Optional[int] = None) -> bool:
    """True if ``ckpt_dir`` is a fully-loadable periodic DeepSpeed checkpoint.

    A ZeRO checkpoint loads only when ``pytorch_model/`` holds the model-states
    file AND every rank's optimizer shard. Multi-node runs launched by older/
    buggy code could scatter shards across sibling ``<timestamp>`` dirs (each
    rank wrote to a different ``output_dir``), leaving a checkpoint missing some
    ``bf16_zero_pp_rank_<r>_...`` shard -> a cryptic ``FileNotFoundError`` at
    resume. When ``world_size`` is known (WORLD_SIZE under accelerate launch)
    require ranks ``0..world_size-1``; otherwise best-effort (model states +
    rank 0 present).
    """
    pm = Path(ckpt_dir) / "pytorch_model"
    if not (pm / "mp_rank_00_model_states.pt").exists():
        return False
    ranks = _zero_shard_ranks(ckpt_dir)
    if not ranks:
        return False
    if world_size and world_size > 0:
        return set(range(world_size)).issubset(ranks)
    return 0 in ranks


def _max_checkpoint_in(run_dir, world_size: Optional[int] = None) -> Optional[Path]:
    """Return the ``checkpoint-<N>`` dir with the largest N directly under
    ``run_dir`` (e.g. a single ``<output_dir>/<timestamp>`` run directory). When
    ``world_size`` is given, checkpoints missing any rank's ZeRO shard are
    skipped so a scattered/incomplete checkpoint is never selected. Returns None
    if run_dir doesn't exist / has no usable checkpoint-<N> subdir.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best_path = None
    best_step = -1
    for ckpt_dir in run_dir.glob("checkpoint-*"):
        if not ckpt_dir.is_dir():
            continue
        suffix = ckpt_dir.name[len("checkpoint-"):]
        if not suffix.isdigit() or int(suffix) <= best_step:
            continue
        if world_size is not None and not _checkpoint_is_complete(ckpt_dir, world_size):
            continue
        best_step = int(suffix)
        best_path = ckpt_dir
    return best_path


def _find_latest_checkpoint(output_dir_root, world_size: Optional[int] = None) -> Optional[str]:
    """Scan every ``<output_dir_root>/<timestamp>/checkpoint-<N>`` dir and return
    the absolute path of the COMPLETE one with the largest N (most training
    progress). Ties are broken by the newest timestamp sub-directory. Incomplete
    (scattered) checkpoints are skipped when ``world_size`` is known. Returns
    None if no usable checkpoint is found.
    """
    root = Path(output_dir_root)
    if not root.exists():
        return None
    best_path = None
    best_key = None
    for ts_dir in sorted(root.iterdir()):
        if not ts_dir.is_dir():
            continue
        ckpt = _max_checkpoint_in(ts_dir, world_size)
        if ckpt is None:
            continue
        key = (int(ckpt.name[len("checkpoint-"):]), ts_dir.name)
        if best_key is None or key > best_key:
            best_key = key
            best_path = ckpt
    return str(best_path.resolve()) if best_path is not None else None


def _resolve_resume_path(resume_path, world_size: Optional[int] = None) -> Optional[str]:
    """Accept ``resume_path`` as either an exact checkpoint dir
    (``.../checkpoint-<N>``) or a run/timestamp dir (``.../<timestamp>``) -- in
    the latter case resolve it to the newest COMPLETE checkpoint-<N> inside it
    (scattered/incomplete checkpoints are skipped when ``world_size`` is known).
    This lets the config simply point at a run's timestamp directory and always
    continue from wherever that run got to, without editing resume_path after
    every new checkpoint. An exact ``checkpoint-<N>`` is returned as-is (the
    explicit choice is respected; completeness is validated later in
    ``try_resume`` with a clear error). Returns an absolute path string, or None
    if resume_path is empty/missing or has no usable checkpoint.
    """
    if not resume_path:
        return None
    p = Path(resume_path)
    if not p.exists():
        return None
    if p.name.startswith("checkpoint-"):
        return str(p.resolve())
    ckpt = _max_checkpoint_in(p, world_size)
    return str(ckpt.resolve()) if ckpt is not None else None


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    cot_cfg = cfg.get("cot", {})

    # Resolve ${training.run_name} in output_dir if present
    if "run_name" in tcfg and "${training.run_name}" in tcfg.get("output_dir", ""):
        tcfg["output_dir"] = tcfg["output_dir"].replace("${training.run_name}", tcfg["run_name"])

    # ---- resume switch (config-driven) ----
    # The YAML ``resume`` section is a convenience switch mirrored onto the CLI
    # ``--resume_from`` flag, so runs launched purely from a config (e.g. the
    # deepspeed launch script, which passes no CLI resume flag) can resume
    # without editing the command line:
    #   resume:
    #     enabled: false       # default -> train from scratch (model.qwen_model_name)
    #     resume_path: null    # EITHER an exact checkpoint dir (.../checkpoint-<N>) OR a
    #                          # run/timestamp dir (.../<timestamp>) -- the latter always
    #                          # resumes from whichever checkpoint-<N> inside it has the
    #                          # largest N, so you can just point this at a run's date/time
    #                          # folder once and keep re-running without editing it again;
    #                          # null/empty -> auto-resume from the LATEST checkpoint found
    #                          # anywhere under output_dir (across all timestamp runs), so
    #                          # simply re-running the same launch script after a crash
    #                          # continues training instead of starting a brand-new
    #                          # timestamped run directory.
    # An explicit CLI ``--resume_from`` always wins over the config. If enabled
    # but no checkpoint (explicit or auto-detected) can be found, fall back to a
    # fresh start from model.qwen_model_name instead of erroring out.
    resume_cfg = cfg.get("resume", {}) or {}
    if args.resume_from is None and resume_cfg.get("enabled", False):
        resume_path = resume_cfg.get("resume_path")
        # Expand config interpolations that plain yaml.safe_load leaves literal:
        #   ${training.output_dir} -> the (already run_name-expanded) output_dir root
        #   ${training.run_name}   -> the run name
        # so a config can point resume_path at e.g. "${training.output_dir}/20260715_163601".
        if isinstance(resume_path, str):
            resume_path = resume_path.replace("${training.output_dir}", tcfg["output_dir"])
            resume_path = resume_path.replace("${training.run_name}", tcfg.get("run_name", ""))
        world_size = _expected_world_size()
        resolved_path = _resolve_resume_path(resume_path, world_size)
        if resolved_path:
            args.resume_from = resolved_path
            print(f"[resume] enabled via config -> resume_from={args.resume_from}", flush=True)
        else:
            if resume_path:
                print(
                    f"[resume] configured resume_path ({resume_path}) has no usable checkpoint; "
                    f"searching for the latest checkpoint under output_dir instead",
                    flush=True,
                )
            auto_path = _find_latest_checkpoint(tcfg["output_dir"], world_size)
            if auto_path:
                args.resume_from = auto_path
                print(f"[resume] auto-detected latest checkpoint -> resume_from={args.resume_from}", flush=True)
            else:
                print(
                    f"[resume] resume.enabled=true but no checkpoint found (resume_path="
                    f"{resume_path!r}, output_dir={tcfg['output_dir']!r}); starting fresh "
                    f"from model.qwen_model_name",
                    flush=True,
                )

    # --- Multiprocess-safe output directory creation ---
    output_dir_file = None
    if args.resume_from:
        resume_dir = Path(args.resume_from).resolve().parent
        tcfg["output_dir"] = str(resume_dir)
    else:
        # Only the main process creates and records the timestamped directory; other processes read it
        out_dir_root = Path(tcfg["output_dir"]).resolve()
        output_dir_file = out_dir_root / "output_dir.txt"

        # We cannot rely on `accelerator` here (not created yet). Detect the
        # SINGLE global main from the launcher's RANK env var (global rank across
        # ALL nodes). LOCAL_RANK would be wrong for multi-node: every node has a
        # local rank 0, so each node would mint its own timestamp dir and ranks
        # would scatter their DeepSpeed shards across several <timestamp> dirs.
        try:
            global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            global_rank = 0
        is_main_env = (global_rank == 0)

        if is_main_env:
            out_dir_root.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            real_out_dir = out_dir_root / timestamp
            real_out_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir_file, "w") as f:
                f.write(str(real_out_dir))
            tcfg["output_dir"] = str(real_out_dir)
        else:
            # Non-main ranks wait for the main process to publish the timestamp
            # dir via the shared-FS marker. Best-effort: if it is not visible in
            # time (e.g. cross-node NFS attribute-cache lag) fall back to the root
            # and let the post-Accelerator broadcast below correct it. Never
            # hard-fail here -- that would needlessly abort a healthy job.
            import time
            published = None
            for _ in range(600):
                if output_dir_file.exists():
                    try:
                        published = output_dir_file.read_text().strip()
                    except OSError:
                        published = None
                    if published:
                        break
                time.sleep(0.1)
            if published:
                tcfg["output_dir"] = published
            else:
                print(
                    f"[output_dir] marker {output_dir_file} not visible yet; using root "
                    f"{out_dir_root} (will be fixed by broadcast)",
                    flush=True,
                )
                tcfg["output_dir"] = str(out_dir_root)

    # ---- logging backends  (EasyR1 style: logger: ["console", "wandb"]) ----
    log_backends = tcfg.get("logger", ["console"])

    grad_accum_steps = tcfg.get("gradient_accumulation_steps", 1)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=tcfg["mixed_precision"],
        gradient_accumulation_steps=grad_accum_steps,
        project_dir=tcfg["output_dir"],
        kwargs_handlers=[ddp_kwargs],
    )

    # ---- make ABSOLUTELY sure every rank agrees on output_dir/resume_from ----
    # The output_dir resolution above runs BEFORE any process group exists, so it
    # falls back to filesystem polling ("Multiprocess-safe output directory creation"), which is
    # best-effort and can race (e.g. a stale output_dir.txt left over from a
    # PREVIOUS run). If any rank ends up with a different output_dir/resume_from
    # than the main process, DeepSpeed silently shards a checkpoint save/load
    # across two different directories (one rank's optimizer state lands in the
    # wrong place, e.g. the launch script's CWD) -- causing a FileNotFoundError
    # the next time that checkpoint is loaded. Now that `accelerator` exists we
    # have a real process group, so broadcast the main process's values to every
    # other rank and overwrite whatever they computed independently.
    _sync_state = [tcfg["output_dir"], args.resume_from] if accelerator.is_main_process else [None, None]
    broadcast_object_list(_sync_state, from_process=0)
    tcfg["output_dir"], args.resume_from = _sync_state

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    set_seed(tcfg["seed"])
    acc_set_seed(tcfg["seed"])

    # ---- initialise multi-backend logger ----
    tlogger = None
    if accelerator.is_main_process and log_backends:
        tlogger = TrainingLogger(
            backends=log_backends,
            project=tcfg.get("wandb_project", "transfusion"),
            run_name=tcfg.get("run_name", None),
            config=cfg,
            output_dir=tcfg["output_dir"],
            token=tcfg.get("wandb_token", None),
        )

    # ---- save config yaml to output_dir for reproducibility ----
    if accelerator.is_main_process:
        out_dir = Path(tcfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        import yaml
        with open(out_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        # Also copy the original config file
        if args.config and Path(args.config).exists():
            shutil.copy2(args.config, out_dir / "config_original.yaml")

    # ---- model ----
    logger.info("Building model …", main_process_only=True)
    model = build_model(cfg)
    # ---- gradient checkpointing (trade ~25% speed for ~60% activation memory) ----
    # Single-pass forward is compatible with gradient checkpointing.
    # Actions attend to prefix through the Transfusion mask in the same pass.
    gc_enabled = tcfg.get("gradient_checkpointing", True)
    if gc_enabled:
        model.qwen_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("Gradient checkpointing enabled", main_process_only=True)

    pi = count_parameters(model)
    logger.info(
        f"Params: total={fmt(pi['total'])}  trainable={fmt(pi['trainable'])} "
        f"({pi['trainable_pct']:.1f}%)",
        main_process_only=True,
    )

    # ---- data ----
    logger.info("Loading data …", main_process_only=True)
    # Configure processor for pre-sampled video frames:
    # - Disable re-sampling (frames are already at correct fps)
    # - Suppress the fps warning (Qwen3.5 processor defaults to fps=24 for timestamps
    #   when pre-sampled frames lack metadata; this is cosmetic, training is unaffected)
    if hasattr(model.processor, "video_processor") and model.processor.video_processor is not None:
        model.processor.video_processor.do_sample_frames = False
        import transformers.models.qwen3_vl.processing_qwen3_vl as _qwen3_proc_mod
        _qwen3_proc_mod.logger.warning_once = lambda *a, **kw: None
    train_dl, val_dl = build_loaders(cfg, model.processor)

    # ---- optimizer & scheduler ----
    base_lr = tcfg["learning_rate"]
    backbone_lr_scale = tcfg.get("backbone_lr_scale", 1.0)
    groups = model.get_param_groups(
        tcfg["weight_decay"],
        backbone_lr_scale=backbone_lr_scale,
        base_lr=base_lr,
    )
    optimizer = torch.optim.AdamW(
        groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8
    )
    if backbone_lr_scale != 1.0:
        logger.info(
            f"  backbone_lr   = {base_lr * backbone_lr_scale:.2e} "
            f"(scale={backbone_lr_scale})",
            main_process_only=True,
        )
    steps_per_epoch = len(train_dl)
    total_steps = (steps_per_epoch * tcfg["num_epochs"]) // grad_accum_steps
    min_lr_ratio = tcfg.get("min_lr", 1.5e-5) / tcfg["learning_rate"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, tcfg["warmup_steps"], total_steps, min_lr_ratio
    )
    # Optional early stop: stop after this many optimizer
    # steps WITHOUT changing the LR schedule (total_steps still follows
    # num_epochs, so the reference run's LR curve at steps < max_steps is
    # preserved). 0/null disables the early stop.
    max_steps = tcfg.get("max_steps") or 0

    # ---- accelerate prepare ----
    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )
    if val_dl is not None:
        val_dl = accelerator.prepare(val_dl)

    # ---- resume from checkpoint ----
    resume_step = try_resume(accelerator, args, tcfg)

    # ---- logging banner ----
    eff_bs = tcfg["per_device_train_batch_size"] * accelerator.num_processes * grad_accum_steps
    logger.info("=" * 60, main_process_only=True)
    logger.info(f"  samples       = {len(train_dl.dataset)}", main_process_only=True)
    logger.info(f"  epochs        = {tcfg['num_epochs']}", main_process_only=True)
    if max_steps:
        logger.info(f"  max_steps     = {max_steps} (early stop, LR schedule unchanged)", main_process_only=True)
    logger.info(f"  effective bs  = {eff_bs}", main_process_only=True)
    logger.info(f"  total steps   = {total_steps}", main_process_only=True)
    logger.info(f"  resume step   = {resume_step}", main_process_only=True)
    logger.info(f"  traj_token    = {cfg['trajectory']['traj_token']}", main_process_only=True)
    logger.info(f"  action_dim    = {cfg['trajectory']['action_dim']}", main_process_only=True)
    logger.info(f"  diffusion     = flow_matching (x0)", main_process_only=True)
    logger.info(f"  λ_diff        = {cfg['diffusion']['lambda_diffusion']}", main_process_only=True)
    logger.info(f"  CoT           = {cot_cfg.get('enable_cot', False)}", main_process_only=True)
    logger.info(f"  action_norm   = {cfg.get('normalization', {}).get('action_norm', True)}", main_process_only=True)
    logger.info(f"  action_repr   = {cfg['trajectory'].get('action_repr', 'absolute')}", main_process_only=True)
    logger.info(f"  grad_accum    = {grad_accum_steps}", main_process_only=True)
    logger.info(f"  output_dir    = {tcfg['output_dir']}", main_process_only=True)
    logger.info(f"  loggers       = {log_backends}", main_process_only=True)
    logger.info("=" * 60, main_process_only=True)

    # ---- EMA (Exponential Moving Average) ----
    ema_enable = tcfg.get("ema_enable", False)
    ema_decay = tcfg.get("ema_decay", 0.0)
    ema_update_every = tcfg.get("ema_update_every", 10)
    ema = None
    if ema_enable and ema_decay > 0:
        # EMA operates on the unwrapped model (not the DDP/DeepSpeed wrapper)
        unwrapped = accelerator.unwrap_model(model)
        ema = EMA(unwrapped, decay=ema_decay, update_every=ema_update_every)
        # Try to restore EMA state from checkpoint
        if args.resume_from:
            ema_path = Path(args.resume_from) / "ema_state.pt"
            if ema_path.exists():
                ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
                logger.info(f"  EMA restored from {ema_path}", main_process_only=True)
        logger.info(f"  EMA enabled   = True (decay={ema_decay}, update_every={ema_update_every})", main_process_only=True)
    else:
        logger.info(f"  EMA enabled   = False", main_process_only=True)

    # ---- train loop  (openpi style: forward → backward → clip → step) ----
    global_step = resume_step
    # Best metric selection: "L2" (lower=better) or "PDMS" (higher=better)
    best_metric_name = tcfg.get("best_metric", "L2")
    best_metric_higher_is_better = best_metric_name.upper() in ("PDMS",)
    best_val = float("-inf") if best_metric_higher_is_better else float("inf")
    # Checkpoint rotation: keep only N most recent periodic checkpoints
    max_checkpoints = tcfg.get("max_checkpoints", 3)
    saved_checkpoints: list[Path] = []  # oldest first
    first_step_logged = (resume_step > 0)
    enable_cot = cot_cfg.get("enable_cot", False)
    # PDMS metric cache paths
    # metric_cache_test: used during evaluation (val set) — always compute PDMS
    # metric_cache_train: used for training-set PDMS stats (optional, for experiments)
    eval_metric_cache_path = cfg.get("data", {}).get("metric_cache_test")
    if eval_metric_cache_path and not Path(eval_metric_cache_path).exists():
        logger.info(f"Test metric cache not found ({eval_metric_cache_path}), PDMS will be skipped during eval.", main_process_only=True)
        eval_metric_cache_path = None

    # ---- resume: map global_step back to (epoch, in-epoch batch offset) ----
    # global_step counts OPTIMIZER steps. On a fresh run both values are 0 and
    # the loop below is unchanged. On resume we start at the epoch that owns
    # resume_step and fast-forward the dataloader to the exact batch, so we
    # neither re-see already-trained data nor overshoot the LR schedule.
    opt_steps_per_epoch = (len(train_dl) + grad_accum_steps - 1) // grad_accum_steps
    start_epoch = 0
    resume_into_epoch = 0
    if resume_step > 0 and opt_steps_per_epoch > 0:
        start_epoch = resume_step // opt_steps_per_epoch
        resume_into_epoch = resume_step % opt_steps_per_epoch
        logger.info(
            f"  resume from   = epoch {start_epoch}, +{resume_into_epoch} opt-steps",
            main_process_only=True,
        )

    for epoch in range(start_epoch, tcfg["num_epochs"]):
        if max_steps and global_step >= max_steps:
            logger.info(
                f"  max_steps={max_steps} reached before epoch {epoch + 1} — stopping",
                main_process_only=True,
            )
            break
        model.train()
        num_opt_steps = (len(train_dl) + grad_accum_steps - 1) // grad_accum_steps
        # On the first resumed epoch, skip the batches already consumed before
        # the checkpoint (no-op for a fresh run where resume_into_epoch == 0).
        epoch_dl = train_dl
        pbar_initial = 0
        if epoch == start_epoch and resume_into_epoch > 0:
            skip_batches = resume_into_epoch * grad_accum_steps
            if hasattr(accelerator, "skip_first_batches"):
                epoch_dl = accelerator.skip_first_batches(train_dl, skip_batches)
            else:
                from accelerate import skip_first_batches as _skip_first_batches
                epoch_dl = _skip_first_batches(train_dl, skip_batches)
            pbar_initial = resume_into_epoch
            logger.info(
                f"  fast-forwarded {skip_batches} batches into epoch {start_epoch}",
                main_process_only=True,
            )
        pbar = tqdm(
            total=num_opt_steps,
            initial=pbar_initial,
            desc=f"Epoch {epoch+1}/{tcfg['num_epochs']}",
            disable=not accelerator.is_local_main_process,
        )
        for batch in epoch_dl:
            with accelerator.accumulate(model):
                # ---- forward ----
                out = model(**batch)
                loss = out.loss

                # ---- NaN detection: skip step if loss is NaN ----
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"[step {global_step+1}] NaN/Inf loss detected, skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                # ---- backward ----
                accelerator.backward(loss)

                # ---- gradient clipping ----
                grad_norm_val = 0.0
                grad_norm_per_group = {}
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                    if grad_norm is not None:
                        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

                    # Per-component gradient norms for diagnosing large grad_norm.
                    # This is cheap (one reduction per group) and logged every logging_steps.
                    unwrapped = accelerator.unwrap_model(model)
                    _group_params = {
                        "backbone": [],
                        "action_encoder": [],
                        "action_decoder": [],
                        "latent_proj": [],
                        "adanorm": [],
                        "other": [],
                    }
                    for name, p in unwrapped.named_parameters():
                        if p.grad is None:
                            continue
                        if "qwen_model" in name and "ada_dense" not in name:
                            _group_params["backbone"].append(p.grad)
                        elif "action_encoder" in name:
                            _group_params["action_encoder"].append(p.grad)
                        elif "action_decoder" in name:
                            _group_params["action_decoder"].append(p.grad)
                        elif "latent_action_proj" in name:
                            _group_params["latent_proj"].append(p.grad)
                        elif "ada_dense" in name or "time_embed" in name or "adanorm" in name:
                            _group_params["adanorm"].append(p.grad)
                        else:
                            _group_params["other"].append(p.grad)
                    for gname, grads in _group_params.items():
                        if grads:
                            total_norm = torch.norm(
                                torch.stack([g.detach().norm(2) for g in grads]), 2
                            ).item()
                            grad_norm_per_group[gname] = total_norm

                # ---- optimizer step ----
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # ---- EMA update (after optimizer step) ----
                if ema is not None:
                    ema.update()

            # Only count a step when gradients are actually synced
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            pbar.update(1)

            # ---- log training data samples on first step ----
            if not first_step_logged and tlogger is not None and accelerator.is_main_process:
                log_data_samples(
                    tlogger, batch, model.module.processor
                    if hasattr(model, "module") else model.processor,
                    step=global_step, num_samples=3, prefix="train_data",
                )
                first_step_logged = True

            diff_v = out.diffusion_loss.item() if out.diffusion_loss is not None else 0
            lm_v = out.lm_loss.item() if out.lm_loss is not None else 0
            cur_lr = scheduler.get_last_lr()[0]

            # ---- per-dimension loss ----
            pdl = out.per_dim_loss or {}

            # ---- compute prompt token lengths for diagnostics ----
            attn = batch.get("attention_mask")
            if attn is not None:
                prompt_lens = attn.sum(dim=-1).float()  # [B]
                prompt_len_mean = prompt_lens.mean().item()
                prompt_len_max = prompt_lens.max().item()
                prompt_len_min = prompt_lens.min().item()
            else:
                seq_len = float(batch["input_ids"].shape[1])
                prompt_len_mean = prompt_len_max = prompt_len_min = seq_len

            # ---- compute CoT (output) token lengths for diagnostics ----
            # cot_labels is RIGHT-padded with -100; non-(-100) entries are the
            # real CoT target tokens the model must emit. This is the OUTPUT
            # length (complements the input prompt_len above) and lets us see
            # how much short_prediction shrinks the generation budget.
            cot_labels_b = batch.get("cot_labels")
            if cot_labels_b is not None:
                cot_lens = (cot_labels_b != -100).sum(dim=-1).float()  # [B]
                cot_len_mean = cot_lens.mean().item()
                cot_len_max = cot_lens.max().item()
                cot_len_min = cot_lens.min().item()
            else:
                cot_len_mean = cot_len_max = cot_len_min = 0.0

            # ---- compute L2 metric for this batch ----
            # Only compute if both pred and GT are available
            l2_val = None
            if hasattr(out, "noise_pred") and out.noise_pred is not None and "actions" in batch and batch["actions"] is not None:
                # Try to get predicted actions (denoised output)
                # If model returns denoised actions, use them; else skip
                # Here, out.noise_pred is usually the predicted noise, not the denoised action
                # So we skip unless model returns denoised actions in debug_info or elsewhere
                # If debug_info contains 'denoised_actions', use it
                denoised = None
                if out.debug_info and "denoised_actions" in out.debug_info:
                    denoised = out.debug_info["denoised_actions"]
                # Fallback: if noise_pred is actually denoised actions (project-specific)
                if denoised is None and out.noise_pred is not None and out.noise_pred.shape == batch["actions"].shape:
                    denoised = out.noise_pred
                if denoised is not None:
                    if not torch.is_tensor(denoised):
                        denoised = torch.as_tensor(
                            denoised,
                            dtype=batch["actions"].dtype,
                            device=batch["actions"].device,
                        )
                    from model.utils import compute_L2
                    if denoised.shape == batch["actions"].shape:
                        l2_val = compute_L2(
                            denoised.detach().cpu(),
                            batch["actions"].detach().cpu(),
                        ).mean().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                diff=f"{diff_v:.4f}",
                lm=f"{lm_v:.4f}",
                lr=f"{cur_lr:.2e}",
            )

            # ---- periodic logging ----
            if global_step % tcfg["logging_steps"] == 0 and tlogger is not None and accelerator.is_main_process:
                log_data = {
                    "train/loss": loss.item(),
                    "train/diff_loss": diff_v,
                    "train/lm_loss": lm_v,
                    "train/lr": cur_lr,
                    "train/grad_norm": grad_norm_val,
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                    "train/prompt_len_mean": prompt_len_mean,
                    "train/prompt_len_max": prompt_len_max,
                    "train/prompt_len_min": prompt_len_min,
                    "train/cot_len_mean": cot_len_mean,
                    "train/cot_len_max": cot_len_max,
                    "train/cot_len_min": cot_len_min,
                }
                # Per-dimension loss
                for dim_name, dim_val in pdl.items():
                    log_data[f"train/diff_loss_{dim_name}"] = dim_val
                if l2_val is not None:
                    log_data["train/L2"] = l2_val
                # Log sampled timestep so user knows at which t the L2 was computed.
                # train/L2 is at a RANDOM t and NOT comparable with test L2 (full ODE from t=1).
                if out.debug_info and "t" in out.debug_info:
                    log_data["train/sampled_t"] = out.debug_info["t"]
                # Per-component gradient norms (pre-clipping)
                for gname, gnorm in grad_norm_per_group.items():
                    log_data[f"train/grad_norm_{gname}"] = gnorm
                tlogger.log(log_data, step=global_step)

                # ---- Debug: print first sample details ----
                dbg = out.debug_info
                if dbg:
                    def _fmt(arr):
                        def _round_nested(x):
                            if torch.is_tensor(x):
                                x = x.detach().cpu().tolist()
                            if isinstance(x, (list, tuple)):
                                return [_round_nested(v) for v in x]
                            if isinstance(x, (int, float)):
                                return round(float(x), 4)
                            return x

                        rounded = _round_nested(arr)
                        if isinstance(rounded, list) and rounded and isinstance(rounded[0], list):
                            return "\n      ".join([str(row) for row in rounded])
                        return str(rounded)
                    logger.info(f"[DEBUG step {global_step}] mode={dbg.get('mode')} t={dbg.get('t')}", main_process_only=True)
                    for k, v in dbg.items():
                        if k in ("mode", "t"):
                            continue
                        logger.info(f"  {k}: {_fmt(v)}", main_process_only=True)

            # ---- evaluation ----
            if val_dl and tcfg.get("eval_steps") and global_step % tcfg["eval_steps"] == 0:
                # Swap to EMA weights for evaluation
                if ema is not None:
                    ema.apply_shadow()
                processor = model.module.processor if hasattr(model, "module") else model.processor
                vm = evaluate(
                    model, val_dl, accelerator, tlogger, processor,
                    step=global_step, enable_cot=enable_cot,
                    metric_cache_path=eval_metric_cache_path,
                )
                # Restore training weights
                if ema is not None:
                    ema.restore()
                logger.info(f"Step {global_step} val: {vm}", main_process_only=True)
                if tlogger is not None and accelerator.is_main_process:
                    tlogger.log({f"val/{k}": v for k, v in vm.items()}, step=global_step)

                # Decide whether to save "best" on rank 0, then broadcast the
                # decision so ALL ranks call save_checkpoint together.  The
                # save itself is a collective op (DeepSpeed gathers sharded
                # state); calling it only on rank 0 would deadlock against
                # the next periodic save_state collective.  ``vm`` differs
                # per rank because evaluate() doesn't gather metrics, so we
                # rely solely on rank 0's view.
                do_save_best = torch.zeros(1, device=accelerator.device)
                cur_metric = vm.get(best_metric_name, vm.get("val_loss", float("inf")))
                if accelerator.is_main_process:
                    is_better = (cur_metric > best_val) if best_metric_higher_is_better else (cur_metric < best_val)
                    if is_better:
                        do_save_best.fill_(1.0)
                        best_val = cur_metric
                        logger.info(f"New best {best_metric_name}={cur_metric:.4f}", main_process_only=True)
                if torch.distributed.is_initialized():
                    torch.distributed.broadcast(do_save_best, src=0)
                if do_save_best.item() > 0.5:
                    save_checkpoint(accelerator, tcfg, global_step, tag="best", ema=ema,
                                    model=accelerator.unwrap_model(model),
                                    processor=accelerator.unwrap_model(model).processor,
                                    save_optimizer=False)

            # ---- periodic save ----
            if tcfg.get("save_steps") and global_step % tcfg["save_steps"] == 0:
                save_opt = tcfg.get("save_optimizer", False)
                save_checkpoint(accelerator, tcfg, global_step, ema=ema,
                                model=accelerator.unwrap_model(model),
                                processor=accelerator.unwrap_model(model).processor,
                                save_optimizer=save_opt)
                ckpt_path = Path(tcfg["output_dir"]) / f"checkpoint-{global_step}"
                saved_checkpoints.append(ckpt_path)
                # Rotate: remove oldest checkpoints beyond max_checkpoints
                while max_checkpoints > 0 and len(saved_checkpoints) > max_checkpoints:
                    old_ckpt = saved_checkpoints.pop(0)
                    if accelerator.is_main_process and old_ckpt.exists():
                        import shutil as _shutil
                        _shutil.rmtree(old_ckpt, ignore_errors=True)
                        logger.info(f"Removed old checkpoint: {old_ckpt}", main_process_only=True)

            # ---- early stop (training.max_steps, after eval/save of this step) ----
            if max_steps and global_step >= max_steps:
                logger.info(
                    f"  max_steps={max_steps} reached — stopping",
                    main_process_only=True,
                )
                break

        pbar.close()

    # ---- final save ----
    save_checkpoint(accelerator, tcfg, global_step, tag="final", ema=ema,
                    model=accelerator.unwrap_model(model),
                    processor=accelerator.unwrap_model(model).processor,
                    save_optimizer=False)
    if tlogger is not None:
        tlogger.finish()
    logger.info("Training complete.", main_process_only=True)


if __name__ == "__main__":
    main()
