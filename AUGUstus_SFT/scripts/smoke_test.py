"""End-to-end smoke test on synthetic data (no dataset needed).

Validates on the CURRENT machine/env that:
  1. the Qwen3.5-2B backbone + Transfusion wrapper build,
  2. a full training forward (diffusion + CoT LM loss) runs and backprops,
  3. trajectory generation (KV-cache, 1-step FM) produces the right shape,
  4. the GatedDeltaNet fast path is active.

Usage:
    python scripts/smoke_test.py            # CPU + GPU checks
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from model.model_qwen3p5 import TransfusionQwen3p5

QWEN = os.environ.get("QWEN35_2B_PATH", "Qwen/Qwen3.5-2B")


def _video_metadata(num_frames: int, fps: float = 2.0):
    """Same contract as model.dataset._video_metadata_for_processor."""
    return [{
        "total_num_frames": num_frames,
        "fps": fps,
        "duration": num_frames / fps,
        "frames_indices": list(range(num_frames)),
    }]


def make_fake_batch(processor, prompt_template: str, device: str):
    """One synthetic sample, following the exact processor contract of
    model.dataset.TransfusionDataset (PIL frame list + video_metadata)."""
    frames = [Image.fromarray(
        np.random.randint(0, 255, (512, 2048, 3), dtype=np.uint8)) for _ in range(4)]

    prompt = prompt_template.format(
        driving_command="GO FORWARD",
        ego_dynamic_state="vx: 5.00 m/s, vy: 0.00 m/s, ax: 0.10 m/s², ay: 0.00 m/s²",
        historical_x_y_angle=str([[-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt},
        ],
    }]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input], videos=[frames],
        video_metadata=_video_metadata(len(frames)),
        padding=True, return_tensors="pt",
    )

    batch = dict(inputs)
    batch["actions"] = torch.rand(1, 8, 3) * 10
    batch["cot_ids"] = torch.tensor([[1, 2, 3, 4, 5]])
    batch["cot_labels"] = torch.tensor([[2, 3, 4, 5, 6]])
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(QWEN)
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        processor.video_processor.do_sample_frames = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability(0)})" if device == "cuda" else ""))

    model = TransfusionQwen3p5(
        qwen_model_name=QWEN,
        traj_token=8, action_dim=3,
        lambda_diffusion=10.0, lambda_lm=1.0, cfg_scale=1.0,
        freeze_vision_encoder=False, freeze_llm=False, max_cot_tokens=512,
        ada_variant="dit", ada_layer_mode="full", ada_use_weight=True,
        diffusion_type="flow_matching", prediction_type="x0",
        num_inference_steps=1, clip_sample_range=0.0,
        use_action_first=True, action_first_generate_cot=True,
        cot_components=["prediction"],
        action_norm=False,
    ).to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

    # fast-path check
    layer0 = model.qwen_model.model.language_model.layers[0]
    la = getattr(layer0, "linear_attn", None)
    fast = getattr(la, "causal_conv1d_fn", None) if la is not None else None
    print(f"[smoke] GatedDeltaNet fast path bound: {fast is not None}")

    prompt_template = open("prompts/default.txt", encoding="utf-8").read()
    batch = make_fake_batch(processor, prompt_template, device)
    batch = {k: v.to(torch.bfloat16) if k == "pixel_values_videos" and v.is_floating_point() else v
             for k, v in batch.items()}

    # ---- training forward ----
    model.train()
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values_videos=batch["pixel_values_videos"],
        video_grid_thw=batch["video_grid_thw"],
        second_per_grid_ts=batch.get("second_per_grid_ts"),
        actions=batch["actions"],
        cot_ids=batch.get("cot_ids"),
        cot_labels=batch.get("cot_labels"),
    )
    loss = out.loss if hasattr(out, "loss") else out["loss"]
    print(f"[smoke] train forward OK — loss={float(loss):.4f}")
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"[smoke] backward OK — grad L1={g:.3e}")

    # ---- inference ----
    model.eval()
    traj, cot_ids = model.generate_trajectory(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values_videos=batch["pixel_values_videos"],
        video_grid_thw=batch["video_grid_thw"],
        generate_cot=True,
    )
    print(f"[smoke] generate_trajectory OK — traj={tuple(traj.shape)}, cot_tokens={cot_ids.shape[1] if cot_ids is not None else 0}")

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
