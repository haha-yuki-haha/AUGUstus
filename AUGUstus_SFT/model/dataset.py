"""
Dataset for Transfusion trajectory prediction.

Each sample is a (text, image/video, actions) triplet:
    - text:    natural-language instruction / scene description
    - media:   path to an image or video file
    - actions: [traj_token, action_dim] ground-truth waypoints

action_dim = 2  →  (acceleration, angular_velocity)
action_dim = 3  →  (x, y, theta)

Supports two data sources:
1. JSON file (list of dicts):
   [{"id": "...", "text": "...", "media_path": "...", "media_type": "image", "actions": [...]}]

2. HuggingFace Dataset cache (Arrow format, created by build_hf_cache.py):
   Contains pre-loaded video tensors as binary, text prompts, and future trajectories.
   Set data_path to the cache directory (containing batch_0/, batch_1/, ...).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_video_frames(
    video_path: str, max_frames: int = 4, fps: int = 2
) -> List[Image.Image]:
    """Decode video into a list of PIL frames at the given fps."""
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    total_duration = float(stream.duration * stream.time_base)
    target_frames = min(int(total_duration * fps), max_frames)
    frame_interval = max(1, int(stream.average_rate / fps))

    frames: List[Image.Image] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % frame_interval == 0:
            frames.append(frame.to_image())
        if len(frames) >= target_frames:
            break
    container.close()

    assert len(frames) == target_frames, (
        f"Video frame count mismatch in {video_path}: "
        f"got {len(frames)}, expected {target_frames} "
        f"(fps={fps}, max_frames={max_frames}, duration={total_duration:.2f}s)"
    )
    return frames


def _video_metadata_for_processor(num_frames: int, fps: float) -> List[Dict[str, Any]]:
    """Build the ``video_metadata`` kwarg for ``processor(..., videos=[frames], ...)``.

    Qwen3VLProcessor (used by this Qwen3.5 checkpoint -- see processor_config.json
    's ``processor_class``) embeds a ``<X.X seconds>`` text token before each
    frame's vision tokens, computed from ``video_metadata.fps``. Without this,
    frames passed as an already-sampled list (no container/duration attached) get
    ``fps=None`` -> the processor silently assumes ``fps=24`` and warns
    ("Qwen3VL requires frame timestamps ... Defaulting to fps=24."). Our frames
    are sampled at ``fps`` (2 Hz for NAVSIM history, matching
    ``load_video_frames``'s / ``TransfusionDataset``'s own default), not 24, so
    the embedded timestamps would otherwise be wrong by ~12x. Passing this
    metadata only changes the prompt's timestamp TEXT -- verified it does NOT
    alter ``pixel_values_videos`` / ``video_grid_thw`` (frame sampling/content
    unaffected).
    """
    return [
        {
            "total_num_frames": num_frames,
            "fps": fps,
            "duration": num_frames / fps,
            "frames_indices": list(range(num_frames)),
        }
    ]


class TransfusionDataset(Dataset):
    """Dataset that loads (text, image/video, actions) for Transfusion."""

    def __init__(
        self,
        data_path: str,
        processor: Any,
        traj_token: int = 12,
        action_dim: int = 2,
        # Defaults match configs/navsim_qwen3p5.yaml (data.video_max_frames /
        # data.video_fps): 4 historical frames at 2 Hz = NAVSIM's 2s history window.
        video_max_frames: int = 4,
        video_fps: int = 2,
        is_training: bool = True,
    ):
        super().__init__()
        self.processor = processor
        self.traj_token = traj_token
        self.action_dim = action_dim
        self.video_max_frames = video_max_frames
        self.video_fps = video_fps
        self.is_training = is_training

        with open(data_path, "r", encoding="utf-8") as f:
            self.data: List[Dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _load_media(self, sample: Dict) -> Dict:
        media_type = sample.get("media_type", "image")
        path = sample["media_path"]
        if media_type == "video":
            frames = load_video_frames(path, self.video_max_frames, self.video_fps)
            return {"type": "video", "frames": frames}
        image = Image.open(path).convert("RGB")
        return {"type": "image", "image": image}

    def _build_messages(self, sample: Dict, media: Dict) -> List[Dict]:
        content: List[Dict] = []
        if media["type"] == "video":
            content.append({"type": "video", "video": media["frames"]})
        else:
            content.append({"type": "image", "image": media["image"]})
        text = sample.get("text", "Predict the trajectory.")
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _prepare_actions(self, sample: Dict) -> torch.Tensor:
        """Parse actions and validate shape matches (traj_token, action_dim)."""
        raw = np.array(sample["actions"], dtype=np.float32)

        assert raw.shape[0] == self.traj_token, (
            f"Action temporal length mismatch: got {raw.shape[0]}, expected {self.traj_token}"
        )
        assert raw.shape[1] == self.action_dim, (
            f"Action dimension mismatch: got {raw.shape[1]}, expected {self.action_dim}"
        )

        return torch.from_numpy(raw)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        media = self._load_media(sample)
        messages = self._build_messages(sample, media)

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if media["type"] == "video":
            inputs = self.processor(
                text=[text_input], videos=[media["frames"]],
                video_metadata=_video_metadata_for_processor(len(media["frames"]), self.video_fps),
                padding=True, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text_input], images=[media["image"]],
                padding=True, return_tensors="pt",
            )

        result = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        labels = result["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        result["labels"] = labels

        result["actions"] = self._prepare_actions(sample)
        return result


class TransfusionHFCacheDataset(Dataset):
    """Dataset that loads from HuggingFace Dataset cache (Arrow format).

    Expects cache directory with batch_0/, batch_1/, ... sub-directories,
    each created by datasets.Dataset.save_to_disk().

    Prompt is loaded from a separate txt file and filled at runtime with
    data fields: driving_command, ego_dynamic_state, historical_x_y_angle,
    historical_vx_vy.
    """

    def __init__(
        self,
        cache_dir: str,
        processor: Any,
        prompt_file: str,
        traj_token: int = 8,
        action_dim: int = 3,
        is_training: bool = True,
        enable_cot: bool = False,
        cot_components: Optional[List[str]] = None,
        # NAVSIM history videos are cached/decoded at 2 Hz, 4 frames (matches
        # configs/navsim_qwen3p5.yaml's data.video_fps / .video_max_frames,
        # and ``load_video_frames``'s / ``TransfusionDataset``'s own defaults).
        # Used ONLY to build the processor's ``video_metadata`` (correct
        # ``<X.X seconds>`` prompt timestamps); does not affect which/how many
        # frames are decoded (``_decode_video`` reads every frame in the cached
        # mp4 as-is).
        video_fps: float = 2.0,
    ):
        super().__init__()
        from datasets import concatenate_datasets, load_from_disk

        self.processor = processor
        self.traj_token = traj_token
        self.action_dim = action_dim
        self.is_training = is_training
        self.enable_cot = enable_cot
        self.video_fps = video_fps

        # CoT component configuration: an ordered list of components that are
        # concatenated (in the given order) to form the CoT text. Allowed values:
        #   "perception" / "prediction" / "planning" -> structured solution_*
        #        columns, each wrapped in <perception>...</perception> etc.
        #   "reasoning"                               -> free-form reasoning_trace
        #        column, wrapped in <reasoning>...</reasoning>.
        # The list may be empty (no CoT) or contain any subset in any order.
        self.cot_components = self._resolve_cot_components(cot_components)

        # Load prompt template from txt file
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        # Load all batches
        cache_path = Path(cache_dir)
        batch_dirs = sorted(cache_path.glob("batch_*"))
        if not batch_dirs:
            raise FileNotFoundError(f"No batch directories found in {cache_dir}")
        datasets = [load_from_disk(str(d)) for d in batch_dirs]
        self.hf_dataset = concatenate_datasets(datasets)

    # Mapping from a CoT component name to the cached row column that holds it.
    # All components are emitted wrapped in <name>...</name> tags, EXCEPT
    # "short_prediction" which is already a self-delimiting sequence of atomic
    # <|sp_...|> tokens and is emitted verbatim (no wrapper) to minimise tokens.
    # The config name "reasoning" reads from the original "reasoning_trace" column.
    COT_COMPONENT_COLUMNS = {
        "perception": "solution_perception",
        "prediction": "solution_prediction",
        "planning": "solution_planning",
        "reasoning": "reasoning_trace",
        # Compact prediction tokens built by 4_3_v2_short_prediction.py.
        "short_prediction": "solution_short_prediction",
    }
    # Default components when none are specified in the config.
    DEFAULT_COT_COMPONENTS = ["perception", "prediction", "planning"]

    def _resolve_cot_components(self, components: Optional[List[str]]) -> List[str]:
        """Normalize the requested CoT components, preserving their order.

        ``components`` is an ordered list; the CoT text is built by concatenating
        each component in the given order. An empty list means "no CoT". When
        ``components`` is None (key absent), the default P3 components are used.
        """
        if components is None:
            return list(self.DEFAULT_COT_COMPONENTS)
        resolved: List[str] = []
        for c in components:
            name = str(c).strip().lower()
            if not name:
                continue
            if name not in self.COT_COMPONENT_COLUMNS:
                raise ValueError(
                    f"Invalid cot_components entry {name!r}; "
                    f"allowed values are {list(self.COT_COMPONENT_COLUMNS)}"
                )
            resolved.append(name)
        # "short_prediction" uses dedicated atomic tokens and must be used alone.
        if "short_prediction" in resolved and len(resolved) != 1:
            raise ValueError(
                "cot_components: 'short_prediction' must be the ONLY component when used; "
                f"got {components!r}"
            )
        return resolved

    def _build_cot_text(self, row: Dict) -> str:
        """Build the CoT text for a row by concatenating the configured components.

        Components are emitted in the configured order, each wrapped in
        <comp>...</comp> tags (e.g. <reasoning>...</reasoning>).
        """
        parts = []
        for comp in self.cot_components:
            content = row.get(self.COT_COMPONENT_COLUMNS[comp])
            if not content:
                continue
            if comp == "short_prediction":
                # Already a self-delimiting <|sp_...|> token sequence -> no wrapper.
                parts.append(content)
            else:
                parts.append(f"<{comp}>{content}</{comp}>")
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _decode_video(self, video_bytes: bytes) -> List:
        """Decode video from raw mp4 bytes to list of PIL Image frames."""
        import av
        import io
        container = av.open(io.BytesIO(video_bytes))
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
        container.close()
        return frames

    @staticmethod
    def _fill_prompt_fields(template: str, row: Dict) -> str:
        """Fill a prompt template's data-field placeholders from a cached row."""
        return template.format(
            driving_command=row.get("driving_command", ""),
            ego_dynamic_state=row.get("ego_dynamic_state", ""),
            historical_x_y_angle=row.get("historical_x_y_angle", ""),
            historical_vx_vy=row.get("historical_vx_vy", ""),
        )

    def _build_prompt(self, row: Dict) -> str:
        """Fill prompt template with data fields from the cached row."""
        return self._fill_prompt_fields(self.prompt_template, row)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.hf_dataset[idx]

        # Decode video
        frames = self._decode_video(row["video"])

        # Build prompt from template + data fields
        prompt_text = self._build_prompt(row)
        content = [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt_text},
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        # Only pass ``enable_thinking`` when the chat template actually uses it
        # (Qwen3.5's template does; Qwen2.5-VL's does not). Passing an unknown
        # template kwarg makes transformers >= 5.x log the misleading
        # "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`"
        # warning on every sample. Cache the check per dataset instance.
        has_enable_thinking = getattr(self, "_has_enable_thinking", None)
        if has_enable_thinking is None:
            _tmpl = getattr(self.processor, "chat_template", None)
            if isinstance(_tmpl, dict):
                _tmpl = _tmpl.get("default", "")
            has_enable_thinking = "enable_thinking" in str(_tmpl or "")
            self._has_enable_thinking = has_enable_thinking
        _apply_kwargs = {"enable_thinking": False} if has_enable_thinking else {}
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **_apply_kwargs,
        )
        # Strip think tokens if present (Qwen3.5 template always adds them)
        text_input = text_input.replace("<think>\n\n</think>\n\n", "")

        inputs = self.processor(
            text=[text_input], videos=[frames],
            video_metadata=_video_metadata_for_processor(len(frames), self.video_fps),
            padding=True, return_tensors="pt",
        )
        result = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Save raw video frames for logging (list of PIL Images)
        result["_raw_video_frames"] = frames

        labels = result["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        result["labels"] = labels

        # Parse actions
        actions = np.array(json.loads(row["future_x_y_angle"]), dtype=np.float32)
        assert actions.shape[0] == self.traj_token, (
            f"Action temporal length mismatch: got {actions.shape[0]}, expected {self.traj_token}"
        )
        # The HF cache always stores (x, y, heading). When action_dim < the
        # cached width (e.g. nuScenes drops the unused heading -> action_dim=2),
        # keep only the leading action_dim channels; action_dim LARGER than the
        # cache is a real misconfiguration.
        assert actions.shape[1] >= self.action_dim, (
            f"Action dimension too small: cache stores {actions.shape[1]} channels, "
            f"config action_dim={self.action_dim}"
        )
        actions = np.ascontiguousarray(actions[:, :self.action_dim])

        result["actions"] = torch.from_numpy(actions)

        # Store metadata for evaluation / result saving
        result["_frame_id"] = row.get("frame_id", "")
        result["_scene_token"] = row.get("scene_token", "")
        result["_driving_command"] = row.get("driving_command", "")
        result["_ego_dynamic_state"] = row.get("ego_dynamic_state", "")
        result["_historical_x_y_angle"] = row.get("historical_x_y_angle", "")
        result["_historical_vx_vy"] = row.get("historical_vx_vy", "")

        # CoT: build CoT text (reasoning_trace or structured P3) and tokenize.
        cot_text = self._build_cot_text(row)
        if self.enable_cot and self.is_training:
            if cot_text and isinstance(cot_text, str):
                cot_token_ids = self.processor.tokenizer.encode(
                    cot_text, add_special_tokens=False,
                )
                cot_ids = torch.tensor(cot_token_ids, dtype=torch.long)
                result["cot_ids"] = cot_ids
                result["cot_labels"] = cot_ids.clone()

        # Always pass GT CoT text as string for evaluation output
        result["_reasoning_trace"] = cot_text

        return result


class TransfusionCollator:
    """Pad variable-length text sequences (LEFT padding) and stack fixed-length actions.

    LEFT padding (content right-aligned) is critical for Transfusion because
    action token position IDs are computed from the last position of input_ids.
    With right padding, the last position would be a pad token, causing incorrect
    position IDs for action tokens.
    """

    def __init__(self, processor: Any, traj_token: int = 12, action_dim: int = 2):
        self.pad_id = processor.tokenizer.pad_token_id or 0
        self.traj_token = traj_token
        self.action_dim = action_dim

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        actions = torch.stack([f.pop("actions") for f in features])

        # Extract COT ids/labels (variable length, need padding)
        cot_ids_list = [f.pop("cot_ids", None) for f in features]
        cot_labels_list = [f.pop("cot_labels", None) for f in features]

        max_len = max(f["input_ids"].shape[0] for f in features)

        batch: Dict[str, torch.Tensor] = {}
        for key in list(features[0].keys()):
            if key == "labels":
                # LEFT pad with -100 (ignore index)
                t = torch.full((len(features), max_len), -100, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = f[key].shape[0]
                    t[i, max_len - seq_len :] = f[key]
                batch[key] = t
            elif key == "input_ids":
                # LEFT pad with pad_id
                t = torch.full((len(features), max_len), self.pad_id, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = f[key].shape[0]
                    t[i, max_len - seq_len :] = f[key]
                batch[key] = t
            elif key == "attention_mask":
                # LEFT pad with 0
                t = torch.zeros(len(features), max_len, dtype=torch.long)
                for i, f in enumerate(features):
                    seq_len = f[key].shape[0]
                    t[i, max_len - seq_len :] = f[key]
                batch[key] = t
            elif key in ("pixel_values", "pixel_values_videos"):
                vals = [f[key] for f in features if key in f]
                if vals:
                    # Concatenate along dim=0 (patch dimension, not batch)
                    # Qwen2.5-VL visual encoder expects [total_patches, features]
                    batch[key] = torch.cat(vals, dim=0)
            elif key in ("image_grid_thw", "video_grid_thw"):
                vals = [f[key] for f in features if key in f]
                if vals:
                    # Each val should be [N, 3]; ensure 2D before concat
                    vals_2d = [v.unsqueeze(0) if v.dim() == 1 else v for v in vals]
                    batch[key] = torch.cat(vals_2d, dim=0)
            elif key == "second_per_grid_ts":
                # second_per_grid_ts is a list of floats from the processor
                vals = []
                for f in features:
                    if key in f:
                        v = f[key]
                        if isinstance(v, list):
                            vals.extend(v)
                        elif isinstance(v, (int, float)):
                            vals.append(v)
                        elif isinstance(v, torch.Tensor):
                            if v.dim() == 0:
                                vals.append(v.item())
                            else:
                                vals.extend(v.tolist())
                if vals:
                    batch[key] = torch.tensor(vals, dtype=torch.float32)
            elif key == "_raw_video_frames":
                # Keep raw video frames as list of lists of PIL Images for logging
                batch["_raw_video_frames"] = [f[key] for f in features if key in f]
            elif key.startswith("_") and key not in ("_raw_video_frames",):
                # Keep string metadata (frame_id, scene_token, etc.) as list of strings
                batch[key] = [f[key] for f in features if key in f]

        batch["actions"] = actions

        # Collate COT ids/labels: RIGHT-pad cot_ids with pad_id, cot_labels with -100
        if any(x is not None for x in cot_ids_list):
            max_cot_len = max(x.shape[0] for x in cot_ids_list if x is not None)
            cot_ids_padded = torch.full((len(features), max_cot_len), self.pad_id, dtype=torch.long)
            cot_labels_padded = torch.full((len(features), max_cot_len), -100, dtype=torch.long)
            for i, (cid, clbl) in enumerate(zip(cot_ids_list, cot_labels_list)):
                if cid is not None:
                    cot_ids_padded[i, :cid.shape[0]] = cid
                    cot_labels_padded[i, :clbl.shape[0]] = clbl
            batch["cot_ids"] = cot_ids_padded
            batch["cot_labels"] = cot_labels_padded

        return batch
