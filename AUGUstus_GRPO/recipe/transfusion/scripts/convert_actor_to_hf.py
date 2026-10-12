#!/usr/bin/env python
# ============================================================================
# Convert a verl RL actor FSDP checkpoint into a HuggingFace-runnable folder
# (model.safetensors + config.json + tokenizer + chat_template + processor).
#
# verl saves RL actor weights as per-rank FSDP shards:
#     <actor_dir>/model_world_size_<W>_rank_<r>.pt
# plus, on rank 0, an HF config/tokenizer seed under:
#     <actor_dir>/huggingface/
#
# That `huggingface/` seed is written on EVERY save and therefore ONLY holds
# config + tokenizer, NOT weights. Converted weights are produced in-training
# only when the RL config requests saving the full HF model (`should_save_hf_model`);
# this recipe does NOT, so the trained actor must be converted offline here.
#
# IMPLEMENTATION:
#   * Uses verl's upstream merger logic to reconstruct the full state dict from
#     the per-rank FSDP shards and writes model.safetensors + copies the HF
#     config/tokenizer seed. It does NOT instantiate the Qwen3.5 transformers
#     model class, so no AutoModel rebuild is involved -- the merged keys are
#     already in the exact HF layout the SFT model uses.
#   * Post-fix: verl's merger concatenates NON-DTensor values across ranks.
#     Rank-replicated buffers (Transfusion's action_q01/action_q99/dim_weights)
#     would be exported 8x too large ([64,3] instead of [8,3]), so they are
#     restored to the rank-0 shape after verifying every rank saved the same
#     values.
#
# TORCH COMPATIBILITY (no environment changes needed):
#   Shards saved by torch >= 2.13 reference torch.distributed._mesh_layout
#   (_MeshLayout/_FlatLayout) and torch.distributed.tensor._dtensor_spec.
#   ShardOrderEntry, which a torch 2.7 env does not have, plus a renamed
#   DeviceMesh state layout (>=2.13: `_rank_map`/`_mesh_dim_names` vs <2.8:
#   `mesh`/`mesh_dim_names`). Conversely, a torch >= 2.13 env loading shards
#   written by torch < 2.8 hits "AttributeError: 'DeviceMesh' object has no
#   attribute '_mesh_dim_names'" (FSDP sharded-state-dict load). The
#   `_install_torch_legacy_shims()` therefore registers IN-PROCESS, data-only
#   replicas of the missing classes (faithful to upstream torch v2.13.0 field
#   names) and a BIDIRECTIONAL DeviceMesh.__setstate__ normalizer that detects
#   the pickle's format and restores whichever attribute set the local torch
#   expects (plus DTensorSpec default backfill). Pickle loads these classes via
#   NEWOBJ+BUILD (attribute assignment), so no upstream __init__ logic is
#   skipped; the numeric payload is untouched.
#
# USAGE:
#   python convert_actor_to_hf.py --actor <actor_dir> --output <hf_model_dir>
# ============================================================================

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _install_torch_legacy_shims() -> None:
    """Enable this env's torch (< 2.13) to unpickle shards saved by torch >= 2.13.

    No files or packages are modified: everything is registered in `sys.modules`
    / on the live classes of this process only. The shim classes are faithful,
    data-only replicas of the upstream torch v2.13.0 definitions (same module
    path, class name and field names), because the payload restored by pickle
    is exactly what training wrote; the shim only *names* it.
    """
    try:
        import torch.distributed  # noqa: F401
    except Exception:
        return

    import types
    from typing import NamedTuple

    import torch

    # --- 1) torch.distributed.tensor._dtensor_spec.ShardOrderEntry ----------
    try:
        from torch.distributed.tensor import _dtensor_spec as _spec_mod

        if not hasattr(_spec_mod, "ShardOrderEntry"):

            class ShardOrderEntry(NamedTuple):
                """Upstream replica: torch/distributed/tensor/_dtensor_spec.py (v2.13.0)."""

                tensor_dim: int
                mesh_dims: tuple  # guaranteed non-empty upstream

            ShardOrderEntry.__module__ = _spec_mod.__name__
            _spec_mod.ShardOrderEntry = ShardOrderEntry
    except Exception:
        pass

    # --- 2) torch.distributed._mesh_layout {_MeshLayout, _FlatLayout} -------
    import torch.distributed as _dist

    if not hasattr(_dist, "_mesh_layout") and "torch.distributed._mesh_layout" not in sys.modules:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _FlatLayout:
            """Upstream replica: torch/distributed/_mesh_layout.py (v2.13.0)."""

            shape: tuple
            stride: tuple

        @dataclass(frozen=True)
        class _MeshLayout:
            """Upstream replica: torch/distributed/_mesh_layout.py (v2.13.0)."""

            axes: tuple

        mod = types.ModuleType("torch.distributed._mesh_layout")
        mod._FlatLayout = _FlatLayout
        mod._MeshLayout = _MeshLayout
        sys.modules["torch.distributed._mesh_layout"] = mod
        setattr(_dist, "_mesh_layout", mod)

    # --- 3) DeviceMesh / DTensorSpec: cross-version state normalization -----
    # DeviceMesh pickle state changed layout between torch versions:
    #   torch < 2.8  : public keys  device_type / mesh / mesh_dim_names
    #                  (+ _flatten_mesh_list, _thread_id, _dim_group_infos, ...)
    #   torch >= 2.8 : underscore keys  _device_type / _rank_map / _mesh_dim_names
    #                  (+ _layout, _flatten_rank_map, _dim_group_names, ...)
    # A unified __setstate__ detects which format the pickle stream carries and
    # restores whichever attribute set the LOCAL torch expects, so shards saved
    # by one version load cleanly in the other:
    #   * torch >= 2.13 process loading shards written by a torch 2.7 process
    #     (this happened resuming global_step_6400: "AttributeError: 'DeviceMesh'
    #     object has no attribute '_mesh_dim_names'" inside FSDP's sharded
    #     pre-load hook, because the unpickled mesh only had the old names);
    #   * torch < 2.13 process loading shards written by torch >= 2.13 (the
    #     original direction this shim was written for).
    try:
        from torch.distributed.device_mesh import DeviceMesh

        try:
            from torch.distributed.device_mesh import _resolve_process_group
        except ImportError:
            _resolve_process_group = None

        _orig_dm_setstate = DeviceMesh.__dict__.get("__setstate__")
        # Does the LOCAL torch use the renamed (underscore) internals?
        _new_style_local = "_rank_map" in getattr(DeviceMesh, "__annotations__", {})
        _already_patched = getattr(getattr(DeviceMesh, "__setstate__", None), "_transfusion_normalized", False)

        if not _already_patched and (_orig_dm_setstate is not None or not _new_style_local):

            def _devicemesh_setstate(self, state) -> None:
                state = dict(state)
                if "_device_type" in state or "_rank_map" in state:
                    # ---- state saved by torch >= 2.8 (underscore keys) -------
                    if _new_style_local:
                        if _orig_dm_setstate is not None:
                            _orig_dm_setstate(self, state)  # native restore
                        else:  # pragma: no cover - old torch with new keys
                            self.__dict__.update(state)
                        return
                    # local torch < 2.8: rename to the public names it expects
                    rank_map = state.pop("_rank_map", None)
                    if rank_map is not None and "mesh" not in state:
                        state["mesh"] = rank_map
                    mesh_dim_names = state.pop("_mesh_dim_names", None)
                    if mesh_dim_names is not None and "mesh_dim_names" not in state:
                        state["mesh_dim_names"] = mesh_dim_names
                    device_type = state.pop("_device_type", None)
                    if device_type is not None and "device_type" not in state:
                        state["device_type"] = device_type
                    self.__dict__.update(state)
                    # torch < 2.13 lazily derives caches in __init__ that
                    # unpickling never populates; rebuild the cheap ones.
                    mesh_t = self.__dict__.get("mesh")
                    if mesh_t is not None and "_flatten_mesh_list" not in self.__dict__:
                        try:
                            self.__dict__["_flatten_mesh_list"] = tuple(mesh_t.flatten().tolist())
                        except Exception:
                            pass
                    if "_thread_id" not in self.__dict__:
                        self.__dict__["_thread_id"] = None
                    if "_hash" not in self.__dict__:
                        self.__dict__["_hash"] = None
                    if "_coordinate_on_dim" not in self.__dict__:
                        self.__dict__["_coordinate_on_dim"] = None
                    if "_dim_group_infos" not in self.__dict__ and mesh_t is not None:
                        try:
                            from torch.distributed.distributed_c10d import (
                                _get_default_group,
                                _get_group_tag,
                            )

                            pg = _get_default_group()
                            ranks = tuple(mesh_t.flatten().tolist())
                            self.__dict__["_dim_group_infos"] = [(_get_group_tag(pg), ranks, pg.group_name)]
                        except Exception:
                            pass
                    return

                # ---- state saved by torch < 2.8 (public keys) ----------------
                if not _new_style_local:
                    self.__dict__.update(state)  # names already match locally
                    return
                # local torch >= 2.8: translate old -> new attribute names
                device_type = state.pop("device_type", None)
                if device_type is not None:
                    state["_device_type"] = device_type
                rank_map = state.pop("mesh", None)
                if rank_map is not None:
                    state["_rank_map"] = rank_map
                mesh_dim_names = state.pop("mesh_dim_names", None)
                if mesh_dim_names is not None:
                    state["_mesh_dim_names"] = mesh_dim_names
                # rebuild torch >= 2.8-only internals the old pickle lacks
                if "_layout" not in state and state.get("_rank_map") is not None:
                    try:
                        from torch.distributed._mesh_layout import _MeshLayout

                        rm = state["_rank_map"]
                        state["_layout"] = _MeshLayout.from_sizes_strides(
                            tuple(rm.size()), tuple(rm.stride())
                        )
                    except Exception:
                        pass
                if "_flatten_rank_map" not in state and state.get("_rank_map") is not None:
                    try:
                        state["_flatten_rank_map"] = tuple(state["_rank_map"].flatten().tolist())
                    except Exception:
                        pass
                state.setdefault("_thread_id", None)
                state.setdefault("_root_mesh", None)
                state.setdefault("_flatten_mapping", {})
                state.setdefault("_pg_registry", {})
                if _dist.is_initialized():
                    state.setdefault("_rank", _dist.get_rank())
                else:
                    state.setdefault("_rank", None)
                # 2.7 kept per-dim PGs as `_dim_group_infos` [(tag, ranks, name)];
                # 2.8+ keeps a flat list of group names.
                infos = state.pop("_dim_group_infos", None)
                if infos and "_dim_group_names" not in state:
                    state["_dim_group_names"] = [info[2] for info in infos if info]
                self.__dict__.update(state)
                # Replicate torch >= 2.8's __setstate__: resolve PGs by name so
                # later collectives (e.g. FSDP redistribute/all-gather) work.
                self._pg_registry = {}
                for name in self.__dict__.get("_dim_group_names", None) or []:
                    try:
                        pg = _resolve_process_group(name)
                        if pg is not None:
                            self._pg_registry[name] = pg
                    except Exception:
                        pass  # PGs may not exist in this process (pickled elsewhere)

            _devicemesh_setstate._transfusion_normalized = True
            DeviceMesh.__setstate__ = _devicemesh_setstate
    except Exception:
        pass

    # 2.13 added DTensorSpec fields (`shard_order`, `use_strided_shard_as_
    # shard_order`) that older pickles lack; fill the 2.13 defaults after a
    # plain dict-restore so downstream code never sees a missing attribute.
    try:
        from torch.distributed.tensor import _dtensor_spec as _spec_mod

        if not hasattr(_spec_mod.DTensorSpec, "__setstate__"):

            def _dtensorspec_setstate(self, state) -> None:
                self.__dict__.update(state)
                if "shard_order" not in self.__dict__:
                    self.__dict__["shard_order"] = None
                if "use_strided_shard_as_shard_order" not in self.__dict__:
                    placements = self.__dict__.get("placements") or ()
                    self.__dict__["use_strided_shard_as_shard_order"] = any(
                        type(p).__name__ == "_StridedShard" for p in placements
                    )

            _spec_mod.DTensorSpec.__setstate__ = _dtensorspec_setstate
    except Exception:
        pass


def _resolve_actor_dir(actor: str | None, ckpt: str | None) -> Path:
    if actor:
        path = Path(actor).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"[convert_actor_to_hf] actor dir not found: {path}")
        return path

    if ckpt:
        root = Path(ckpt).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"[convert_actor_to_hf] ckpt path not found: {root}")

        direct = root / "actor"
        if direct.is_dir() and (direct / "huggingface").is_dir():
            return direct

        actor_dirs = [a for a in root.glob("*/actor") if (a / "huggingface").is_dir()]
        if not actor_dirs:
            raise SystemExit(
                f"[convert_actor_to_hf] could not locate a verl 'actor' dir under {root}. "
                f"Pass --actor <dir>/actor explicitly."
            )
        best = [a for a in actor_dirs if a.parent.name.startswith("best")]
        if best:
            return sorted(best)[0]

        def _step(p: Path) -> int:
            try:
                return int(p.parent.name.rsplit("_", 1)[-1])
            except Exception:
                return -1

        return sorted(actor_dirs, key=_step)[-1]

    raise SystemExit("[convert_actor_to_hf] either --actor or --ckpt is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=str, default=None, help="verl actor dir (has model_world_size_*.pt + fsdp_config.json + huggingface/)")
    parser.add_argument("--ckpt", type=str, default=None, help="run timestamp dir / global_step_N / *.ckpt dir containing an 'actor'")
    parser.add_argument("--output", type=str, required=True, help="target directory for the merged HuggingFace model")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="OPTIONAL SFT/base HF dir (with model.safetensors) used for structure check: "
        "converted output must have the SAME key set + shape + dtype per key "
        "(only values change). Strongly recommended to catch layout bugs.",
    )
    args = parser.parse_args()

    actor_dir = _resolve_actor_dir(args.actor, args.ckpt)
    seed_dir = actor_dir / "huggingface"
    if not seed_dir.is_dir():
        raise SystemExit(f"[convert_actor_to_hf] no HF config/tokenizer seed at {seed_dir}")
    print(f"[convert_actor_to_hf] actor dir: {actor_dir}")

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # ---- reconstruct the full state dict from the FSDP shards (verl merger) ----
    import torch

    verllib = Path(__file__).resolve().parents[3]  # .../AUGUstus_GRPO
    sys.path.insert(0, str(verllib))

    # install cross-torch-version unpickle shims *before* the merger torch.load()s
    _install_torch_legacy_shims()

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(output),
        hf_model_config_path=str(seed_dir),
        local_dir=str(actor_dir),
    )
    merger = FSDPModelMerger(cfg)
    world_size = merger._get_world_size()
    rank_zero = merger._load_rank_zero_state_dict(world_size)
    mesh, dim_names = merger._extract_device_mesh_info(rank_zero, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, dim_names)
    print(f"[convert_actor_to_hf] world_size={world_size} total_shards={total_shards} mesh={mesh_shape}")

    # verl's FSDP merger concatenates NON-DTensor values across ranks
    # (fsdp_model_merger._load_and_merge_state_dicts -> torch.cat(..., dim=0)).
    # Replicated buffers (registered via register_buffer, identical on every
    # rank, e.g. Transfusion's action_q01/action_q99/dim_weights) must NOT be
    # concatenated: record their per-rank (true) shapes here and undo the
    # concat below.
    try:
        from torch.distributed.tensor import DTensor as _DTensor
    except ImportError:  # older torch location
        from torch.distributed._tensor import DTensor as _DTensor
    replicated_shapes = {k: tuple(v.shape) for k, v in rank_zero.items() if not isinstance(v, _DTensor)}

    state_dict = merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, dim_names)
    del rank_zero

    # ---- undo the concat for rank-replicated buffers (assert all ranks equal) ----
    for key, shape in replicated_shapes.items():
        if key not in state_dict or tuple(state_dict[key].shape) == shape:
            continue
        merged = state_dict[key]
        view = merged.reshape(world_size, *shape)
        ref = view[0]
        for r in range(1, world_size):
            if not torch.equal(ref, view[r]):
                raise RuntimeError(
                    f"[convert_actor_to_hf] buffer '{key}' differs across ranks "
                    f"(rank 0 vs rank {r}); cannot restore single-copy shape {shape} safely."
                )
        state_dict[key] = ref.contiguous()
        print(f"  restored replicated buffer {key}: {tuple(merged.shape)} -> {shape}")

    print(f"[convert_actor_to_hf] merged {len(state_dict)} tensors; saving safetensors ...")

    # ---- write merged weights as a single HF safetensors (bf16) ----
    from safetensors.torch import save_file

    # normalize to contiguous bf16 CPU tensors (discard any leftover shard metadata)
    tensors = {k: v.bfloat16().contiguous() for k, v in state_dict.items()}
    safetensors_path = output / "model.safetensors"
    save_file(tensors, safetensors_path)

    # ---- post-conversion verification (do NOT skip: cheap, catches silent bugs) ----
    _verify_safetensors(safetensors_path, expected=tensors)
    if args.reference:
        _verify_structure_against_reference(safetensors_path, args.reference)
    del tensors, state_dict

    # ---- copy the HF config/tokenizer seed files ----
    copy_names = [
        "config.json",
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for name in copy_names:
        src = seed_dir / name
        if src.is_file():
            shutil.copy2(src, output / name)
            print(f"  copied {name}")

    # ---- add generation_config.json (matches the SFT pytorch_model layout) ----
    gen_cfg = output / "generation_config.json"
    if not gen_cfg.exists():
        gen_cfg.write_text(
            json.dumps(
                {
                    "_from_model_config": True,
                    "eos_token_id": 248044,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "transformers_version": "5.9.0",
                    "use_cache": True,
                },
                indent=2,
            )
            + "\n"
        )

    # single-shard layout: transformers auto-discovers model.safetensors (no index needed)

    size_bytes = safetensors_path.stat().st_size
    print(f"[convert_actor_to_hf] done -> {output}  ({size_bytes/1e9:.2f} GB)")


def _verify_safetensors(safetensors_path: "Path", expected: dict) -> None:
    """Re-read the just-written safetensors and compare EVERY tensor bitwise
    against the in-memory merged state dict. Byte-identical is required because
    bf16 cast already happened in memory before saving."""
    import torch
    from safetensors.torch import load_file

    loaded = load_file(str(safetensors_path))
    if set(loaded.keys()) != set(expected.keys()):
        only_file = sorted(set(loaded) - set(expected))
        only_mem = sorted(set(expected) - set(loaded))
        raise SystemExit(
            f"[verify] key set mismatch: only_in_file={only_file[:5]} (n={len(only_file)}), "
            f"only_in_memory={only_mem[:5]} (n={len(only_mem)})"
        )
    for key in sorted(expected):
        a, b = loaded[key], expected[key]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            raise SystemExit(f"[verify] {key}: dtype/shape mismatch {b.dtype}{tuple(b.shape)} -> {a.dtype}{tuple(a.shape)}")
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs().max().item()
            raise SystemExit(f"[verify] {key}: values differ (max abs diff {diff})")
    print(f"[verify] round-trip safetensors OK: {len(loaded)} tensors byte-identical")


def _verify_structure_against_reference(safetensors_path: "Path", reference: str) -> None:
    """Compare key set + shape + dtype against a base HF model (e.g. the SFT
    pytorch_model dir). Values must differ (training happened) but layout must
    be identical — this catches missing/extra/renamed keys."""
    from safetensors import safe_open

    ref_path = Path(reference).expanduser().resolve()
    cand = ref_path / "model.safetensors"
    if not cand.is_file():
        cand = ref_path / "pytorch_model" / "model.safetensors"
    if not cand.is_file():
        print(f"[verify-reference] no model.safetensors found under {ref_path}; skipped")
        return
    import torch

    from safetensors.torch import load_file

    new = load_file(str(safetensors_path))
    old = load_file(str(cand))
    if set(new.keys()) != set(old.keys()):
        only_new = sorted(set(new) - set(old))
        only_old = sorted(set(old) - set(new))
        raise SystemExit(
            f"[verify-reference] key set differs from base {cand}: "
            f"new-only={only_new[:5]} (n={len(only_new)}), base-only={only_old[:5]} (n={len(only_old)})"
        )
    bad = []
    n_same = 0
    for k in sorted(new):
        if tuple(new[k].shape) != tuple(old[k].shape) or new[k].dtype != old[k].dtype:
            bad.append(k)
        elif torch.equal(new[k], old[k]):
            n_same += 1
    if bad:
        raise SystemExit(f"[verify-reference] shape/dtype mismatch on {len(bad)} keys, first: {bad[:5]}")
    # Looseness by design: float32/bf16 buffers (e.g. action_q01) may coincide.
    print(f"[verify-reference] layout OK: {len(new)} keys match base {cand} "
          f"({n_same} tensors coincidentally equal, {len(new)-n_same} changed)")


if __name__ == "__main__":
    main()
