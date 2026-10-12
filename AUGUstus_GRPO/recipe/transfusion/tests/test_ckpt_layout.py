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

"""Unit tests for the Transfusion-style run/checkpoint layout (ckpt_layout.py).

Run::

    python recipe/transfusion/tests/test_ckpt_layout.py
"""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from omegaconf import OmegaConf  # noqa: E402

from recipe.transfusion.ckpt_layout import apply_checkpoint_layout  # noqa: E402

_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def _make_cfg(tmp, resume=None, save_optimizer=True):
    cfg = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": os.path.join(tmp, "RL", "exp"),
                "rollout_data_dir": "${trainer.default_local_dir}/rollout_data",
                "resume_mode": "disable",
                "resume_from_path": None,
                "save_optimizer": save_optimizer,
            },
            "actor_rollout_ref": {
                "actor": {
                    "checkpoint": {
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": "${.save_contents}",
                    }
                }
            },
        }
    )
    if resume is not None:
        cfg.trainer.resume = resume
    return cfg


def _mk_step(run_dir, step, tracker=True):
    step_dir = os.path.join(run_dir, f"global_step_{step}")
    os.makedirs(os.path.join(step_dir, "actor"), exist_ok=True)
    if tracker:
        with open(os.path.join(run_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(step))
    return step_dir


def test_fresh_run_mints_timestamp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume={"enabled": False, "resume_path": None})
        apply_checkpoint_layout(cfg)
        run_dir = cfg.trainer.default_local_dir
        parent, leaf = os.path.split(run_dir)
        assert parent == os.path.join(tmp, "RL", "exp"), run_dir
        assert _TS_RE.match(leaf), f"leaf dir is not a timestamp: {leaf}"
        assert os.path.isdir(run_dir)
        assert cfg.trainer.resume_mode == "disable" and cfg.trainer.resume_from_path is None
        # ${trainer.default_local_dir} interpolations follow the timestamped dir
        assert os.path.normpath(cfg.trainer.rollout_data_dir) == os.path.join(run_dir, "rollout_data")


def test_no_resume_block_behaves_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume=None)  # yaml without trainer.resume at all
        apply_checkpoint_layout(cfg)
        leaf = os.path.basename(cfg.trainer.default_local_dir)
        assert _TS_RE.match(leaf) and cfg.trainer.resume_mode == "disable"


def test_resume_explicit_step_dir():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "RL", "exp", "20260730_120000")
        step_dir = _mk_step(run_dir, 7)
        cfg = _make_cfg(tmp, resume={"enabled": True, "resume_path": step_dir})
        apply_checkpoint_layout(cfg)
        # continues IN PLACE: no new timestamp dir
        assert cfg.trainer.default_local_dir == run_dir
        assert cfg.trainer.resume_mode == "resume_path"
        assert cfg.trainer.resume_from_path == step_dir


def test_resume_run_dir_picks_latest():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "RL", "exp", "20260730_120000")
        _mk_step(run_dir, 3, tracker=False)
        step5 = _mk_step(run_dir, 5)  # tracker says 5
        cfg = _make_cfg(tmp, resume={"enabled": True, "resume_path": run_dir})
        apply_checkpoint_layout(cfg)
        assert cfg.trainer.resume_from_path == step5
        assert cfg.trainer.default_local_dir == run_dir


def test_resume_null_autodetects_latest_run():
    with tempfile.TemporaryDirectory() as tmp:
        old_run = os.path.join(tmp, "RL", "exp", "20260729_010000")
        _mk_step(old_run, 2)
        new_run = os.path.join(tmp, "RL", "exp", "20260730_010000")
        step8 = _mk_step(new_run, 8)
        cfg = _make_cfg(tmp, resume={"enabled": True, "resume_path": None})
        apply_checkpoint_layout(cfg)
        assert cfg.trainer.default_local_dir == new_run
        assert cfg.trainer.resume_from_path == step8


def test_resume_autodetect_falls_back_fresh():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume={"enabled": True, "resume_path": None})  # nothing on disk
        apply_checkpoint_layout(cfg)
        leaf = os.path.basename(cfg.trainer.default_local_dir)
        assert _TS_RE.match(leaf) and cfg.trainer.resume_mode == "disable"


def test_resume_invalid_path_raises():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume={"enabled": True, "resume_path": os.path.join(tmp, "nope")})
        try:
            apply_checkpoint_layout(cfg)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError for an empty resume_path")


def test_save_optimizer_true_keeps_full_contents():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume={"enabled": False, "resume_path": None}, save_optimizer=True)
        apply_checkpoint_layout(cfg)
        assert list(cfg.actor_rollout_ref.actor.checkpoint.save_contents) == ["model", "optimizer", "extra"]
        # load_contents interpolates save_contents -> follows automatically
        assert list(cfg.actor_rollout_ref.actor.checkpoint.load_contents) == ["model", "optimizer", "extra"]


def test_save_optimizer_false_drops_optimizer():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume={"enabled": False, "resume_path": None}, save_optimizer=False)
        apply_checkpoint_layout(cfg)
        assert list(cfg.actor_rollout_ref.actor.checkpoint.save_contents) == ["model", "extra"]
        assert list(cfg.actor_rollout_ref.actor.checkpoint.load_contents) == ["model", "extra"]


def test_save_optimizer_absent_leaves_contents_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _make_cfg(tmp, resume=None, save_optimizer=True)
        del cfg.trainer["save_optimizer"]
        OmegaConf.update(cfg, "actor_rollout_ref.actor.checkpoint.save_contents", ["model"])
        apply_checkpoint_layout(cfg)
        assert list(cfg.actor_rollout_ref.actor.checkpoint.save_contents) == ["model"]


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
