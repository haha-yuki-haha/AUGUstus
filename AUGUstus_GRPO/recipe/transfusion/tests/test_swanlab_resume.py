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

"""Unit tests for the swanlab-init resume patch (registrations.py).

Run::

    python recipe/transfusion/tests/test_swanlab_resume.py
"""

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recipe.transfusion.registrations import _patch_swanlab_init_for_resume  # noqa: E402


def _fake_swanlab():
    """A fake `swanlab` module: init() records kwargs and mimics id assignment."""
    mod = types.SimpleNamespace()
    mod.calls = []

    def fake_init(**kwargs):
        mod.calls.append(kwargs)
        run = types.SimpleNamespace()
        run.id = kwargs.get("id") or "fresh-id-123"
        return run

    mod.init = fake_init
    return mod


def test_fresh_init_creates_and_persists_id():
    with tempfile.TemporaryDirectory() as tmp:
        mod = _fake_swanlab()
        assert _patch_swanlab_init_for_resume(mod) is True
        run = mod.init(project="p", experiment_name="e", logdir=tmp)
        assert mod.calls[-1].get("id") is None  # fresh: no id override
        assert "resume" not in mod.calls[-1]
        # experiment id persisted for the NEXT launch into the same logdir
        with open(os.path.join(tmp, "run_id.txt")) as f:
            assert f.read().strip() == run.id == "fresh-id-123"


def test_relaunch_same_logdir_resumes_experiment():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run_id.txt"), "w") as f:
            f.write("existing-exp-42")
        mod = _fake_swanlab()
        _patch_swanlab_init_for_resume(mod)
        run = mod.init(project="p", experiment_name="e", logdir=tmp)
        assert mod.calls[-1]["id"] == "existing-exp-42"
        assert mod.calls[-1]["resume"] == "allow"
        assert run.id == "existing-exp-42"
        assert open(os.path.join(tmp, "run_id.txt")).read().strip() == "existing-exp-42"


def test_explicit_id_wins_over_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run_id.txt"), "w") as f:
            f.write("persisted")
        mod = _fake_swanlab()
        _patch_swanlab_init_for_resume(mod)
        mod.init(logdir=tmp, id="explicit", resume="must")
        assert mod.calls[-1]["id"] == "explicit"
        assert mod.calls[-1]["resume"] == "must"  # caller's resume untouched
        # the effectively-used id becomes the one persisted for next time
        assert open(os.path.join(tmp, "run_id.txt")).read().strip() == "explicit"


def test_env_var_logdir_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run_id.txt"), "w") as f:
            f.write("env-id-7")
        os.environ["SWANLAB_LOG_DIR"] = tmp
        try:
            mod = _fake_swanlab()
            _patch_swanlab_init_for_resume(mod)
            mod.init(project="p", experiment_name="e")  # NO logdir kwarg
            assert mod.calls[-1]["id"] == "env-id-7"
            assert mod.calls[-1]["resume"] == "allow"
        finally:
            del os.environ["SWANLAB_LOG_DIR"]


def test_patch_is_idempotent():
    mod = _fake_swanlab()
    assert _patch_swanlab_init_for_resume(mod) is True
    wrapped = mod.init
    assert _patch_swanlab_init_for_resume(mod) is False  # not re-wrapped
    assert mod.init is wrapped


def test_noncloud_mode_skips_injection():
    # swanlab SDK demotes resume to 'never' outside cloud mode and 'never'+id
    # raises RuntimeError -> the wrapper must NOT inject id for local/disabled.
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "run_id.txt"), "w") as f:
            f.write("persisted")
        mod = _fake_swanlab()
        _patch_swanlab_init_for_resume(mod)
        mod.init(logdir=tmp, mode="disabled")
        assert mod.calls[-1].get("id") is None and "resume" not in mod.calls[-1]
        mod.init(logdir=tmp, mode="local")
        assert mod.calls[-1].get("id") is None and "resume" not in mod.calls[-1]
        mod.init(logdir=tmp, mode="cloud")
        assert mod.calls[-1]["id"] == "persisted" and mod.calls[-1]["resume"] == "allow"


def test_no_id_file_no_resume_kwargs():
    with tempfile.TemporaryDirectory() as tmp:
        mod = _fake_swanlab()
        _patch_swanlab_init_for_resume(mod)
        mod.init(logdir=tmp)
        assert "id" not in mod.calls[-1] or mod.calls[-1]["id"] is None
        assert "resume" not in mod.calls[-1]


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
