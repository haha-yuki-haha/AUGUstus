#!/bin/bash
# ==============================================================================
# AUGUstus_GRPO — environment installation (from scratch)
#
# Two-stage stack, ONE command:
#   Stage A — the AUGUstus_SFT base env (torch/transformers/navsim/nuplan/...)
#             built from the sibling repo's pinned requirements.txt, exactly as
#             documented in AUGUstus_SFT/docs/INSTALL.md;
#   Stage B — the verl RL dependencies added on top
#             (recipe/transfusion/requirements_rl.txt + constraints_rl.txt),
#             then this repo installed editable (`pip install -e . --no-deps`).
#
# Layout (fast local disk, NOT the NFS repo — see docs/INSTALL.md):
#   /opt/augustus_grpo/miniconda3   Miniconda "bootstrapper" (any base python)
#   /opt/augustus_grpo/env          project env (Python 3.12)
#
# Usage:
#   bash scripts/setup_env.sh               # install everything
#   bash scripts/setup_env.sh --check       # only verify imports
#
# GPU PROFILE (AUGUSTUS_GPU_PROFILE, default "pro", passed through to the SFT
# stage-A requirements) — selects the CUDA wheel stack, which must match the
# machine's NVIDIA DRIVER (see docs/INSTALL.md §5):
#   pro   (default): torch 2.13.0 + cu130 stack (constraints_rl.txt),
#                    driver >= 580 — TransferQueue installed WITH deps
#   h800           : torch 2.7.0  + cu126 stack (constraints_rl_h800.txt),
#                    driver 550.x — TransferQueue installed --no-deps
#                    (with deps it upgrades torch to cu130 and kills the env)
#   AUGUSTUS_GPU_PROFILE=h800 bash scripts/setup_env.sh
#
# Requires the sibling AUGUstus_SFT repo ($ADT_ROOT/AUGUstus_SFT): the RL
# recipe reuses its model/dataset code AND its third_party/ devkits.
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/env.sh"

MINICONDA_ROOT="${AUGUSTUS_MINICONDA_ROOT}"
PIP_INDEX=${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}
PIP_TRUSTED=${PIP_TRUSTED_HOST:-mirrors.aliyun.com}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

# ---- GPU profile -> requirements / constraints files ------------------------
AUGUSTUS_GPU_PROFILE="${AUGUSTUS_GPU_PROFILE:-pro}"
case "${AUGUSTUS_GPU_PROFILE}" in
    pro)
        SFT_REQUIREMENTS="${AUGUSTUS_SFT_ROOT}/requirements.txt"
        CONSTRAINTS_RL="${REPO_ROOT}/recipe/transfusion/constraints_rl.txt"
        ;;
    h800)
        SFT_REQUIREMENTS="${AUGUSTUS_SFT_ROOT}/requirements-h800.txt"
        CONSTRAINTS_RL="${REPO_ROOT}/recipe/transfusion/constraints_rl_h800.txt"
        ;;
    *)  echo "ERROR: unknown AUGUSTUS_GPU_PROFILE='${AUGUSTUS_GPU_PROFILE}' (pro|h800)" >&2; exit 1 ;;
esac

echo "[setup_env] repo (GRPO)   : ${REPO_ROOT}"
echo "[setup_env] sibling (SFT) : ${AUGUSTUS_SFT_ROOT}"
echo "[setup_env] gpu profile   : ${AUGUSTUS_GPU_PROFILE}"
echo "[setup_env] miniconda     : ${MINICONDA_ROOT}"
echo "[setup_env] env location  : ${AUGUSTUS_ENV_ROOT}"
echo "[setup_env] pip index     : ${PIP_INDEX}"

if [ ! -f "${SFT_REQUIREMENTS}" ]; then
    echo "ERROR: sibling AUGUstus_SFT repo not found at ${AUGUSTUS_SFT_ROOT}" >&2
    echo "       (the RL recipe reuses its model/dataset code + third_party devkits)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# causal-conv1d (Qwen3.5 GatedDeltaNet fast path) — one strategy per profile
#
# "pro"  (torch 2.13 / CUDA 13 wheels): no prebuilt wheel exists, so it is
#        COMPILED FROM SOURCE against the pip-installed nvcc with these fixes:
#   FIX 1: nvidia-cuda-nvcc -> 13.3.73   (ptxas that handles PTX 9.3)
#   FIX 2: nvidia-cuda-cccl 13.3.3.4.1   (nv/target + cub headers)
#   FIX 3: packaging >= 24               (setuptools canonicalize_version)
#   FIX 4: nvcc.profile += -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#          (nvcc 13.3 vs CUDA-13.0 toolkit headers version check)
#   FIX 5: libcudart.so -> libcudart.so.13 symlink (linker needs unversioned)
#
# "h800" (torch 2.7 / cu126 wheels): the upstream v1.6.2.post1 GitHub release
#        ships a PREBUILT cu12/torch2.7/cp312 wheel (cxx11 ABI TRUE, matching
#        the official torch 2.7 wheels) — install it directly.
# ---------------------------------------------------------------------------
build_causal_conv1d() {
    if [ "${AUGUSTUS_GPU_PROFILE}" = "h800" ]; then
        build_causal_conv1d_h800
    else
        build_causal_conv1d_pro
    fi
}

build_causal_conv1d_h800() {
    if "${PY}" -c "import causal_conv1d" 2>/dev/null; then
        echo "[setup_env] causal-conv1d already installed — skipping install"
        return 0
    fi
    # Prebuilt wheel of the upstream v1.6.2.post1 release: cu12 + torch 2.7 +
    # cp312 + cxx11 ABI TRUE (the official torch 2.7 wheels use the new ABI).
    local WHEEL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    local WHEEL_NAME="causal_conv1d-1.6.2.post1+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    local TMP="$(mktemp -d /tmp/causal_conv1dXXXXXX)"
    echo "[setup_env] installing prebuilt causal-conv1d 1.6.2.post1 (cu12/torch2.7) wheel ..."
    # The 194 MB asset downloads slowly through some proxies; retry + resume
    # (-C -) so a throttled/refreshed connection cannot leave a truncated file,
    # and VALIDATE the wheel (a truncated download is not a zip) before pip.
    if curl -sL --retry 5 --retry-all-errors -C - --max-time 900 "${WHEEL_URL}" \
        -o "${TMP}/${WHEEL_NAME}" \
        && "${PY}" -c "import zipfile,sys; zipfile.ZipFile('${TMP}/${WHEEL_NAME}').testzip()" 2>/dev/null; then
        "${PIP_UNSET[@]}" "${PY}" -m pip install "${TMP}/${WHEEL_NAME}" \
            --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}" \
            && rm -rf "${TMP}" \
            && "${PY}" -c "import causal_conv1d; print('[setup_env] causal_conv1d', causal_conv1d.__version__, 'OK')" \
            && return 0
    fi
    rm -rf "${TMP}"
    echo "[setup_env] WARNING: prebuilt causal-conv1d wheel download failed"
    echo "             (github unreachable?) — skipping; GatedDeltaNet uses the"
    echo "             slower torch fallback. To install manually, download"
    echo "             ${WHEEL_URL}"
    echo "             and 'pip install' it, or build from the v1.6.2.post1 sdist"
    echo "             against a CUDA-12 nvcc toolchain."
    return 0
}

build_causal_conv1d_pro() {
    local CUDA_DIR="$(${PY} -c "import importlib.util, os; spec = importlib.util.find_spec('nvidia'); print(os.path.join(list(spec.submodule_search_locations)[0], 'cu13'))" 2>/dev/null || true)"
    if [ -z "${CUDA_DIR}" ] || [ ! -d "${CUDA_DIR}" ]; then
        echo "[setup_env] nvidia/cu13 toolchain dir not found — skipping causal-conv1d build"
        echo "             (model still runs, GatedDeltaNet uses the slower torch fallback)"
        return 0
    fi
    if "${PY}" -c "import causal_conv1d" 2>/dev/null; then
        echo "[setup_env] causal-conv1d already installed — skipping build"
        return 0
    fi
    echo "[setup_env] building causal-conv1d 1.6.2.post1 from source (CUDA_DIR=${CUDA_DIR}) ..."
    "${PIP_UNSET[@]}" "${PY}" -m pip install --no-cache-dir --upgrade \
        "nvidia-cuda-nvcc==13.3.73" "nvidia-cuda-cccl==13.3.3.4.1" "packaging>=24.0" \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
    # Keep the NVVM backend on the SAME version as nvcc/ptxas: torch 2.13's
    # dependency chain can pull nvidia-nvvm 13.4.x (cicc emitting PTX 9.4),
    # which nvcc/ptxas 13.3.73 cannot assemble ("Unsupported .version 9.4").
    "${PIP_UNSET[@]}" "${PY}" -m pip install --no-cache-dir --force-reinstall --no-deps \
        "nvidia-nvvm==13.3.73" \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
    # FIX 4: patch nvcc.profile
    sed -i 's|SYSTEM_INCLUDES +=  "-isystem" "$(TOP)/include/cccl" $(_SPACE_)|SYSTEM_INCLUDES +=  "-isystem" "$(TOP)/include/cccl" $(_SPACE_) "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" $(_SPACE_)|' \
        "${CUDA_DIR}/bin/nvcc.profile"
    grep -q 'CCCL_DISABLE_CTK_COMPATIBILITY_CHECK' "${CUDA_DIR}/bin/nvcc.profile" \
        || echo "[setup_env] WARNING: nvcc.profile patch not applied"
    # FIX 5: libcudart.so symlink
    ln -sf "${CUDA_DIR}/lib/libcudart.so.13" "${CUDA_DIR}/lib/libcudart.so"

    local TMP="$(mktemp -d /tmp/causal_conv1dXXXXXX)"
    # Source of truth: the PyPI sdist (identical to the GitHub v1.6.2.post1 tag;
    # github.com is unreachable on some servers, so PyPI is the primary).
    if ! curl -sL "https://files.pythonhosted.org/packages/63/5c/2403b8410122d159405c4bd8456340c7251c193358fa24d30cb273fb5048/causal_conv1d-1.6.2.post1.tar.gz" \
        -o "${TMP}/causal-conv1d.tar.gz" || [ ! -s "${TMP}/causal-conv1d.tar.gz" ]; then
        curl -sL "https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.6.2.post1.tar.gz" \
            -o "${TMP}/causal-conv1d.tar.gz"
    fi
    tar xzf "${TMP}/causal-conv1d.tar.gz" -C "${TMP}"
    # The PyPI sdist extracts to `causal_conv1d-…` (underscores), the GitHub
    # tarball to `causal-conv1d-…` (hyphens) — resolve whichever appeared.
    local SRC
    SRC="$(find "${TMP}" -maxdepth 1 -type d -name 'causal*conv1d-1.6.2.post1' | head -n1)"
    ( cd "${SRC}" && \
      CUDA_HOME="${CUDA_DIR}" PATH="${CUDA_DIR}/bin:${PATH}" \
      CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="" MAX_JOBS=4 \
      "${PIP_UNSET[@]}" "${PY}" -m pip install --no-build-isolation . )
    rm -rf "${TMP}"
    "${PY}" -c "import causal_conv1d; print('[setup_env] causal_conv1d', causal_conv1d.__version__, 'OK')"
}

# --------------------------------------------------- conda bootstrap + env --
# Reuse an EXISTING Miniconda when available (e.g. /opt/miniconda3) to avoid a
# second bootstrap; otherwise install a private one under AUGUSTUS_MINICONDA_ROOT.
CONDA="${MINICONDA_ROOT}/bin/conda"
if [ ! -x "${CONDA}" ] && [ -x /opt/miniconda3/bin/conda ]; then
    CONDA=/opt/miniconda3/bin/conda
fi
if [ ! -x "${CONDA}" ]; then
    echo "[setup_env] installing Miniconda -> ${MINICONDA_ROOT}"
    mkdir -p "${MINICONDA_ROOT%/*}"
    INSTALLER="$(mktemp -u /tmp/minicondaXXXXXX.sh)"
    wget -q "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${MINICONDA_ROOT}"
    rm -f "${INSTALLER}"
    # Use mirrors + conda-forge to avoid the repo.anaconda.com ToS prompt.
    cat > "${MINICONDA_ROOT}/.condarc" <<'EOF'
channels:
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
fi

if [ -x "${AUGUSTUS_ENV_ROOT}/bin/python" ]; then
    echo "[setup_env] env already exists at ${AUGUSTUS_ENV_ROOT} — reusing it"
else
    echo "[setup_env] creating Python ${PYTHON_VERSION} env -> ${AUGUSTUS_ENV_ROOT}"
    # Accept repo.anaconda.com ToS non-interactively (newer conda refuses
    # `conda create` until the channel ToS is accepted).
    "${CONDA}" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
    "${CONDA}" tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
    "${CONDA}" create -y -p "${AUGUSTUS_ENV_ROOT}" "python=${PYTHON_VERSION}"
fi

# ------------------------------------------------ named-env convenience link --
# The env lives at $AUGUSTUS_ENV_ROOT (prefix env, on the fast local disk). Also
# expose it under the NAME `augustus_grpo` via a symlink in the conda
# installation's envs/ dir, so plain `conda activate augustus_grpo` works:
#   conda env list      ->  augustus_grpo  /opt/augustus_grpo/env
#   conda activate augustus_grpo
# (A real env already registered under that name is never touched.)
_CONDA_ROOT="$(dirname "$(dirname "${CONDA}")")"
ENV_NAME="${AUGUSTUS_ENV_NAME:-augustus_grpo}"
if [ -d "${_CONDA_ROOT}/envs" ] && [ ! -e "${_CONDA_ROOT}/envs/${ENV_NAME}" ]; then
    ln -sfn "${AUGUSTUS_ENV_ROOT}" "${_CONDA_ROOT}/envs/${ENV_NAME}" \
        && echo "[setup_env] named-env link: conda activate ${ENV_NAME} -> ${AUGUSTUS_ENV_ROOT}"
fi

PY="${AUGUSTUS_ENV_ROOT}/bin/python"
echo "[setup_env] python        : $(${PY} --version 2>&1)"

if [ "${1:-}" = "--check" ]; then
    SKIP_INSTALL=1
else
    SKIP_INSTALL=0
fi

# ------------------------------------------------------------- pip packages --
# NOTE: unset any inherited global pip config (index mirrors / constraints from
# /etc/pip or ~/.pip) so the pinned versions in requirements*.txt are honored.
PIP_UNSET=(env -u PIP_CONSTRAINT -u PIP_EXTRA_INDEX_URL -u PIP_INDEX_URL -u PIP_TRUSTED_HOST)

if [ "${SKIP_INSTALL}" = "0" ]; then
    # ---- Stage A: the AUGUstus_SFT base (pinned training stack) -------------
    echo "[setup_env] [A] installing AUGUstus_SFT requirements (${AUGUSTUS_GPU_PROFILE} profile) ..."
    "${PIP_UNSET[@]}" "${PY}" -m pip install -r "${SFT_REQUIREMENTS}" \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"

    # --------------------------------------------- navsim / nuplan devkits ---
    # Shipped in the SFT repo's third_party/ (Apache-2.0; the navsim copy
    # carries local modifications, e.g. PDMScorerConfig). Editable + no-deps:
    # every dependency they need is pinned in the SFT requirements.
    echo "[setup_env] [A] installing navsim + nuplan-devkit (editable, no deps) ..."
    "${PIP_UNSET[@]}" "${PY}" -m pip install -e "${AUGUSTUS_SFT_ROOT}/third_party/nuplan-devkit" --no-deps \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
    "${PIP_UNSET[@]}" "${PY}" -m pip install -e "${AUGUSTUS_SFT_ROOT}/third_party/navsim" --no-deps \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"

    # ------------------------------------------ causal-conv1d (fast path) ---
    # Qwen3.5's GatedDeltaNet fast path additionally needs causal-conv1d.
    # Strategy depends on the GPU profile (see the builders above):
    # "pro"  — compiled from source against the pip-installed CUDA-13 nvcc
    #          (no prebuilt wheel exists for the torch 2.13 wheel stack);
    # "h800" — prebuilt cu12/torch2.7 wheel from the upstream release.
    build_causal_conv1d

    # ---- Stage B: the verl RL dependencies ----------------------------------
    # Install WITH the profile's constraints file so pip never upgrades torch /
    # transformers / numpy / protobuf / tensordict / nccl (see the header of
    # recipe/transfusion/requirements_rl.txt for the version rationale).
    #
    # h800 PITFALL: TransferQueue must be installed FIRST with --no-deps
    # (+ msgspec). With deps, pip "satisfies" the torch pin by UPGRADING torch
    # to the cu130 build — which cannot run on the 550 driver and leaves a
    # dead env. verl guards `import transfer_queue` with a try/except + mock
    # (verl/utils/transferqueue_utils.py) and the recipe uses
    # verl.trainer.main_ppo (NOT main_ppo_sync), so this is always safe.
    if [ "${AUGUSTUS_GPU_PROFILE}" = "h800" ]; then
        echo "[setup_env] [B] installing TransferQueue --no-deps (h800 pitfall guard) ..."
        "${PIP_UNSET[@]}" "${PY}" -m pip install --no-deps "TransferQueue==0.1.7" \
            --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
        echo "[setup_env] [B] installing TransferQueue runtime deps (pure python) ..."
        "${PIP_UNSET[@]}" "${PY}" -m pip install msgspec pyzmq "prometheus_client>=0.20.0" \
            --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
    fi
    echo "[setup_env] [B] installing RL deps (${AUGUSTUS_GPU_PROFILE} constraints) ..."
    "${PIP_UNSET[@]}" "${PY}" -m pip install \
        -c "${CONSTRAINTS_RL}" \
        -r "${REPO_ROOT}/recipe/transfusion/requirements_rl.txt" \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"

    # ---- Stage B: this repo (vendored verl + the transfusion recipe) --------
    echo "[setup_env] [B] installing AUGUstus_GRPO (editable, no deps) ..."
    "${PIP_UNSET[@]}" "${PY}" -m pip install -e "${REPO_ROOT}" --no-deps \
        --index-url "${PIP_INDEX}" --trusted-host "${PIP_TRUSTED}"
fi

# ------------------------------------------------------------------- verify --
echo "[setup_env] verifying imports ..."
"${PY}" - <<'EOF'
import importlib
mods = ["torch", "torchvision", "transformers", "accelerate", "deepspeed",
        "datasets", "av", "fla", "causal_conv1d", "cv2", "moviepy", "PIL", "yaml",
        "shapely", "pyquaternion", "pandas", "sklearn", "hydra", "omegaconf",
        "nuplan", "navsim", "ray", "aiofiles", "rasterio",
        # --- RL additions (stage B) ---
        "tensordict", "cachetools", "codetiming", "pylatexenc", "torchdata", "uvicorn",
        "fastapi", "dill", "math_verify", "latex2sympy2_extended", "pyvers",
        "orjson", "pyarrow", "verl", "recipe.transfusion", "transfer_queue", "peft"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  OK   {m}")
    except Exception as e:
        failed.append(m)
        print(f"  FAIL {m}: {e}")
if failed:
    raise SystemExit(f"missing/broken modules: {failed}")
print("All imports OK.")
EOF

echo "[setup_env] DONE. Activate with: source ${REPO_ROOT}/env.sh && _activate_augustus_env"
