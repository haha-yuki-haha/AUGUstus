#!/usr/bin/env bash
# ============================================================================
# Launch GRPO/PPO RL fine-tuning of the Transfusion diffusion planner with verl.
#
# This script ONLY sets paths / environment and selects the recipe config. It
# does NOT pass any RL hyper-parameter overrides — all RL settings come from
# verl's `ppo_trainer` defaults (inherited by the recipe config), unchanged.
# ============================================================================
set -euo pipefail

# ============================================================================
# MULTI-NODE LAUNCH — auto-detects nodes & GPUs-per-node
# ============================================================================
# Mirrors the SFT DeepSpeed launcher's auto-detection, adapted for verl
# (verl is Ray-based, NOT rendezvous `accelerate launch`):
#
#   * The SAME script below is submitted once PER NODE (e.g. cluster launcher `--nodes=2
#     --gpus-per-node=<N> --num-proc-per-node=1 ...`), and each node figures out
#     its own role from the injected env vars:
#         SLURM_JOB_NUM_NODES / WORLD_SIZE   -> total node count
#         SLURM_GPUS_PER_NODE / GPUS         -> GPUs on THIS node
#         NODE_RANK / GROUP_RANK / RANK      -> 0 = head, 1+ = worker
#         MASTER_ADDR / MASTER_PORT          -> head rendezvous addr
#   * NODE_RANK 0 (head) starts the Ray HEAD and runs the verl trainer (only ONE
#     trainer process in the whole job — Ray drives the rest).
#   * NODE_RANK >= 1 (workers) ONLY join the Ray cluster and then idle; they do
#     NOT launch a second trainer (that would be a competing Ray job).
#
# So to scale 1 -> 2 machines you change ONLY the node count in your job command
# and, if you don't want all detected GPUs, the per-node GPU count. Everything
# else is detected. Outside a job scheduler, a plain `bash run_grpo_transfusion.sh`
# on one box just trains locally with all detected GPUs (single-node fallback).
# ============================================================================

# ---- conda env (AUGUstus_GRPO env = SFT base + verl RL deps; docs/INSTALL.md) ----
source "$(cd "$(dirname "$0")/../../.." && pwd)/env.sh"
_activate_augustus_env

# ---- repo locations (env.sh is authoritative; here for the snapshot/CP paths) ----
VERL_ROOT="${AUGUSTUS_GRPO_ROOT}"
TRANSFUSION_ROOT="${TRANSFUSION_ROOT:-$AUGUSTUS_SFT_ROOT}"

# The Transfusion model/dataset/rollout build from THIS YAML (the supervised
# training config). Override-able by exporting TRANSFUSION_CONFIG before calling.
export TRANSFUSION_ROOT

# ---- auto-detect nodes & GPUs-per-node (cluster launcher-first, local fallbacks) ------
# Same env-var contract as the SFT DeepSpeed launcher, so the SAME command is
# submitted per node and each node works out its own role. You only override
# GPUS_PER_NODE (default: every GPU on this node) and the job's node count.
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}}}  # GPUs on THIS node
GPUS_PER_NODE=${GPUS_PER_NODE:-8}   # final fallback if nvidia-smi unavailable
NUM_NODES=${NUM_NODES:-${SLURM_JOB_NUM_NODES:-${WORLD_SIZE:-1}}}      # total machines
NODE_RANK=${NODE_RANK:-${GROUP_RANK:-${RANK:-0}}}                     # 0=master(head), 1=2nd node, ...
MASTER_ADDR=${MASTER_ADDR:-${RAY_HEAD_NODE_IP:-"127.0.0.1"}}         # head rendezvous addr (may be hostname/IP)
MASTER_PORT=${MASTER_PORT:-6379}                                     # Ray head GCS port
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
# ---- resolve a NUMERIC IP for the Ray head and for THIS node. Ray placement
# groups are keyed by node IP; a HOSTNAME `--node-ip-address` (cluster launcher's MASTER_ADDR
# is e.g. "cluster-...-master-0") can make Ray collapse both nodes onto one IP so
# all 16 workers land on one node -> "Duplicate GPU detected". Always use a
# routable dotted-quad IP for Ray node identity.
_pick_ip() { # $1=head-addr-or-empty ; returns THIS node's ROUTABLE ip (172.x preferred)
    local me
    me=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^172\.[0-9]+\.[0-9]+\.[0-9]+$' | head -n1)
    [[ -z "${me}" ]] && me=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "${me:-127.0.0.1}"
}
_resolve_head_ip() { # resolve MASTER_ADDR to a numeric IP if it is a hostname
    local a="${1}"
    if [[ "${a}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then echo "${a}"; return; fi
    local r; r=$(getent ahostsv4 "${a}" 2>/dev/null | awk '{print $1; exit}')
    echo "${r:-$(_pick_ip)}"
}
# Fixed diagnostic log: the launcher's own stdout/echo lines sometimes do NOT land
# in cluster launcher's captured log (only the piped python output does). Always mirror the
# cluster-topology diagnostics to this stable file so they are easy to read back.
# _DIAG_LOG="${VERL_ROOT}/recipe/transfusion/_raydiag_$(whoami).log"
# mkdir -p "$(dirname "${_DIAG_LOG}")"
# _diag(){ printf '%s | %s\n' "$(date +%FT%T%z)" "$*" >> "${_DIAG_LOG}"; }
_THIS_IP="$(_pick_ip)"                 # this node's routable IP
MASTER_NUM_IP="$(_resolve_head_ip "${MASTER_ADDR}")"   # head's numeric IP
# Per-node Ray temp dir so simultaneous `ray start` on BOTH nodes don't collide
# on a shared /tmp/ray (the classic cause of a worker raylet crashing with
# "WaitForDashboardAgentPorts ... Timed out waiting for file /tmp/ray/.../
# dashboard_agent_listen_port" -> its GPUs never register -> 16 workers packed
# onto one node -> NCCL "Duplicate GPU detected").
_RAY_TMP="/dev/shm/ray_${NODE_RANK}_$(hostname | tr -d '.-')"
# ---- Bound the Ray object store. Ray auto-sizes the in-memory object store to
# ~30% of node RAM (on these 1.5TB boxes that is ~450GB reserved under /dev/shm).
# During verl init_model() every worker simultaneously loads the fp32 checkpoint +
# optimizer and stages tensors through the object store; the auto-sizing plus the
# worker processes can push a node over its memory limit and the kernel OOM-kills
# the raylet ("Raylet is terminated ... SIGKILL ... OOM killer" -> the node is
# marked dead from missed heartbeats -> ActorDiedError, the recurring 2-node
# failure). Capping it frees headroom; 120GB is far more than a 2B model needs.
# Tunable via env (default 120GB in bytes).
_RAY_OBJ_MEM="${_RAY_OBJ_MEM:-128849018880}"   # default 120 GiB
# ---- Give the worker raylet slack on heartbeat liveness. The recurring 2-node
# failure is "the node has been marked dead because the detector has missed too
# many heartbeats" right at init_model (when 8 colocated worker actors burst-spawn
# and run the first cross-node NCCL collectives on the worker node). On an
# otherwise healthy node this is usually a *lag* (busy/slow heartbeat thread),
# and Ray's default liveness window (~30 heartbeats) declares the node dead too
# early, which then tears down all its workers (ActorDiedError). The model is tiny
# (a 6GB checkpoint) and bounding the object store does NOT fix it, so the slope
# is a heartbeat/liveness timeout, not a real OOM. Raise the tolerance so a briefly-
# slow worker node is kept alive long enough for init_model to finish. All nodes
# must agree (same script on every node) -- the GCS (head) enforces the window.
export RAY_num_heartbeats_timeout="${RAY_num_heartbeats_timeout:-300}"        # was 30
export RAY_raylet_heartbeat_timeout_ms="${RAY_raylet_heartbeat_timeout_ms:-2000}"  # was 1000
export RAY_num_heartbeats_warning="${RAY_num_heartbeats_warning:-150}"
export RAY_num_heartbeats_warning_2="${RAY_num_heartbeats_warning_2:-250}"
export RAY_health_check_initial_delay_ms="${RAY_health_check_initial_delay_ms:-30000}"    # 30s grace before health checks
export RAY_health_check_period_ms="${RAY_health_check_period_ms:-10000}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"
export RAY_object_store_full_delay_ms="${RAY_object_store_full_delay_ms:-60000}"
echo "[run_grpo] ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU = ${TOTAL_GPUS} total | node_rank=${NODE_RANK} head=${MASTER_ADDR}:${MASTER_PORT} (num_ip=${MASTER_NUM_IP}) this_ip=${_THIS_IP} raytmp=${_RAY_TMP} objmem=${_RAY_OBJ_MEM}"
# _diag "role: rank=${NODE_RANK} this_node_ip=${_THIS_IP} head(${MASTER_ADDR})_num_ip=${MASTER_NUM_IP}:${MASTER_PORT} nnodes=${NUM_NODES} gpus_per_node=${GPUS_PER_NODE}"

# ---- Cross-node NCCL transport (2-node only). Single-node trains fine (intra-node
# NCCL), but EVERY 2-node attempt dies at init_model on the worker-role node, on
# every distinct node pair, right at the first inter-node FSDP collective -- this
# is the genuine 2-node init failure, not a node/hardware/GPU issue. The most common cause
# on these multi-VM boxes is NCCL picking the wrong/hardcoded socket interface or
# timing out over the inter-node link. Tell NCCL to use the interface that carries
# this node's default route, fall back to TCP (disable IB), and give cross-node
# sockets generous timeouts. These only apply to the very first cross-node exchange
# (and are harmless for single-node, which never uses the socket transport). If the
# inter-node interconnect is architecturally down, they will not fix it, but they
# are the right thing to try and are gated to multi-node.
if [[ "${NUM_NODES}" -gt 1 ]]; then
  if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then :  # respect an explicit override
  else
    _defif=$(cat /proc/net/route 2>/dev/null | awk '$2=="00000000"{print $1; exit}')
    [[ -n "${_defif}" ]] && export NCCL_SOCKET_IFNAME="${_defif}"
  fi
  # Prefer TCP over (often-absent/wrong) RDMA on these VMs; a node without a usable
  # IB/RoCE device can otherwise block forever waiting on a link that will never come,
  # hanging the first cross-node collective so the worker node misses heartbeats and
  # Ray declares it dead. NOTE: the image / verl force NCCL_IB_DISABLE=0, so we MUST
  # overwrite (not just default) it for the multi-node case. Only the cross-node link
  # uses this; intra-node (single-node) NCCL never touches it.
  export NCCL_IB_DISABLE=1
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_defif:-enp161s0f0}}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
  export NCCL_SOCKET_TIMEOUT="${NCCL_SOCKET_TIMEOUT:-3600}"
  export NCCL_ALGO="${NCCL_ALGO:-Ring}"
  echo "[run_grpo] multi-node: NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  # _diag "nnodes>1 nccl_socket_ifname=${NCCL_SOCKET_IFNAME} nccl_ib_disable=${NCCL_IB_DISABLE}"
fi

# ---- mode check: does Ray already see all nodes? (a running multi-node cluster
# that we should just join, vs. we need to build one now). When a cluster with
# NUM_NODES is already up we skip starting Ray and go straight to training.
# NOTE: the job-done sentinel (_HEAD_SENTINEL) is defined LATER, once OUT_DIR is
# known, so it lands in each experiment's own output dir (see the output-dir
# block below). This keeps it out of the shared recipe dir across experiments.
_ray_active_nodes=0
if command -v ray >/dev/null 2>&1 && ray status >/dev/null 2>&1; then
  _ray_active_nodes=$(ray status 2>/dev/null | awk '/^Active:/{f=1;next} /^Pending:/{f=0} f && / [0-9]+ node_/ {c++} END{print c+0}')
  _ray_active_nodes=${_ray_active_nodes:-0}
fi
echo "[run_grpo] ray already sees ${_ray_active_nodes} node(s)"
# ---- GPU PHYSICAL IDENTITY on THIS node. The recurring "Duplicate GPU detected:
# rank N and rank N+8 on the SAME CUDA device" can only happen if TWO different
# Ray node IPs expose the SAME physical GPUs (e.g. cluster launcher handing one box two IPs,
# or both nodes mounting the same GPUs). This dumps this node's GPU count + the
# first GPU UUID, so the combined submission log (printed by EVERY node) reveals
# whether the two nodes' GPUs overlap. If both nodes print the SAME device-UUIDs,
# the "two machines" share one set of GPUs and no Ray/config change can fix it.
echo "[run_grpo] $(hostname) nvidia-smi GPU count: $(nvidia-smi -L 2>/dev/null | wc -l), first UUID: $(nvidia-smi -L 2>/dev/null | head -1)"
# _diag "host=$(hostname) nvidia-smi_count=$(nvidia-smi -L 2>/dev/null | wc -l) first_uuid=$(nvidia-smi -L 2>/dev/null | head -1)"

# ---- build OR join the Ray cluster, by node role --------------------------
# verl is Ray-based, so only the HEAD (node_rank 0) runs the trainer; every
# other node joins the cluster and idles. This mirrors the familiar cluster launcher model
# (same script on every node) while keeping verl's single-driver topology.
_kill_stale_ray() {
    # Forcefully remove ANY ray processes still on THIS physical node from a
    # previous job/session. `ray stop -f` only stops raylets attached to the
    # current cluster; a raylet left behind by an EARLIER cluster launcher job on the same
    # node (with a different cluster id) survives it and re-registers a SECOND
    # node_id under this node's IP -> Ray sees 3 nodes, one machine counted twice
    # -> verl packs all workers onto one node -> NCCL 'Duplicate GPU detected'.
    ray stop -f >/dev/null 2>&1 || true
    pkill -9 -f 'raylet' 2>/dev/null || true
    pkill -9 -f 'gcs_server' 2>/dev/null || true
    sleep 3
}

# Robust "is THIS node a live member of the head's cluster?" probe. `ray status`
# output lists nodes as `node_<hex_id>` only -- it contains NO node IP addresses,
# so `ray status | grep <ip>` can NEVER match (this was the recurring 2-node
# failure: the worker's join check always failed, so the retry loop kept
# _kill_stale_ray-ing and restarting its own raylet every ~20s; the head saw the
# worker's raylet die over and over -> "Raylet is terminated unexpectedly" /
# "missed too many heartbeats" -> ActorDiedError at init_model). Ask the GCS
# directly for a live node with our IP instead.
_ray_probe_alive() {
    python - "${MASTER_NUM_IP}" "${MASTER_PORT}" "${_THIS_IP}" <<'PY' >/dev/null 2>&1
import sys
try:
    import ray
    ray.init(address=f"{sys.argv[1]}:{sys.argv[2]}", log_to_driver=False)
    me = sys.argv[3]
    ok = any(n.get("NodeManagerAddress") == me and n.get("Alive") for n in ray.nodes())
    ray.shutdown()
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}

_join_ray_worker() {
    local _tries=0
    # Retry loop that ALSO re-cleans stale raylets before EACH `ray start`, so
    # repeated calls cannot pile up multiple node_ids on this worker (the
    # "one machine counted twice -> Ray sees 3 nodes -> verl packs all workers
    # on one node -> Duplicate GPU" bug). Only ONE raylet/node_id survives.
    while [[ "${_tries}" -lt 90 ]]; do
        _tries=$((_tries + 1))
        if pgrep -f 'raylet' >/dev/null 2>&1 && _ray_probe_alive; then
            echo "[run_grpo] joined Ray head ${MASTER_NUM_IP}:${MASTER_PORT}"
            break
        fi
        _kill_stale_ray   # kill stale + our-own previous raylet before (re)starting
        if ! ray start --address="${MASTER_NUM_IP}:${MASTER_PORT}" --num-gpus="${GPUS_PER_NODE}" \
            --node-ip-address="${_THIS_IP}" \
            --object-store-memory="${_RAY_OBJ_MEM}" \
            --temp-dir="${_RAY_TMP}" >/dev/null 2>&1; then
            echo "[run_grpo] ray start retry ${_tries} (head not up yet)..."
        fi
        sleep 10
    done
    echo "[run_grpo] worker joined; waiting for the head's job to finish..."
    # Idle until the HEAD marks the job done (sentinel written by the head's EXIT
    # trap) OR this node's raylet disappears from the live cluster. On either,
    # shut down this node's raylet and exit so the pod terminates and cluster launcher
    # frees the nodes.
    _gone_seen=0
    while [[ "${_gone_seen}" -lt 3 ]]; do
        if [[ -f "${_HEAD_SENTINEL}" ]]; then
            echo "[run_grpo] head job done (sentinel=${_HEAD_SENTINEL}); worker exiting"
            break
        fi
        if ! _ray_probe_alive; then
            _gone_seen=$((_gone_seen + 1))   # 3 consecutive misses before giving up
        else
            _gone_seen=0
        fi
        sleep 20
    done
    if [[ "${_gone_seen}" -ge 3 ]]; then
        echo "[run_grpo] this node no longer appears live in the cluster; worker exiting"
    fi
    ray stop -f >/dev/null 2>&1 || true
    pkill -9 -f 'raylet' 2>/dev/null || true
    pkill -9 -f 'gcs_server' 2>/dev/null || true
    echo "[run_grpo] worker node shut down"
    exit 0
}

# Once Ray topology is resolved we need one stable output dir shared by HEAD and
# WORKER nodes so they agree on the job-done sentinel path. We resolve OUT_DIR
# HERE (before the worker branch below) so a worker node becomes a live member of
# the cluster, drops into _join_ray_worker and still knows exactly where the head
# will write the sentinel.
CONFIG_NAME="${CONFIG_NAME:-stage1}"  # recipe config (YAML) to launch
read -r _YAML_EXP _YAML_OUT < <(python3 -c "
import os
from omegaconf import OmegaConf

cfg_dir = '${VERL_ROOT}/recipe/transfusion/config'
cfg_name = '${CONFIG_NAME}'
yaml_path = os.path.join(cfg_dir, cfg_name + '.yaml')
cfg = OmegaConf.load(yaml_path)

# Manually resolve 'defaults' inheritance (e.g. flow_grpo inherits base):
# merge parent configs first, then the child (_self_) on top. Parent configs
# that are not in this dir (e.g. verl's 'ppo_trainer') are skipped — we only
# need trainer.experiment_name / trainer.default_local_dir.
defaults = list(cfg.pop('defaults', [])) if hasattr(cfg, 'defaults') else []
merged = OmegaConf.create({})
for entry in defaults:
    if entry == '_self_':
        merged = OmegaConf.merge(merged, cfg)
    elif isinstance(entry, str):
        parent_path = os.path.join(cfg_dir, entry + '.yaml')
        if os.path.exists(parent_path):
            parent = OmegaConf.load(parent_path)
            parent.pop('defaults', None)
            merged = OmegaConf.merge(merged, parent)
# If no defaults list, the config is standalone — use it directly.
if not defaults:
    merged = cfg
OmegaConf.resolve(merged)
exp = merged.trainer.experiment_name
out = merged.trainer.default_local_dir
print(exp, out)
" 2>/dev/null) || true
EXP_NAME="${EXP_NAME:-${_YAML_EXP}}"
OUT_DIR="${OUT_DIR:-${_YAML_OUT}}"
# If EXP_NAME was overridden but OUT_DIR was not, reconstruct OUT_DIR from the
# overridden name (mirrors the YAML's default_local_dir pattern).
if [[ -z "${OUT_DIR}" || "${OUT_DIR}" == "${_YAML_OUT}" ]] && [[ "${EXP_NAME}" != "${_YAML_EXP}" ]]; then
  OUT_DIR="${RL_OUTPUT_ROOT}/${EXP_NAME}"   # mirrors the YAML's default_local_dir pattern
fi
mkdir -p "${OUT_DIR}"

# ---- Job-done sentinel, stored in THIS experiment's output dir (not the shared
# recipe dir). The HEAD writes it when its user command exits (success OR failure,
# via its EXIT trap); the WORKER polls it and, when present, shuts its own raylet
# down and EXITS instead of `sleep infinity`. Because OUT_DIR is derived identically
# on every node, head and worker always agree on the path. Per-experiment location
# keeps multiple concurrent runs from stamping on one shared marker file.
_HEAD_SENTINEL="${OUT_DIR}/_raydone_${MASTER_PORT}.marker"
_ack_sentinel(){ :; }
if [[ "${NODE_RANK}" -eq 0 ]]; then
    rm -f "${_HEAD_SENTINEL}"   # no stale marker from a previous identical-port job
fi

if [[ "${NUM_NODES}" -gt 1 && "${NODE_RANK}" -gt 0 ]]; then
    # ---- worker node: join the head's Ray cluster, do NOT train -----------
    echo "[run_grpo] this is a WORKER node (rank ${NODE_RANK}); joining Ray head"
    _join_ray_worker
    exit 0   # unreachable while sleeping; keeps cluster launcher happy if the head dies
fi

# ---- HEAD node (node_rank 0): make sure a cluster of all NUM_NODES is up,
# wait for workers, then hand off to the verl trainer (below). ---------------
_cluster_ok=0   # 1 = Ray has >= NUM_NODES active nodes
if [[ "${_ray_active_nodes}" -lt "${NUM_NODES}" ]]; then
    echo "[run_grpo] starting Ray head on ${MASTER_NUM_IP}:${MASTER_PORT} (${GPUS_PER_NODE} GPU here)"
    _kill_stale_ray   # clean stale raylets/gcs on this node, then start ONE fresh head
    ray start --head --port="${MASTER_PORT}" --num-gpus="${GPUS_PER_NODE}" \
        --node-ip-address="${MASTER_NUM_IP}" \
        --object-store-memory="${_RAY_OBJ_MEM}" \
        --temp-dir="${_RAY_TMP}" >/dev/null
    echo -n "[run_grpo] waiting for ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU"
    for _ in $(seq 1 60); do
        _now=$(ray status 2>/dev/null | awk '/^Active:/{f=1;next} /^Pending:/{f=0} f && / [0-9]+ node_/ {c++} END{print c+0}')
        if [[ "${_now:-0}" -ge "${NUM_NODES}" ]]; then
            echo " -> ${_now} node(s) up."
            _cluster_ok=1
            break
        fi
        echo -n "."
        sleep 10
    done
fi

# ---- Sanity check the cluster actually has EXACTLY the expected node count.
# For multi-node verl we require Ray to report EXACTLY NUM_NODES distinct nodes:
#   * too FEW  -> partial cluster packs all workers onto the surviving node(s)
#                 and crashes with NCCL 'Duplicate GPU detected';
#   * too MANY -> a stale raylet left a second node_id under one machine's IP
#                 (the observed "3 nodes, two on the same IP"), which ALSO breaks
#                 verl's 2x8 placement into duplicate GPUs.
# Abort clearly instead of running a broken cluster.
if [[ "${NUM_NODES}" -gt 1 ]]; then
    _now_nodes=$(ray status 2>/dev/null | awk '/^Active:/{f=1;next} /^Pending:/{f=0} f && / [0-9]+ node_/ {c++} END{print c+0}')
    _now_nodes=${_now_nodes:-0}
    if [[ "${_now_nodes}" -ne "${NUM_NODES}" ]]; then
        echo "[run_grpo] ERROR: expected EXACTLY ${NUM_NODES} node(s) but Ray reports ${_now_nodes}." >&2
        echo "                     - too few  -> a worker node did not join (check 'ray start --address=${MASTER_NUM_IP}:${MASTER_PORT}')." >&2
        echo "                     - too many -> a STALE raylet left a duplicate node_id under one machine's IP" >&2
        echo "                                  (run 'ray stop -f' on EVERY node and rm -rf /tmp/ray /dev/shm/ray_*)." >&2
        echo "                     Either state makes verl pack multiple workers onto the same GPU (NCCL 'Duplicate GPU detected')." >&2
        echo "[run_grpo] --- ray.nodes() ---" >&2
        python - "${MASTER_NUM_IP}" "${MASTER_PORT}" <<'PY' >&2 2>/dev/null
import sys, ray
ray.init(address=f"{sys.argv[1]}:{sys.argv[2]}", log_to_driver=False)
seen={}
for n in ray.nodes():
    a=n.get("NodeManagerAddress")
    seen.setdefault(a,[]).append(n.get("NodeID")[:8])
for a,ids in seen.items():
    print(f"  {a}  node_ids={ids}")
ray.shutdown()
PY
        exit 1
    fi
fi
# derive actual GPU total Ray sees (diagnostic only)
_cluster_gpus=$(ray status 2>/dev/null \
    | awk 'BEGIN{inres=0} /^Resources/{inres=1} inres && /GPU/{gsub(/[()]/,""); n=$1; split(n,a,"/"); tot+=(a[2]+0)} END{print tot+0}')
echo "[run_grpo] cluster GPU total reported by Ray: ${_cluster_gpus}  (expected ${NUM_NODES} x ${GPUS_PER_NODE} = $((NUM_NODES*GPUS_PER_NODE)))"
# ---- Per-node diagnostic: which IPs does Ray see, and how many GPUs each has.
# This is the deciding check for the "Duplicate GPU" bug: if Ray lists 2 distinct
# node IPs but verl still packs all workers onto ONE of them, or if a node reports
# 16 GPUs (both subnets = same box), we see it here. Connects briefly, prints,
# then disconnects so the trainer's own ray.init() starts clean.
python - "${MASTER_NUM_IP}" "${MASTER_PORT}" <<'PY' 2>&1 | sed 's/^/[ray.nodes] /'
import sys
import ray
ray.init(address=f"{sys.argv[1]}:{sys.argv[2]}", log_to_driver=False)
for n in ray.nodes():
    res = n.get("Resources")
    gpu_total = None
    try:
        if isinstance(res, dict):
            gpu_total = res.get("GPU")
        else:  # string
            import json as _json
            gpu_total = _json.loads(res).get("GPU") if res else None
    except Exception:
        gpu_total = None
    avail = n.get("AvailableResources", {})
    gpu_avail = avail.get("GPU") if isinstance(avail, dict) else None
    print(n.get("NodeManagerAddress"), "| alive=", n.get("Alive"),
          "| GPU total=", gpu_total, " avail=", gpu_avail,
          "| node_id=", n.get("NodeID"))
ray.shutdown()
PY
if [[ "${NUM_NODES}" -gt 1 && "${_cluster_gpus}" -lt $((NUM_NODES * GPUS_PER_NODE)) ]]; then
    echo "[run_grpo] WARNING: Ray reports ${_cluster_gpus} GPU < expected $((NUM_NODES*GPUS_PER_NODE));" >&2
    echo "[run_grpo]          the second node likely failed to register its GPUs." >&2
fi
ray status 2>/dev/null | sed -n '1,12p'

# ---- LOAD A TRAINED MODEL ----
# MODEL_DIR = a trained Transfusion run's timestamp dir; the model architecture
# config and the checkpoint both derive from it. Its config_original.yaml MUST
# match the `best` weights. Point MODEL_DIR (or TRANSFUSION_CONFIG / TRANSFUSION_CKPT
# directly) at your trained model; set TRANSFUSION_CKPT="" for a cold-start.
MODEL_DIR="${MODEL_DIR:-$AUGUSTUS_SFT_RUN}"
export TRANSFUSION_CONFIG="${TRANSFUSION_CONFIG:-${MODEL_DIR}/config_original.yaml}"
export TRANSFUSION_CKPT="${TRANSFUSION_CKPT:-${MODEL_DIR}/best}"      # trained model dir
# ---- SELF-CONTAINED LOADING (no original Qwen3.5-2B backbone / train-data mount) ----
# Backbone architecture + in-model AutoProcessor (tokenizer + image/video processor)
# source: the trained model's own HF dir (best/pytorch_model). The backbone weights
# built from it are OVERWRITTEN by TRANSFUSION_CKPT. Also drives model.path + the RL
# dataset processor via the recipe config.
export TRANSFUSION_QWEN_MODEL_NAME="${TRANSFUSION_QWEN_MODEL_NAME:-${MODEL_DIR}/best/pytorch_model}"
# Action-norm stats JSON (recipe-bundled copy). Only needs to exist; the VALUES are
# overwritten by the checkpoint's action_q01/q99 (avoids reading the train-data mount).
export TRANSFUSION_STATS_FILE="${TRANSFUSION_STATS_FILE:-${VERL_ROOT}/recipe/transfusion/config/action_norm_stats.json}"

# Diffusion-RL knobs consumed by TransfusionRollout (env fallback for the recipe
# config's actor_rollout_ref.rollout.rl_* fields). NUM_STEPS = denoising steps
# used to produce the action (the SFT checkpoint is one-step; RL uses a
# multi-step chain).
export TRANSFUSION_RL_NUM_STEPS="${TRANSFUSION_RL_NUM_STEPS:-10}"
# OPTIONAL: freeze the ViT (visual tower) during RL (default false = trainable).
# export TRANSFUSION_FREEZE_VIT=true

# Make `recipe.*`, `verl.*` and `transfusion.*` importable in driver + workers.
export PYTHONPATH="${VERL_ROOT}:${TRANSFUSION_ROOT}:${PYTHONPATH:-}"

# Reduce CUDA allocator fragmentation (docs/BATCH_SIZE.md #1: zero functional
# risk, first thing to try whenever a step is near the OOM edge).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Opt in to Ray's future accelerator-env-var behaviour now; silences the noisy
# per-process FutureWarning ("Ray will no longer override accelerator visible
# devices env var if num_gpus=0") without changing this recipe's behaviour
# (no worker here relies on Ray clearing CUDA_VISIBLE_DEVICES for CPU tasks).
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"

# Global registration hook (imported by `verl` in every process that sets it).
export VERL_USE_EXTERNAL_MODULES="recipe.transfusion.registrations"

# ---- SwanLab experiment tracking ----
# Export your own SWANLAB_API_KEY only when enabling cloud logging. The
# config sets trainer.logger=[console,swanlab] and forwards these vars to the
# verl driver + every Ray worker (project/experiment come from trainer.*).
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
export SWANLAB_MODE="${SWANLAB_MODE:-offline}"

# ---- output dir + launch-config snapshot (saves the YAML configs used) -------
# EXP_NAME / OUT_DIR were already resolved earlier (above the worker branch, so
# the sentinel path can live under OUT_DIR). This is the HEAD/trainer-only part:
# instance lock, config snapshot, console log, then hand off to verl.

# ---- guard against accidentally launching a SECOND overlapping instance -----
# Each launch spins up a FULL new Ray cluster + trainer.n_gpus_per_node workers;
# two overlapping launches silently compete for the SAME physical GPUs instead
# of erroring (hit on 2026-07-07: two independent runs of this script left ~2.7GB
# duplicated on every GPU and both stuck starved of GPU cycles/NCCL). A stale
# lock (process no longer alive) is auto-reclaimed; a LIVE one aborts instead of
# quietly doubling up -- check `nvidia-smi` / `ps` before overriding.
LOCK_FILE="${OUT_DIR}/run.lock"
if [[ -f "${LOCK_FILE}" ]]; then
  OLD_PID="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "ERROR: run_grpo_transfusion.sh is already running for '${EXP_NAME}' (PID ${OLD_PID})." >&2
    echo "       Refusing to start a SECOND overlapping instance (they would silently share/contend" >&2
    echo "       for the same GPUs). Check 'ps -p ${OLD_PID}' / 'nvidia-smi' first." >&2
    echo "       If you are SURE it's stale: rm -f \"${LOCK_FILE}\"" >&2
    exit 1
  fi
  echo "Stale lock (PID ${OLD_PID:-?} not running) -> reclaiming ${LOCK_FILE}"
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"; printf "done %s\n" "$(date +%FT%T%z)" > "${_HEAD_SENTINEL}"' EXIT

SNAP_DIR="${OUT_DIR}/launch_snapshot/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SNAP_DIR}"
cp -f "${VERL_ROOT}/recipe/transfusion/config/"*.yaml "${SNAP_DIR}/" 2>/dev/null || true
cp -f "${TRANSFUSION_CONFIG}" "${SNAP_DIR}/" 2>/dev/null || true
cp -f "${BASH_SOURCE[0]}" "${SNAP_DIR}/" 2>/dev/null || true
echo "Saved launch config snapshot -> ${SNAP_DIR}"

# ---- full console log, mirrored to a file (driver + every Ray worker's stdout,
# since Ray streams worker logs through the driver process) ----------------
# One timestamped file per launch under the run's own output dir, so ALL of a
# run's output (model build / NCCL / validation / per-step metrics / tracebacks)
# is inspectable later without depending on scrollback or a manually redirected
# `nohup ... > file.log`. `tee` still prints to the terminal too (interactive
# use is unaffected); `set -o pipefail` (part of `set -euo pipefail` above)
# makes the script's own exit code the python process's, not tee's.
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "Full console log -> ${LOG_FILE}  (tail -f \"${LOG_FILE}\" to follow)"

# Per-sample logging output dir (recipe.transfusion.sample_logging writes
# Transfusion-style sample/eval artefacts under here, next to the ckpts). The
# number of samples is governed by trainer.log_val_generations in the YAML.
export TRANSFUSION_OUTPUT_DIR="${OUT_DIR}"

cd "${VERL_ROOT}"

# Worker-side env propagation (PYTHONPATH / TRANSFUSION_* / registration hook) is
# defined statically under `ray_kwargs.ray_init.runtime_env.env_vars` in the
# recipe config, so it survives across Ray workers without a fragile CLI dict.
# Only pass trainer.experiment_name override when EXP_NAME differs from the
# YAML's value (e.g. a smoke run); otherwise the YAML is the single source.
_yaml_exp="${_YAML_EXP}"
_cli_overrides=""
if [[ -n "${EXP_NAME}" && "${EXP_NAME}" != "${_yaml_exp}" ]]; then
  _cli_overrides="trainer.experiment_name=${EXP_NAME}"
fi
# Keep the RL config's world-size in sync with the detected cluster so verl
# builds the actor/rollout pools for exactly the nodes/GPUs that are present.
_cli_overrides="${_cli_overrides} trainer.nnodes=${NUM_NODES} trainer.n_gpus_per_node=${GPUS_PER_NODE}"
# Point verl's ray.init at the cluster this script just built/joined. The recipe
# config carries NO hardcoded address (so it works on any machine). Hydra needs the
# `+` append marker because `ray_init.address` does not already exist in the strict
# struct; a bare override would fail with "Key 'address' is not in struct".
_cli_overrides="${_cli_overrides} +ray_kwargs.ray_init.address=${MASTER_NUM_IP}:${MASTER_PORT}"
echo "[run_grpo] trainer CLI overrides: ${_cli_overrides}"
python -m recipe.transfusion.train \
  --config-dir "${VERL_ROOT}/recipe/transfusion/config" \
  --config-name "${CONFIG_NAME}" \
  ${_cli_overrides} \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
