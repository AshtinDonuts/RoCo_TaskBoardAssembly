#!/usr/bin/env bash
# Sequentially train ACT on the 10 Hz all-parts subtask dataset.
# Grid: chunk_size ∈ {3,5} × 9 parts, 20k steps, keep weights at 8k/14k/20k only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

LEROBOT_ROOT="${LEROBOT_ROOT:-/home/khw/lerobot}"
PY="${ACT_PYTHON:-${LEROBOT_ROOT}/.venv/bin/python}"
PATCHED="${script_dir}/lerobot_train_act_patched.py"
DATASET_ROOT="${ROCO_10HZ_DATASET_ROOT:-${repo_root}/artifacts/aware_35ec027_timeout120_seeds0-99/derived_subtasks_10hz}"
MANIFEST="${DATASET_ROOT}/meta/roco_subtasks.jsonl"
OUT_ROOT="${ACT_OUTPUT_ROOT:-${repo_root}/artifacts/act_10hz_grid}"
LOG_DIR="${repo_root}/artifacts/act_training_logs"
KEEP_STEPS=(8000 14000 20000)
MAX_PASS_EPS="${ACT_MAX_PASS_EPS:-60}"
BATCH_SIZE="${ACT_BATCH_SIZE:-32}"
NUM_WORKERS="${ACT_NUM_WORKERS:-4}"
STEPS="${ACT_STEPS:-20000}"
# save_freq=2000 hits 8k/14k/20k; extras pruned after each run
SAVE_FREQ="${ACT_SAVE_FREQ:-2000}"
N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-1}"

DEFAULT_PARTS=(
  gear_60teeth gear_20teeth rod_16mm bolt_8mm usb_a hdmi pin
  battery_size1 battery_size5
)
DEFAULT_CHUNKS=(3 5)

# Override with whitespace-separated lists to split the grid across machines:
#   ACT_PARTS="gear_60teeth rod_16mm" ACT_CHUNKS="5" ./scripts/train_act_10hz_all_parts_grid.sh
if [[ -n "${ACT_PARTS:-}" ]]; then
  read -r -a PARTS <<<"${ACT_PARTS}"
else
  PARTS=("${DEFAULT_PARTS[@]}")
fi
if [[ -n "${ACT_CHUNKS:-}" ]]; then
  read -r -a CHUNKS <<<"${ACT_CHUNKS}"
else
  CHUNKS=("${DEFAULT_CHUNKS[@]}")
fi

if [[ "${#PARTS[@]}" -eq 0 || "${#CHUNKS[@]}" -eq 0 ]]; then
  echo "ACT_PARTS and ACT_CHUNKS must each select at least one value" >&2
  exit 2
fi

declare -A valid_parts=()
declare -A seen_parts=()
for part in "${DEFAULT_PARTS[@]}"; do
  valid_parts["${part}"]=1
done
for part in "${PARTS[@]}"; do
  if [[ -z "${valid_parts[${part}]+x}" ]]; then
    echo "Unknown part in ACT_PARTS: ${part}" >&2
    echo "Valid parts: ${DEFAULT_PARTS[*]}" >&2
    exit 2
  fi
  if [[ -n "${seen_parts[${part}]+x}" ]]; then
    echo "Duplicate part in ACT_PARTS: ${part}" >&2
    exit 2
  fi
  seen_parts["${part}"]=1
done

declare -A seen_chunks=()
for chunk in "${CHUNKS[@]}"; do
  if [[ "${chunk}" != "3" && "${chunk}" != "5" ]]; then
    echo "Unknown chunk size in ACT_CHUNKS: ${chunk}; valid values: 3 5" >&2
    exit 2
  fi
  if [[ -n "${seen_chunks[${chunk}]+x}" ]]; then
    echo "Duplicate chunk size in ACT_CHUNKS: ${chunk}" >&2
    exit 2
  fi
  seen_chunks["${chunk}"]=1
done

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${ACT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf-cache}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}/torch}"
export PYTHONPATH="${LEROBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

if [[ ! -x "${PY}" ]]; then
  echo "LeRobot python not found: ${PY}" >&2
  exit 2
fi
if [[ ! -f "${PATCHED}" ]]; then
  echo "Patched trainer missing: ${PATCHED}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest missing: ${MANIFEST}" >&2
  exit 2
fi

select_episodes() {
  local part="$1"
  "${PY}" - "${MANIFEST}" "${part}" "${MAX_PASS_EPS}" <<'PY'
import json, sys
manifest, part, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
eps = sorted({
    int(r["episode_index"])
    for line in open(manifest) if line.strip()
    for r in [json.loads(line)]
    if r.get("part") == part and bool(r.get("pass"))
})
if not eps:
    raise SystemExit(f"no pass episodes for part={part}")
chosen = eps[:limit]
print("[" + ",".join(map(str, chosen)) + "]")
print(
    f"part={part}: using {len(chosen)}/{len(eps)} pass episodes (cap={limit})",
    file=sys.stderr,
)
PY
}

prune_to_inference_weights() {
  local out_dir="$1"
  local ckpt_root="${out_dir}/checkpoints"
  [[ -d "${ckpt_root}" ]] || return 0
  local keep
  declare -A keep_set=()
  for keep in "${KEEP_STEPS[@]}"; do
    keep_set["$(printf '%06d' "${keep}")"]=1
  done
  local d name
  for d in "${ckpt_root}"/*; do
    [[ -e "${d}" ]] || continue
    name="$(basename "${d}")"
    if [[ "${name}" == "last" ]]; then
      rm -rf "${d}"
      continue
    fi
    if [[ -z "${keep_set[${name}]+x}" ]]; then
      echo "[prune] remove ${d}"
      rm -rf "${d}"
      continue
    fi
    # Keep deployable weights only.
    rm -rf "${d}/training_state"
    if [[ ! -d "${d}/pretrained_model" ]]; then
      echo "[prune] WARNING: missing pretrained_model in ${d}" >&2
    fi
  done
  # Point last at final kept step if present.
  if [[ -d "${ckpt_root}/020000/pretrained_model" ]]; then
    ln -sfn 020000 "${ckpt_root}/last"
  fi
}

train_one() {
  local part="$1"
  local chunk="$2"
  local job="roco_act_10hz_${part}_chunk${chunk}_b${BATCH_SIZE}"
  local out_dir="${OUT_ROOT}/${job}"
  local log="${LOG_DIR}/${job}_$(date +%Y%m%d_%H%M%S).log"
  local episodes

  echo "======== ${job} $(date) ========"
  episodes="$(select_episodes "${part}")"

  # LeRobot refuses to start if output_dir already exists (resume=False).
  # Remove any prior tree and let the trainer create the directory.
  rm -rf "${out_dir}"

  set +e
  "${PY}" "${PATCHED}" \
    --dataset.repo_id="local/aware_35ec027_timeout120_seeds0-99_10hz_${part}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${episodes}" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.chunk_size="${chunk}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --env_eval_freq=0 \
    --save_freq="${SAVE_FREQ}" \
    --log_freq=100 \
    --wandb.enable=false \
    --wandb.project=roco \
    --job_name="${job}" \
    --output_dir="${out_dir}" \
    >"${log}" 2>&1
  local ec=$?
  set -e
  echo "[train] ${job} exit=${ec} log=${log}"
  if [[ "${ec}" -ne 0 ]]; then
    echo "[train] FAILED ${job}; see ${log}" >&2
    tail -n 40 "${log}" >&2 || true
    return "${ec}"
  fi
  prune_to_inference_weights "${out_dir}"
  du -sh "${out_dir}" || true
}

echo "OUT_ROOT=${OUT_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PARTS=${PARTS[*]} CHUNKS=${CHUNKS[*]}"
echo "KEEP_STEPS=${KEEP_STEPS[*]} STEPS=${STEPS} BATCH=${BATCH_SIZE} MAX_PASS_EPS=${MAX_PASS_EPS}"

fail=0
for chunk in "${CHUNKS[@]}"; do
  for part in "${PARTS[@]}"; do
    if ! train_one "${part}" "${chunk}"; then
      fail=1
      # Continue remaining jobs so one failure does not strand the grid.
      continue
    fi
  done
done

echo "======== grid complete $(date) fail=${fail} ========"
exit "${fail}"
