#!/usr/bin/env bash
# Sequentially train per-part ACT on remaining 15 Hz subtasks.
# Split: 45 train / 5 eval pass episodes. Chunk/action: 10/10.
# Keep inference weights at steps 6000 and 8000 only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

LEROBOT_ROOT="${LEROBOT_ROOT:-/home/khw/lerobot}"
PY="${ACT_PYTHON:-${LEROBOT_ROOT}/.venv/bin/python}"
PATCHED="${script_dir}/lerobot_train_act_patched.py"
DATASET_ROOT="${ROCO_15HZ_DATASET_ROOT:-${repo_root}/derived_subtasks_15hz}"
MANIFEST="${DATASET_ROOT}/meta/roco_subtasks.jsonl"
OUT_ROOT="${ACT_OUTPUT_ROOT:-${repo_root}/artifacts/act_15hz}"
LOG_DIR="${repo_root}/artifacts/act_training_logs"
KEEP_STEPS=(6000 8000)

N_TRAIN="${ACT_N_TRAIN:-45}"
N_EVAL="${ACT_N_EVAL:-5}"
BATCH_SIZE="${ACT_BATCH_SIZE:-32}"
NUM_WORKERS="${ACT_NUM_WORKERS:-4}"
STEPS="${ACT_STEPS:-8000}"
# save_freq=6000 hits 6k; final step always saves → 8k
SAVE_FREQ="${ACT_SAVE_FREQ:-6000}"
CHUNK_SIZE="${ACT_CHUNK_SIZE:-10}"
N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-10}"

DEFAULT_PARTS=(hdmi battery_size1 pin bolt_8mm)
if [[ -n "${ACT_PARTS:-}" ]]; then
  read -r -a PARTS <<<"${ACT_PARTS}"
else
  PARTS=("${DEFAULT_PARTS[@]}")
fi

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

select_split() {
  local part="$1"
  local out_json="$2"
  "${PY}" - "${MANIFEST}" "${part}" "${N_TRAIN}" "${N_EVAL}" "${out_json}" <<'PY'
import json
import sys
from pathlib import Path

manifest, part = Path(sys.argv[1]), sys.argv[2]
n_train, n_eval = int(sys.argv[3]), int(sys.argv[4])
out_json = Path(sys.argv[5])
need = n_train + n_eval

eps = sorted({
    int(r["episode_index"])
    for line in manifest.open(encoding="utf-8")
    if line.strip()
    for r in [json.loads(line)]
    if r.get("part") == part and bool(r.get("pass"))
})
if len(eps) < need:
    raise SystemExit(
        f"part={part}: need {need} pass episodes for {n_train}/{n_eval} split; found {len(eps)}"
    )
chosen = eps[:need]
train_eps = chosen[:n_train]
eval_eps = chosen[n_train:]
payload = {
    "part": part,
    "n_train": n_train,
    "n_eval": n_eval,
    "train_episodes": train_eps,
    "eval_episodes": eval_eps,
    "pass_available": len(eps),
}
out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("[" + ",".join(map(str, train_eps)) + "]")
print(
    f"part={part}: train={len(train_eps)} eval={len(eval_eps)} "
    f"(from {len(eps)} pass; eval={eval_eps})",
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
    rm -rf "${d}/training_state"
    if [[ ! -d "${d}/pretrained_model" ]]; then
      echo "[prune] WARNING: missing pretrained_model in ${d}" >&2
    fi
  done
  if [[ -d "${ckpt_root}/008000/pretrained_model" ]]; then
    ln -sfn 008000 "${ckpt_root}/last"
  fi
}

train_one() {
  local part="$1"
  local job="roco_act_15hz_${part}_chunk${CHUNK_SIZE}_b${BATCH_SIZE}"
  local out_dir="${OUT_ROOT}/${job}"
  local log="${LOG_DIR}/${job}_$(date +%Y%m%d_%H%M%S).log"
  # Keep split JSON outside output_dir: LeRobot raises FileExistsError if
  # output_dir already exists when resume=False.
  local split_json="${LOG_DIR}/${job}_episode_split.json"
  local episodes

  echo "======== ${job} $(date) ========"
  rm -rf "${out_dir}"
  if [[ -e "${out_dir}" ]]; then
    echo "[train] FATAL: could not clear output_dir: ${out_dir}" >&2
    return 2
  fi
  episodes="$(select_split "${part}" "${split_json}")"
  if [[ -e "${out_dir}" ]]; then
    echo "[train] FATAL: output_dir appeared before trainer launch: ${out_dir}" >&2
    return 2
  fi

  set +e
  "${PY}" "${PATCHED}" \
    --dataset.repo_id="local/derived_subtasks_15hz_${part}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episodes="${episodes}" \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.chunk_size="${CHUNK_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.vision_backbone=resnet18 \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --seed=1000 \
    --eval_freq=0 \
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
  if [[ -f "${split_json}" ]]; then
    mkdir -p "${out_dir}"
    mv -f "${split_json}" "${out_dir}/episode_split.json"
  fi
  prune_to_inference_weights "${out_dir}"
  du -sh "${out_dir}" || true
}

echo "OUT_ROOT=${OUT_ROOT}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "PARTS=${PARTS[*]}"
echo "SPLIT=${N_TRAIN}/${N_EVAL} CHUNK=${CHUNK_SIZE}/${N_ACTION_STEPS} BATCH=${BATCH_SIZE} STEPS=${STEPS} KEEP=${KEEP_STEPS[*]}"

fail=0
for part in "${PARTS[@]}"; do
  if ! train_one "${part}"; then
    fail=1
    continue
  fi
done

echo "======== 15hz remaining complete $(date) fail=${fail} ========"
exit "${fail}"
