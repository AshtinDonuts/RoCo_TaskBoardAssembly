#!/usr/bin/env bash
# Run isolated ACT subtask evals with fairness XY randomization.
# Each job uses ROCO_PART_ORDER=<part> so only that skill is executed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export LEROBOT_ROOT="${LEROBOT_ROOT:-/home/khw/lerobot}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export ISAACSIM_ACTIVE_GPU="${ISAACSIM_ACTIVE_GPU:-0}"
export ISAACSIM_PHYSICS_GPU="${ISAACSIM_PHYSICS_GPU:-0}"
export ACT_CUDA_VISIBLE_DEVICES="${ACT_CUDA_VISIBLE_DEVICES:-0}"
export ACT_BLIND_XY="${ACT_BLIND_XY:-1}"
export ACT_EVAL_MAX_PARTS="${ACT_EVAL_MAX_PARTS:-1}"
export ACT_SERVER_PY="${ACT_SERVER_PY:-${LEROBOT_ROOT}/.venv/bin/python}"

outdir="${repo_root}/artifacts/act_eval"
mkdir -p "${outdir}"

jobs=(
  "usb_a|${LEROBOT_ROOT}/outputs/roco_act_aware0-99_usb_a_fast_20260831_150726/checkpoints/005000/pretrained_model|101"
  "battery_size1|${LEROBOT_ROOT}/outputs/roco_act_aware0-99_battery_size1_fast_20260831_162142/checkpoints/005000/pretrained_model|102"
  "gear_60teeth|${LEROBOT_ROOT}/outputs/roco_act_aware0-99_gear_60teeth_fast_20260831_162142/checkpoints/005000/pretrained_model|103"
)

# Skip parts already finished if SKIP_DONE=1
skip_done="${SKIP_DONE:-0}"

for job in "${jobs[@]}"; do
  IFS='|' read -r part ckpt seed <<<"${job}"
  results="${outdir}/${part}_seed${seed}_results.json"
  if [[ "${skip_done}" == "1" && -f "${results}" ]]; then
    echo "===== SKIP ${part} (results exist) ====="
    continue
  fi
  echo "===== BEGIN ${part} seed=${seed} ====="
  export ACT_RANDOM_SEED="${seed}"
  export ACT_EVAL_RESULTS="${results}"
  export ACT_EVAL_VIDEO="${outdir}/${part}_seed${seed}_head.mp4"
  export ACT_SERVER_LOG="${outdir}/${part}_seed${seed}_server.log"
  if ! ./scripts/eval_act_roco.sh "${ckpt}" "${part}"; then
    echo "===== FAIL ${part} ====="
    continue
  fi
  echo "===== DONE ${part} ====="
done

echo ALL_EVALS_FINISHED
