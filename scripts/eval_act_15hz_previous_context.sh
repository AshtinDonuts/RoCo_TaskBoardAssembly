#!/usr/bin/env bash
# Evaluate the trained 15 Hz ACT policies with every earlier canonical part
# already placed at its configured successful endpoint.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_root="${1:-${repo_root}/artifacts/act_eval/act_15hz_all_prior_context_random10_temporal}"
mkdir -p "${out_root}"

export LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd "${repo_root}/.." && pwd)/lerobot_roco_pi05}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/roco-act-uv-cache}"
gcc_lib="/tmp/roco-gcc12-runtime/lib/x86_64-linux-gnu:/tmp/roco-gcc12-runtime/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="${gcc_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ACT_CUDA_VISIBLE_DEVICES="${ACT_CUDA_VISIBLE_DEVICES:-1}"
export ACT_NOMINAL="${ACT_NOMINAL:-0}"
export ACT_BLIND_XY="${ACT_BLIND_XY:-1}"
export ACT_EVAL_MAX_STEPS="${ACT_EVAL_MAX_STEPS:-100}"
export ACT_EVAL_MAX_PARTS=1
export ACT_TEMPORAL_ENSEMBLE_COEFF="${ACT_TEMPORAL_ENSEMBLE_COEFF:-0.01}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export TASK_ENABLE_CAMERA_VIEWPORTS="${TASK_ENABLE_CAMERA_VIEWPORTS:-0}"
# GPU 0 may be occupied by other long-running jobs; use the free A5000 by
# default for both Isaac Sim and the ACT sidecar unless explicitly overridden.
export ISAACSIM_ACTIVE_GPU="${ISAACSIM_ACTIVE_GPU:-1}"
export ISAACSIM_PHYSICS_GPU="${ISAACSIM_PHYSICS_GPU:-1}"

declare -a parts=(usb_a rod_16mm battery_size5 gear_60teeth)
declare -a checkpoints=(
  "${ACT_15HZ_USB_A_CKPT:-${repo_root}/artifacts/act_15hz_grid/roco_act_15hz_usb_a_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_ROD_16MM_CKPT:-${repo_root}/artifacts/act_15hz_grid/roco_act_15hz_rod_16mm_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_BATTERY_SIZE5_CKPT:-${repo_root}/artifacts/act_15hz_grid/roco_act_15hz_battery_size5_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_GEAR_60TEETH_CKPT:-${repo_root}/artifacts/act_15hz_grid/roco_act_15hz_gear_60teeth_chunk10_b32/checkpoints/008000/pretrained_model}"
)

for i in "${!parts[@]}"; do
  part="${parts[$i]}"
  checkpoint="${checkpoints[$i]}"
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Missing checkpoint for ${part}: ${checkpoint}" >&2
    exit 2
  fi
  for seed in $(seq "${ACT_SEED_FIRST:-0}" "${ACT_SEED_LAST:-9}"); do
    export ACT_RANDOM_SEED="${seed}"
    export ACT_EVAL_RESULTS="${out_root}/${part}_seed${seed}_results.json"
    export ACT_EVAL_VIDEO="${out_root}/${part}_seed${seed}_head.mp4"
    export ACT_SERVER_LOG="${out_root}/${part}_seed${seed}_server.log"
    export ACT_PREPLACE_PREVIOUS_SUCCESS=1
    echo "===== BEGIN part=${part} prior-task-context seed=${seed} max_steps=${ACT_EVAL_MAX_STEPS} TE=${ACT_TEMPORAL_ENSEMBLE_COEFF} ====="
    "${repo_root}/scripts/eval_act_roco.sh" "${checkpoint}" "${part}"
    echo "===== DONE part=${part} seed=${seed} ====="
  done
done
