#!/usr/bin/env bash
# Evaluate the trained 15 Hz ACT policies with their immediate predecessor
# already placed at its configured successful endpoint.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_root="${1:-${repo_root}/artifacts/act_eval/act_15hz_previous_context_random10_temporal}"
mkdir -p "${out_root}"

export ACT_NOMINAL="${ACT_NOMINAL:-0}"
export ACT_BLIND_XY="${ACT_BLIND_XY:-1}"
export ACT_PREPLACE_PREVIOUS_SUCCESS=1
export ACT_EVAL_MAX_STEPS="${ACT_EVAL_MAX_STEPS:-100}"
export ACT_EVAL_MAX_PARTS=1
export ACT_TEMPORAL_ENSEMBLE_COEFF="${ACT_TEMPORAL_ENSEMBLE_COEFF:-0.01}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"

declare -a parts=(hdmi battery_size1 pin bolt_8mm)
declare -a checkpoints=(
  "${ACT_15HZ_HDMI_CKPT:-${repo_root}/artifacts/act_15hz/roco_act_15hz_hdmi_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_BATTERY_SIZE1_CKPT:-${repo_root}/artifacts/act_15hz/roco_act_15hz_battery_size1_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_PIN_CKPT:-${repo_root}/artifacts/act_15hz/roco_act_15hz_pin_chunk10_b32/checkpoints/008000/pretrained_model}"
  "${ACT_15HZ_BOLT_8MM_CKPT:-${repo_root}/artifacts/act_15hz/roco_act_15hz_bolt_8mm_chunk10_b32/checkpoints/008000/pretrained_model}"
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
    echo "===== BEGIN part=${part} predecessor-context seed=${seed} max_steps=${ACT_EVAL_MAX_STEPS} TE=${ACT_TEMPORAL_ENSEMBLE_COEFF} ====="
    "${repo_root}/scripts/eval_act_roco.sh" "${checkpoint}" "${part}"
    echo "===== DONE part=${part} seed=${seed} ====="
  done
done
