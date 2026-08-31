#!/usr/bin/env bash
# Train the two 10 Hz ACT prototypes for gear_60teeth, sequentially.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
dataset_root="${ROCO_10HZ_DATASET_ROOT:-${repo_root}/artifacts/aware_35ec027_timeout120_seeds0-99/derived_subtasks_10hz}"
variant="${1:-all}"
if (( $# > 0 )); then shift; fi

train_variant() {
  local chunk_size="$1"
  shift
  ACT_PART=gear_60teeth \
  ACT_DATASET_ROOT="${dataset_root}" \
  ACT_DATASET_REPO_ID="local/aware_35ec027_timeout120_seeds0-99_gear60_10hz" \
  ACT_CHUNK_SIZE="${chunk_size}" \
  ACT_N_ACTION_STEPS=1 \
  ACT_JOB_NAME="roco_act_gear60_10hz_chunk${chunk_size}_closedloop" \
  ACT_OUTPUT_DIR="${ACT_OUTPUT_ROOT:-${LEROBOT_ROOT:-${repo_root}/../lerobot_roco_pi05}/outputs}/roco_act_gear60_10hz_chunk${chunk_size}_closedloop" \
    "${script_dir}/train_act_roco.sh" "$@"
}

case "${variant}" in
  3) train_variant 3 "$@" ;;
  5) train_variant 5 "$@" ;;
  all)
    train_variant 3 "$@"
    train_variant 5 "$@"
    ;;
  *) echo "usage: $0 [3|5|all] [train_act_roco.sh overrides]" >&2; exit 2 ;;
esac
