#!/usr/bin/env bash
# Train the approved 10 Hz ACT presets independently for both gear subtasks.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
dataset_root="${ROCO_10HZ_DATASET_ROOT:-${repo_root}/artifacts/joint_gears_seeds0-39/derived_subtasks_10hz}"
output_root="${ACT_OUTPUT_ROOT:-${repo_root}/artifacts/act_joint_gears_10hz}"
for part in gear_20teeth gear_60teeth; do
  ACT_PART="${part}" ACT_DATASET_ROOT="${dataset_root}" \
    ACT_CHUNK_SIZE=5 ACT_N_ACTION_STEPS=1 ACT_STEPS=8000 \
    ACT_BATCH_SIZE=32 ACT_NUM_WORKERS=4 ACT_SAVE_FREQ=1500 ACT_SEED=1000 \
    ACT_JOB_NAME="roco_act_${part}_10hz" \
    ACT_OUTPUT_DIR="${output_root}/roco_act_${part}_10hz" \
    "${script_dir}/train_act_roco.sh" "$@"
done
