#!/usr/bin/env bash
# Build a frame-exact 10 Hz derivative of the seed 0-99 per-part dataset.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

source_root="${ROCO_30HZ_DATASET_ROOT:-${repo_root}/artifacts/aware_35ec027_timeout120_seeds0-99/derived_subtasks}"
output_root="${ROCO_10HZ_DATASET_ROOT:-${repo_root}/artifacts/aware_35ec027_timeout120_seeds0-99/derived_subtasks_10hz}"

args=("${source_root}" "${output_root}" --fps 10)
if [[ "${ROCO_10HZ_ALL_PARTS:-0}" == "1" ]]; then
  echo "Creating 10 Hz dataset for all parts"
else
  args+=(--part "${ROCO_10HZ_PART:-gear_60teeth}")
  echo "Creating 10 Hz prototype for ${ROCO_10HZ_PART:-gear_60teeth}"
fi
if [[ "${ROCO_10HZ_REPLACE:-0}" == "1" ]]; then
  args+=(--replace)
fi

exec uv run --group collection python "${repo_root}/downsample_lerobot_dataset.py" "${args[@]}" "$@"
