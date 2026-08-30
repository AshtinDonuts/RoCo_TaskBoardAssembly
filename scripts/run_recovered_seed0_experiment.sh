#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: $0 no_safe_retract_timeout120|settings35ec027_timeout120" >&2
  exit 2
fi

profile="$1"
case "${profile}" in
  no_safe_retract_timeout120|settings35ec027_timeout120) ;;
  *)
    echo "unknown profile: ${profile}" >&2
    exit 2
    ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
output_stem="${repo_root}/artifacts/randomized_rollouts/aware_seed0_${profile}"

cd "${repo_root}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export TASK_BASELINE_MOTION_PROFILE="${profile}"

uv run python task/run_pick_place.py \
  --policy policies.baseline_scripted.BaselinePolicy \
  --random-seed 0 \
  --record-video "${output_stem}.mp4" \
  --results-json "${output_stem}_results.json" \
  --trajectory-csv "${output_stem}_trajectory.csv"
