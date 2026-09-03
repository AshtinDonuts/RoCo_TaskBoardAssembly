#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="${1:-${repo_root}/artifacts/act_joint_gears_10hz_save1500/roco_act_gear_20teeth_10hz}"
out_root="${2:-${repo_root}/artifacts/act_eval/joint_gear20_nominal_max150_temporal}"
mkdir -p "${out_root}"

export ACT_NOMINAL=1
export ACT_EVAL_MAX_STEPS=150
export ACT_EVAL_MAX_PARTS=1
# Temporal ensembling (LeRobot forces n_action_steps=1).
export ACT_TEMPORAL_ENSEMBLE_COEFF="${ACT_TEMPORAL_ENSEMBLE_COEFF:-0.01}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"

found=0
for checkpoint in "${run_root}"/checkpoints/[0-9]*/pretrained_model; do
  [[ -d "${checkpoint}" ]] || continue
  found=1
  step="$(basename "$(dirname "${checkpoint}")")"
  export ACT_EVAL_RESULTS="${out_root}/${step}_results.json"
  export ACT_EVAL_VIDEO="${out_root}/${step}_head.mp4"
  export ACT_SERVER_LOG="${out_root}/${step}_server.log"
  echo "===== BEGIN gear_20teeth checkpoint=${step} nominal max_steps=150 TE=${ACT_TEMPORAL_ENSEMBLE_COEFF} ====="
  if ! "${repo_root}/scripts/eval_act_roco.sh" "${checkpoint}" gear_20teeth; then
    echo "===== FAIL checkpoint=${step} ====="
    continue
  fi
  echo "===== DONE checkpoint=${step} ====="
done

if [[ "${found}" == "0" ]]; then
  echo "No checkpoints found under ${run_root}/checkpoints" >&2
  exit 2
fi
