#!/usr/bin/env bash
# Evaluate a Diffusion Policy checkpoint in the RoCo Isaac Sim harness.
# Mirrors scripts/eval_pi05_roco.sh with camera recording enabled.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${LEROBOT_VENV:-${repo_root}/.venv_lerobot}"

ckpt="${1:-${DP_CKPT:-}}"
if [[ -z "${ckpt}" ]]; then
  echo "usage: $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
if [[ ! -d "${ckpt}" ]]; then
  echo "error: checkpoint dir not found: ${ckpt}" >&2
  exit 1
fi

export DP_CKPT="${ckpt}"
export DP_SERVER_PY="${DP_SERVER_PY:-${venv_dir}/bin/python}"
export CUDA_VISIBLE_DEVICES="${DP_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export ISAACSIM_RENDER_QUALITY="${DP_RENDER_QUALITY:-${ISAACSIM_RENDER_QUALITY:-default}}"

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${DP_EVAL_DIR:-${repo_root}/artifacts/dp_eval_${stamp}}"
mkdir -p "${out_dir}"

extra_args=()
if [[ -n "${DP_EVAL_MAX_STEPS:-}" ]]; then
  extra_args+=(--max-steps "${DP_EVAL_MAX_STEPS}")
fi
if [[ -n "${DP_EVAL_MAX_SIM_SECONDS:-}" ]]; then
  extra_args+=(--max-sim-seconds "${DP_EVAL_MAX_SIM_SECONDS}")
fi
if [[ -n "${DP_EVAL_MAX_PARTS:-}" ]]; then
  extra_args+=(--max-parts "${DP_EVAL_MAX_PARTS}")
fi

echo "[eval] ckpt=${DP_CKPT}"
echo "[eval] server_py=${DP_SERVER_PY}"
echo "[eval] out=${out_dir}"
echo "[eval] render_quality=${ISAACSIM_RENDER_QUALITY}"

cd "${repo_root}"
# --record-video enables camera sensor binding even when enable_camera_output=False.
exec ./scripts/run_roco.sh \
  --policy policies.diffusion_lerobot.DiffusionLeRobotPolicy \
  --record-video "${DP_EVAL_VIDEO:-${out_dir}/head.mp4}" \
  --record-video-camera "${DP_EVAL_CAMERA:-head}" \
  --record-video-fps "${DP_EVAL_FPS:-10}" \
  --results-json "${DP_EVAL_RESULTS:-${out_dir}/results.json}" \
  "${extra_args[@]}"
