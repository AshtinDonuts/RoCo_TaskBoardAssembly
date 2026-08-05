#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="$(cd "${repo_root}/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"

ckpt="${1:-${PI05_CKPT:-}}"
if [[ -z "${ckpt}" ]]; then
  echo "usage: $0 /path/to/checkpoint/pretrained_model" >&2
  exit 2
fi
if [[ ! -d "${ckpt}" ]]; then
  echo "error: checkpoint dir not found: ${ckpt}" >&2
  exit 1
fi

export PI05_CKPT="${ckpt}"
export PI05_SERVER_PY="${PI05_SERVER_PY:-${lerobot_root}/.venv/bin/python}"
export PI05_CUDA_VISIBLE_DEVICES="${PI05_CUDA_VISIBLE_DEVICES:-1}"
export PI05_DEVICE="${PI05_DEVICE:-cuda}"
export HF_HOME="${HF_HOME:-${models_root}/.hf-cache}"
if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(<"${HF_HOME}/token")"
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PI05_ACTION_ROTATION="${PI05_ACTION_ROTATION:-euler_xyz}"
export ISAACSIM_RENDER_QUALITY="${PI05_RENDER_QUALITY:-${ISAACSIM_RENDER_QUALITY:-default}}"

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${PI05_EVAL_DIR:-${repo_root}/artifacts/pi05_eval_${stamp}}"
mkdir -p "${out_dir}"

extra_args=()
if [[ -n "${PI05_EVAL_MAX_STEPS:-}" ]]; then
  extra_args+=(--max-steps "${PI05_EVAL_MAX_STEPS}")
fi
if [[ -n "${PI05_EVAL_MAX_SIM_SECONDS:-}" ]]; then
  extra_args+=(--max-sim-seconds "${PI05_EVAL_MAX_SIM_SECONDS}")
fi
if [[ -n "${PI05_EVAL_MAX_PARTS:-}" ]]; then
  extra_args+=(--max-parts "${PI05_EVAL_MAX_PARTS}")
fi

cd "${repo_root}"
echo "[eval] ckpt=${PI05_CKPT}"
echo "[eval] server_py=${PI05_SERVER_PY}"
echo "[eval] out=${out_dir}"
echo "[eval] rotation=${PI05_ACTION_ROTATION} render_quality=${ISAACSIM_RENDER_QUALITY}"
exec ./scripts/run_roco.sh \
  --policy policies.pi05_lerobot.Pi05LeRobotPolicy \
  --record-video "${PI05_EVAL_VIDEO:-${out_dir}/head.mp4}" \
  --record-video-camera "${PI05_EVAL_CAMERA:-head}" \
  --record-video-fps "${PI05_EVAL_FPS:-10}" \
  --results-json "${PI05_EVAL_RESULTS:-${out_dir}/results.json}" \
  "${extra_args[@]}"
