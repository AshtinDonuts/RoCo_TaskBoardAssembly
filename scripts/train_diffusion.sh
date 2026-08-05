#!/usr/bin/env bash
# Train LeRobot Diffusion Policy on the pinned RoCo Industrial Assembly dataset.
# Usage:
#   ./scripts/train_diffusion.sh smoke
#   ./scripts/train_diffusion.sh production
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${LEROBOT_VENV:-${repo_root}/.venv_lerobot}"
python_bin="${DP_SERVER_PY:-${venv_dir}/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "error: missing ${python_bin}; run ./scripts/setup_lerobot_env.sh first" >&2
  exit 1
fi

mode="${1:-smoke}"
repo_id="rocochallenge2025/rocochallenge2026_Industrial_Assembly"
revision="dc03b003f94d184b2b20465ed986456ee1bf2a3c"
# Prefer the already-downloaded local cache when present.
default_root="${repo_root}/.hf-cache/lerobot/${repo_id}"
dataset_root="${DP_DATASET_ROOT:-}"
if [[ -z "${dataset_root}" && -f "${default_root}/meta/info.json" ]]; then
  dataset_root="${default_root}"
fi
output_root="${DP_OUTPUT_DIR:-${repo_root}/outputs/diffusion}"
job_name="roco_dp_${mode}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0}"

# Defaults tuned for ~24 GB; A6000 can raise DP_BATCH_SIZE.
batch_size="${DP_BATCH_SIZE:-48}"
num_workers="${DP_NUM_WORKERS:-4}"
seed="${DP_SEED:-1000}"

case "${mode}" in
  smoke)
    steps="${DP_STEPS:-200}"
    save_freq="${DP_SAVE_FREQ:-100}"
    log_freq="${DP_LOG_FREQ:-20}"
    episodes_arg=(--dataset.episodes="[0,1,2]")
    ;;
  production)
    # ~12 h wall-clock at ~2.2 step/s (batch 64, 3×240×320, pyav).
    # Override with DP_STEPS / DP_BATCH_SIZE / DP_SAVE_FREQ as needed.
    steps="${DP_STEPS:-95000}"
    save_freq="${DP_SAVE_FREQ:-10000}"
    log_freq="${DP_LOG_FREQ:-200}"
    episodes_arg=()
    ;;
  *)
    echo "usage: $0 {smoke|production}" >&2
    exit 2
    ;;
esac

stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${output_root}/${mode}_${stamp}"
# Do not pre-create output_dir: lerobot refuses non-empty/existing dirs unless resume=true.
mkdir -p "${output_root}"

root_args=()
if [[ -n "${dataset_root}" ]]; then
  root_args=(--dataset.root="${dataset_root}")
fi

export CUDA_VISIBLE_DEVICES="${cuda_devices}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="${HF_HOME:-${repo_root}/.hf-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${repo_root}/.hf-cache/lerobot}"
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(<"${HOME}/.cache/huggingface/token")"
fi
# Local-only logging; no Hub / W&B credentials required.
unset WANDB_API_KEY || true

echo "[train] mode=${mode} steps=${steps} batch=${batch_size} out=${output_dir} gpu=${cuda_devices}"
echo "[train] dataset_root=${dataset_root:-<hub>}"

cd "${repo_root}"
exec "${python_bin}" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${repo_id}" \
  --dataset.revision="${revision}" \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  "${root_args[@]}" \
  "${episodes_arg[@]}" \
  --policy.type=diffusion \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=4 \
  --policy.drop_n_last_frames=11 \
  --policy.vision_backbone=resnet18 \
  --policy.push_to_hub=false \
  --batch_size="${batch_size}" \
  --steps="${steps}" \
  --save_freq="${save_freq}" \
  --log_freq="${log_freq}" \
  --eval_freq=0 \
  --num_workers="${num_workers}" \
  --seed="${seed}" \
  --job_name="${job_name}" \
  --output_dir="${output_dir}" \
  --wandb.enable=false \
  --save_checkpoint=true
