#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="$(cd "${repo_root}/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"

export HF_HOME="${HF_HOME:-${models_root}/.hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${models_root}/.uv-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

steps="${PI05_STEPS:-3000}"
batch_size="${PI05_BATCH_SIZE:-1}"
output_dir="${PI05_OUTPUT_DIR:-${lerobot_root}/outputs/roco_pi05_ft}"
save_freq="${PI05_SAVE_FREQ:-${steps}}"
log_freq="${PI05_LOG_FREQ:-10}"
num_workers="${PI05_NUM_WORKERS:-0}"
job_name="${PI05_JOB_NAME:-roco_pi05_ft}"
pretrained_path="${PI05_PRETRAINED_PATH:-lerobot/pi05_base}"
if [[ "${pretrained_path}" == "lerobot/pi05_base" ]]; then
  pretrained_revision="${PI05_PRETRAINED_REVISION:-7de663972b7817d2c4cf2d84c821153dfea772e9}"
else
  pretrained_revision="${PI05_PRETRAINED_REVISION:-}"
fi
dataset_repo_id="${PI05_DATASET_REPO_ID:-rocochallenge2025/rocochallenge2026_Industrial_Assembly}"
dataset_revision="${PI05_DATASET_REVISION:-dc03b003f94d184b2b20465ed986456ee1bf2a3c}"
train_expert_only="${PI05_TRAIN_EXPERT_ONLY:-true}"
freeze_vision_encoder="${PI05_FREEZE_VISION_ENCODER:-true}"
accelerate_mode="${PI05_ACCELERATE_MODE:-single}"
accelerate_config="${PI05_ACCELERATE_CONFIG:-${repo_root}/scripts/accelerate_pi05_fsdp_3gpu.yaml}"
num_processes="${PI05_NUM_PROCESSES:-}"
gpu_ids="${PI05_GPU_IDS:-}"

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(<"${HF_HOME}/token")"
fi
if [[ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}" ]]; then
  echo "HF_TOKEN or ${HF_HOME}/token is required for gated google/paligemma-3b-pt-224 access." >&2
  exit 2
fi

cd "${lerobot_root}"
train_args=(
  --dataset.repo_id="${dataset_repo_id}" \
  --dataset.revision="${dataset_revision}" \
  --dataset.video_backend=pyav \
  --policy.type=pi05 \
  --policy.pretrained_path="${pretrained_path}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.train_expert_only="${train_expert_only}" \
  --policy.freeze_vision_encoder="${freeze_vision_encoder}" \
  --policy.max_state_dim=44 \
  --policy.max_action_dim=32 \
  --output_dir="${output_dir}" \
  --job_name="${job_name}" \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --env_eval_freq=0 \
  --steps="${steps}" \
  --save_freq="${save_freq}" \
  --log_freq="${log_freq}" \
  --batch_size="${batch_size}" \
  --num_workers="${num_workers}"
)
if [[ -n "${pretrained_revision}" ]]; then
  train_args+=(--policy.pretrained_revision="${pretrained_revision}")
fi

case "${accelerate_mode}" in
  single)
    exec uv run python -m lerobot.scripts.lerobot_train "${train_args[@]}"
    ;;
  ddp)
    launcher=(accelerate launch --multi_gpu)
    if [[ -n "${num_processes}" ]]; then
      launcher+=(--num_processes "${num_processes}")
    fi
    if [[ -n "${gpu_ids}" ]]; then
      launcher+=(--gpu_ids "${gpu_ids}")
    fi
    exec uv run "${launcher[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}"
    ;;
  fsdp)
    export LEROBOT_PI05_FSDP_UNIFORM_DTYPE="${LEROBOT_PI05_FSDP_UNIFORM_DTYPE:-1}"
    launcher=(accelerate launch --config_file "${accelerate_config}")
    if [[ -n "${num_processes}" ]]; then
      launcher+=(--num_processes "${num_processes}")
    fi
    if [[ -n "${gpu_ids}" ]]; then
      launcher+=(--gpu_ids "${gpu_ids}")
    fi
    exec uv run "${launcher[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}"
    ;;
  *)
    echo "Unsupported PI05_ACCELERATE_MODE=${accelerate_mode}; use single, ddp, or fsdp." >&2
    exit 2
    ;;
esac
