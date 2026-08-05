#!/usr/bin/env bash
# Fine-tune LeRobot pi0.5 on the pinned RoCo Industrial Assembly dataset.
# Usage: ./scripts/train_pi05.sh {smoke|production} {full|lora}
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="$(cd "${repo_root}/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"
python_bin="${PI05_SERVER_PY:-${lerobot_root}/.venv/bin/python}"
train_bin="${lerobot_root}/.venv/bin/lerobot-train"
accelerate_bin="${lerobot_root}/.venv/bin/accelerate"

if [[ ! -x "${python_bin}" || ! -x "${train_bin}" ]]; then
  echo "error: pi0.5 environment missing; run ./scripts/setup_pi05_env.sh" >&2
  exit 1
fi

mode="${1:-smoke}"
strategy="${2:-lora}"
repo_id="rocochallenge2025/rocochallenge2026_Industrial_Assembly"
revision="dc03b003f94d184b2b20465ed986456ee1bf2a3c"
base_model="${PI05_BASE_MODEL:-lerobot/pi05_base}"
dataset_root="${PI05_DATASET_ROOT:-}"
default_root="${repo_root}/.hf-cache/lerobot/${repo_id}"
if [[ -z "${dataset_root}" && -f "${default_root}/meta/info.json" ]]; then
  dataset_root="${default_root}"
fi

case "${mode}" in
  smoke)
    steps="${PI05_STEPS:-20}"
    save_freq="${PI05_SAVE_FREQ:-10}"
    log_freq="${PI05_LOG_FREQ:-1}"
    episodes_arg=(--dataset.episodes="[0,1,2]")
    ;;
  production)
    steps="${PI05_STEPS:-10000}"
    save_freq="${PI05_SAVE_FREQ:-1000}"
    log_freq="${PI05_LOG_FREQ:-50}"
    episodes_arg=()
    ;;
  *)
    echo "usage: $0 {smoke|production} {full|lora}" >&2
    exit 2
    ;;
esac

case "${strategy}" in
  full)
    batch_size="${PI05_BATCH_SIZE:-2}"
    lr="${PI05_LR:-2.5e-5}"
    decay_lr="${PI05_DECAY_LR:-2.5e-6}"
    num_processes="${PI05_NUM_PROCESSES:-2}"
    peft_args=()
    ;;
  lora)
    batch_size="${PI05_BATCH_SIZE:-4}"
    lr="${PI05_LR:-2.5e-4}"
    decay_lr="${PI05_DECAY_LR:-2.5e-5}"
    num_processes="${PI05_NUM_PROCESSES:-1}"
    peft_args=(
      --peft.method_type=LORA
      --peft.r="${PI05_LORA_R:-16}"
      --peft.lora_alpha="${PI05_LORA_ALPHA:-16}"
    )
    ;;
  *)
    echo "usage: $0 {smoke|production} {full|lora}" >&2
    exit 2
    ;;
esac

num_workers="${PI05_NUM_WORKERS:-4}"
seed="${PI05_SEED:-1000}"
normalization="${PI05_NORMALIZATION:-quantiles}"
case "${normalization}" in
  quantiles)
    normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}'
    ;;
  mean_std)
    normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
    ;;
  *)
    echo "error: PI05_NORMALIZATION must be quantiles or mean_std" >&2
    exit 2
    ;;
esac
warmup_steps="${PI05_WARMUP_STEPS:-$((steps < 1000 ? (steps / 30 + 1) : 1000))}"
decay_steps="${PI05_DECAY_STEPS:-${steps}}"
output_root="${PI05_OUTPUT_DIR:-${repo_root}/outputs/pi05}"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${output_root}/${strategy}_${mode}_${stamp}"
mkdir -p "${output_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${repo_root}/.hf-cache/lerobot}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
unset WANDB_API_KEY || true
if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(<"${HF_HOME}/token")"
fi
if [[ "${PI05_SKIP_ACCESS_CHECK:-0}" != "1" ]]; then
  if ! "${python_bin}" - <<'PY'
from huggingface_hub import hf_hub_download

for repo_id in ("lerobot/pi05_base", "google/paligemma-3b-pt-224"):
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception as exc:
        raise SystemExit(
            f"error: cannot access {repo_id}; accept its Hugging Face terms "
            f"and set HF_TOKEN ({type(exc).__name__}: {exc})"
        ) from exc
print("[train] Hugging Face model access OK")
PY
  then
    exit 1
  fi
fi

root_args=()
if [[ -n "${dataset_root}" ]]; then
  root_args=(--dataset.root="${dataset_root}")
fi

resume_args=()
policy_args=(
  --policy.path="${base_model}"
  --policy.input_features=null
  --policy.output_features=null
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.max_state_dim=44
  --policy.chunk_size=50
  --policy.n_action_steps="${PI05_N_ACTION_STEPS:-10}"
  --policy.normalization_mapping="${normalization_mapping}"
  --policy.optimizer_lr="${lr}"
  --policy.scheduler_decay_lr="${decay_lr}"
  --policy.scheduler_warmup_steps="${warmup_steps}"
  --policy.scheduler_decay_steps="${decay_steps}"
  --policy.push_to_hub=false
)
if [[ -n "${PI05_RESUME_CONFIG:-}" ]]; then
  resume_args=(--resume=true --config_path="${PI05_RESUME_CONFIG}")
  policy_args=()
fi

train_args=(
  --dataset.repo_id="${repo_id}"
  --dataset.revision="${revision}"
  --dataset.use_imagenet_stats=false
  --dataset.video_backend=pyav
  "${root_args[@]}"
  "${episodes_arg[@]}"
  "${policy_args[@]}"
  "${peft_args[@]}"
  "${resume_args[@]}"
  --batch_size="${batch_size}"
  --steps="${steps}"
  --save_freq="${save_freq}"
  --log_freq="${log_freq}"
  --env_eval_freq=0
  --num_workers="${num_workers}"
  --seed="${seed}"
  --job_name="roco_pi05_${strategy}_${mode}"
  --output_dir="${output_dir}"
  --wandb.enable=false
  --save_checkpoint=true
)

launcher=("${train_bin}")
if ((num_processes > 1)); then
  launcher=(
    "${accelerate_bin}" launch
    --multi_gpu
    --num_processes="${num_processes}"
    --mixed_precision=bf16
    "${train_bin}"
  )
fi

launch_record="${output_root}/.${strategy}_${mode}_${stamp}.launch.txt"
{
  echo "date=$(date --iso-8601=seconds)"
  echo "repo_root=${repo_root}"
  echo "lerobot_root=${lerobot_root}"
  echo "lerobot_revision=$(git -C "${lerobot_root}" rev-parse HEAD)"
  echo "dataset=${repo_id}@${revision}"
  echo "strategy=${strategy}"
  echo "per_device_batch_size=${batch_size}"
  echo "num_processes=${num_processes}"
  echo "effective_batch_size=$((batch_size * num_processes))"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  printf "command="
  printf "%q " "${launcher[@]}" "${train_args[@]}"
  printf "\n"
} >"${launch_record}"

copy_launch_record() {
  if [[ -d "${output_dir}" && -f "${launch_record}" ]]; then
    cp "${launch_record}" "${output_dir}/launch.txt"
  fi
}
trap copy_launch_record EXIT

echo "[train] strategy=${strategy} mode=${mode} steps=${steps}"
echo "[train] batch_per_device=${batch_size} processes=${num_processes} output=${output_dir}"
cd "${repo_root}"
"${launcher[@]}" "${train_args[@]}"
