#!/usr/bin/env bash
#SBATCH --job-name=roco-pi05-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%j.out
# Add site-specific --account/--partition directives with sbatch CLI options.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PI05_CLUSTER_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "${PI05_CLUSTER_SETUP}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PI05_NUM_PROCESSES=1
export PI05_OUTPUT_DIR="${PI05_OUTPUT_DIR:-${repo_root}/outputs/pi05}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[slurm] job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown}"
echo "[slurm] restart=${SLURM_RESTART_COUNT:-0} gpu=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

cd "${repo_root}"
exec ./scripts/train_pi05.sh "${1:-production}" lora
