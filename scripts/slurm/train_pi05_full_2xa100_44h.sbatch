#!/usr/bin/env bash
#SBATCH --job-name=roco-pi05-full-44h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --gres=gpu:ampere:2
#SBATCH --mem=180G
#SBATCH --time=44:00:00
# Give 10 minutes after SIGTERM for a final ~23GB checkpoint write.
#SBATCH --signal=B:TERM@600
#SBATCH --output=slurm-%x-%j.out
# Add site-specific --account/--partition with sbatch CLI options.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PI05_CLUSTER_SETUP:-}" ]]; then
  # shellcheck source=/dev/null
  source "${PI05_CLUSTER_SETUP}"
fi

umask 002
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PI05_NUM_PROCESSES=2
# VRAM-safe full FT on 40GB A100s (smoke peak ~37GB at batch 1).
export PI05_BATCH_SIZE="${PI05_BATCH_SIZE:-1}"
# Conservative ~6.5s/step from smoke timings with headroom for DDP/checkpoint I/O:
#   ~10h ≈ 5500 steps; ~42h train budget ≈ 23000 steps inside 44h walltime.
export PI05_STEPS="${PI05_STEPS:-23000}"
export PI05_SAVE_FREQ="${PI05_SAVE_FREQ:-5500}"
export PI05_LOG_FREQ="${PI05_LOG_FREQ:-50}"
export PI05_NUM_WORKERS="${PI05_NUM_WORKERS:-4}"
export PI05_OUTPUT_DIR="${PI05_OUTPUT_DIR:-${repo_root}/outputs/pi05/full_2xa100_44h}"
export HF_HOME="${HF_HOME:-${repo_root}/.hf-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PI05_SKIP_ACCESS_CHECK="${PI05_SKIP_ACCESS_CHECK:-0}"

mkdir -p "${PI05_OUTPUT_DIR}"
chmod a+rx "${PI05_OUTPUT_DIR}" 2>/dev/null || true

echo "[slurm] job=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-unknown}"
echo "[slurm] restart=${SLURM_RESTART_COUNT:-0} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[slurm] steps=${PI05_STEPS} save_freq=${PI05_SAVE_FREQ} batch=${PI05_BATCH_SIZE} out=${PI05_OUTPUT_DIR}"
nvidia-smi || true

cd "${repo_root}"

publish_bin="${repo_root}/scripts/slurm/publish_pi05_checkpoints.sh"
publish_loop() {
  while true; do
    "${publish_bin}" "${PI05_OUTPUT_DIR}" || true
    sleep 300
  done
}
publish_loop &
publish_pid=$!

train_pid=""
forward_term() {
  echo "[slurm] TERM received; forwarding to training process group for graceful checkpoint"
  if [[ -n "${train_pid}" ]]; then
    kill -TERM -- "-${train_pid}" 2>/dev/null || kill -TERM "${train_pid}" 2>/dev/null || true
  fi
}
trap forward_term TERM

set -m
./scripts/train_pi05.sh "${1:-production}" full &
train_pid=$!
set +e
wait "${train_pid}"
rc=$?
set -e

kill "${publish_pid}" 2>/dev/null || true
wait "${publish_pid}" 2>/dev/null || true
"${publish_bin}" "${PI05_OUTPUT_DIR}" || true

echo "[slurm] train exit=${rc}"
exit "${rc}"
