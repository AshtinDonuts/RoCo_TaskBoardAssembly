#!/usr/bin/env bash
# Keep only the newest N complete diffusion checkpoints across all runs.
# A complete checkpoint contains both pretrained_model weights and, when
# present, training_state needed to resume training.
set -euo pipefail

root="${1:-outputs/diffusion}"
keep="${2:-2}"

if [[ ! "${keep}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: keep count must be a positive integer, got '${keep}'" >&2
  exit 2
fi
if [[ ! -d "${root}" ]]; then
  echo "[prune] checkpoint root does not exist: ${root}"
  exit 0
fi

# Sort by modification time of pretrained_model, oldest first. Each selected
# path is the numbered checkpoint directory containing pretrained_model.
mapfile -d '' -t entries < <(
  find "${root}" -type d -name pretrained_model \
    -printf '%T@ %h\0' | sort -z -n
)
n=${#entries[@]}

if (( n <= keep )); then
  echo "[prune] retaining all ${n} diffusion checkpoints (limit=${keep})"
  exit 0
fi

for ((i=0; i<n-keep; i++)); do
  checkpoint="${entries[$i]#* }"
  echo "[prune] removing old checkpoint ${checkpoint}"
  rm -rf -- "${checkpoint}"
done

# Remove per-run `last` links whose checkpoint was just pruned.
while IFS= read -r -d '' link; do
  if [[ ! -e "${link}" ]]; then
    echo "[prune] removing dangling link ${link}"
    rm -- "${link}"
  fi
done < <(find "${root}" -type l -name last -print0)

echo "[prune] retained the newest ${keep} diffusion checkpoints"
