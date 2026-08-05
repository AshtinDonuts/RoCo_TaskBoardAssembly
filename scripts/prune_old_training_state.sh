#!/usr/bin/env bash
# Keep only pretrained_model for old checkpoints to save disk.
# Always preserves the newest checkpoint fully (for resume) and every
# pretrained_model directory.
set -euo pipefail
root="${1:-outputs/diffusion}"
keep_full="${2:-1}"  # number of newest checkpoints to keep with training_state

mapfile -t dirs < <(find "${root}" -type d -name 'training_state' | sort)
n=${#dirs[@]}
if (( n <= keep_full )); then
  echo "[prune] nothing to prune (${n} training_state dirs)"
  exit 0
fi
# Drop training_state from all but the newest keep_full
for ((i=0; i<n-keep_full; i++)); do
  echo "[prune] rm ${dirs[$i]}"
  rm -rf "${dirs[$i]}"
done
