#!/usr/bin/env bash
# Make completed deployable pi0.5 checkpoints readable for other users.
# Safe to run repeatedly; only touches finished pretrained_model dirs.
set -euo pipefail

root="${1:?usage: $0 /path/to/run_or_output_root}"
if [[ ! -d "${root}" ]]; then
  exit 0
fi

find "${root}" -type d -name pretrained_model -print0 2>/dev/null \
  | while IFS= read -r -d '' ckpt; do
      if [[ -f "${ckpt}/train_config.json" ]] && {
           [[ -f "${ckpt}/model.safetensors" ]] || [[ -f "${ckpt}/adapter_config.json" ]]
         }; then
        chmod -R a+rX "${ckpt}" 2>/dev/null || true
        # Keep optimizer/resume blobs owner-private when present beside pretrained_model.
        train_state="$(dirname "${ckpt}")/training_state"
        if [[ -d "${train_state}" ]]; then
          chmod -R go-rwx "${train_state}" 2>/dev/null || true
        fi
      fi
    done
