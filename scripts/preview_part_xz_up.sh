#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ -z "${DISPLAY:-}" && "${ISAACSIM_HEADLESS:-}" != "1" ]]; then
  echo "[part_preview] ERROR: DISPLAY is not set; this tool needs a GUI." >&2
  exit 1
fi

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

exec uv run python task/preview_part_from_config.py "$@"
