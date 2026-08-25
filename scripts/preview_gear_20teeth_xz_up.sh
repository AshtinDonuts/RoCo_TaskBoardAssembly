#!/usr/bin/env bash
# Preview gear_20teeth in an empty Isaac scene (ground + XZ face up / +Z).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cache_root="$(cd -- "${repo_root}/.." && pwd)"

cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/.uv-cache}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[preview_gear] ERROR: DISPLAY is not set; this tool needs a GUI." >&2
  exit 1
fi

unset ISAACSIM_HEADLESS || true

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

exec uv run python task/preview_gear_20teeth_xz_up.py "$@"
