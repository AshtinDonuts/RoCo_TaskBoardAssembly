#!/usr/bin/env bash
# Launch the L-arm init joint calibration tool (robot + floor, omni.ui sliders).
# Requires a GUI display — do not run headless.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cache_root="$(cd -- "${repo_root}/.." && pwd)"

cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/.uv-cache}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[calibrate_l_arm] ERROR: DISPLAY is not set; this tool needs a GUI." >&2
  exit 1
fi

# Never force headless for calibration UI.
unset ISAACSIM_HEADLESS || true

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

exec uv run python task/calibrate_l_arm_joints.py "$@"
