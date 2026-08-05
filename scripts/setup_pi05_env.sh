#!/usr/bin/env bash
# Create the isolated, pinned LeRobot checkout used by pi0.5 training/inference.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_root="$(cd "${repo_root}/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"
lerobot_repo="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"
lerobot_revision="${LEROBOT_REVISION:-2f2b567}"
python_bin="${PI05_PYTHON:-python3.12}"
torch_index="${PI05_TORCH_INDEX:-https://download.pytorch.org/whl/cu118}"
torch_build_suffix="${PI05_TORCH_BUILD_SUFFIX:-+cu118}"
torch_version="${PI05_TORCH_VERSION:-2.7.1}"
torchvision_version="${PI05_TORCHVISION_VERSION:-0.22.1}"
torchcodec_version="${PI05_TORCHCODEC_VERSION:-0.3.0}"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required" >&2
  exit 1
fi

if [[ ! -d "${lerobot_root}/.git" ]]; then
  if [[ -e "${lerobot_root}" ]]; then
    echo "error: ${lerobot_root} exists but is not a git checkout" >&2
    exit 1
  fi
  git clone "${lerobot_repo}" "${lerobot_root}"
fi

current_revision="$(git -C "${lerobot_root}" rev-parse HEAD)"
if [[ "${current_revision}" != "${lerobot_revision}" && "${current_revision:0:7}" != "${lerobot_revision}" ]]; then
  if [[ -n "$(git -C "${lerobot_root}" status --porcelain)" ]]; then
    echo "error: ${lerobot_root} has local changes; refusing to switch revisions" >&2
    exit 1
  fi
  git -C "${lerobot_root}" fetch origin "${lerobot_revision}"
  git -C "${lerobot_root}" checkout --detach "${lerobot_revision}"
fi

echo "[setup] lerobot_root=${lerobot_root}"
echo "[setup] revision=$(git -C "${lerobot_root}" rev-parse HEAD)"
(
  cd "${lerobot_root}"
  uv sync \
    --python "${python_bin}" \
    --extra pi \
    --extra training \
    --extra peft \
    --extra test
  # LeRobot's lock defaults to CUDA 12.8, which requires a newer driver than
  # many A100 clusters. The cu118 build remains compatible with driver 510.
  uv pip install \
    --python "${lerobot_root}/.venv/bin/python" \
    --reinstall \
    "torch==${torch_version}${torch_build_suffix}" \
    "torchvision==${torchvision_version}${torch_build_suffix}" \
    --index-url "${torch_index}"
  uv pip install \
    --python "${lerobot_root}/.venv/bin/python" \
    --reinstall \
    --no-deps \
    "torchcodec==${torchcodec_version}"
)

"${lerobot_root}/.venv/bin/python" - <<'PY'
from importlib.metadata import version

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

print(
    f"[setup] lerobot={version('lerobot')} torch={torch.__version__} "
    f"cuda={torch.cuda.is_available()} policy={PI05Policy.name}"
)
PY

echo "[setup] done"
echo "export LEROBOT_ROOT=${lerobot_root}"
echo "export PI05_SERVER_PY=${lerobot_root}/.venv/bin/python"
