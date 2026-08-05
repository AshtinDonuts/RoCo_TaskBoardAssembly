#!/usr/bin/env bash
# Bootstrap an isolated LeRobot 0.4.4 env for Diffusion Policy training/inference.
# Keeps Isaac Sim's pinned Python stack untouched.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${LEROBOT_VENV:-${repo_root}/.venv_lerobot}"
python_bin="${LEROBOT_PYTHON:-python3.11}"
export PATH="${HOME}/.local/bin:${PATH}"

echo "[setup] creating ${venv_dir} with ${python_bin}"
if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required to create the LeRobot venv (ensurepip is unavailable)" >&2
  exit 1
fi
uv venv "${venv_dir}" --python "${python_bin}" --clear
uv pip install --python "${venv_dir}/bin/python" pip setuptools wheel

"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
# Pin the training/inference stack used by dp_server.py.
# Prefer CUDA wheels when available.
"${venv_dir}/bin/python" -m pip install \
  "torch" "torchvision" \
  --index-url https://download.pytorch.org/whl/cu124 \
  || "${venv_dir}/bin/python" -m pip install "torch" "torchvision"

"${venv_dir}/bin/python" -m pip install \
  "lerobot[diffusion]==0.4.4" \
  "scipy" \
  "opencv-python-headless" \
  "pytest"

"${venv_dir}/bin/python" - <<'PY'
from importlib.metadata import version
import torch
print(f"[setup] lerobot={version('lerobot')} torch={torch.__version__} cuda={torch.cuda.is_available()}")
assert version("lerobot") == "0.4.4", version("lerobot")
PY

echo "[setup] done: ${venv_dir}/bin/python"
echo "export DP_SERVER_PY=${venv_dir}/bin/python"
