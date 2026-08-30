#!/usr/bin/env bash
# Bootstrap RoCo Task Board Assembly on Ubuntu 22.04 + NVIDIA RTX GPU.
# Idempotent where possible. Does not install the NVIDIA driver (manual step).
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

log() { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARNING: %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  die "This project only supports Linux x86_64 (see pyproject.toml)."
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" == "ubuntu" ]]; then
    case "${VERSION_ID:-}" in
      22.04|24.04) ;;
      *)
        warn "Detected Ubuntu ${VERSION_ID:-unknown}; Isaac Sim 5.1 targets 22.04/24.04."
        ;;
    esac
  else
    warn "Non-Ubuntu Linux (${ID:-unknown}); continuing anyway."
  fi
fi

need_sudo=0
if [[ "${EUID}" -ne 0 ]]; then
  need_sudo=1
  command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages."
fi

run_root() {
  if [[ "${need_sudo}" -eq 1 ]]; then
    sudo "$@"
  else
    "$@"
  fi
}

log "Installing system packages (apt)…"
run_root apt-get update -y
run_root DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  curl \
  git \
  git-lfs \
  ffmpeg \
  libvulkan1 \
  vulkan-tools \
  mesa-vulkan-drivers \
  libgl1 \
  libglib2.0-0 \
  libxkbcommon0 \
  libx11-6 \
  libxi6 \
  libxrandr2 \
  libxcursor1 \
  libxinerama1 \
  libxss1 \
  ca-certificates

if [[ ! -f /etc/sysctl.d/99-isaac-inotify.conf ]]; then
  log "Raising fs.inotify.max_user_watches…"
  echo 'fs.inotify.max_user_watches=524288' | run_root tee /etc/sysctl.d/99-isaac-inotify.conf >/dev/null
  run_root sysctl --system >/dev/null || warn "sysctl --system failed; reboot may be needed for inotify limit."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found. Install NVIDIA driver R580 (recommended for Isaac Sim 5.1), then reboot."
  warn "See docs/SETUP_UBUNTU22_RTX4090.md §2."
else
  log "GPU:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1 || true)"
  if [[ -n "${driver_ver}" && "${driver_ver}" =~ ^[0-9]+$ ]]; then
    if (( driver_ver < 525 )); then
      warn "Driver ${driver_ver}.x is older than recommended; Isaac Sim 5.1 documents 580.65.06."
    elif (( driver_ver >= 590 )); then
      warn "Driver ${driver_ver}.x is newer than Isaac Sim 5.1's validated R580 branch; prefer 580.x if you hit CUDA/render crashes."
    fi
  fi
fi

if [[ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  warn "Missing /etc/vulkan/icd.d/nvidia_icd.json — Vulkan NVIDIA ICD may be incomplete until the driver is installed."
fi

log "Configuring Git LFS…"
git lfs install --skip-repo >/dev/null 2>&1 || git lfs install
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "Fetching Git LFS objects…"
  git lfs pull || warn "git lfs pull failed; large USD/mesh assets may still be pointer stubs."
fi

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv not on PATH; add \$HOME/.local/bin to PATH and re-run."
log "uv $(uv --version)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$(cd -- "${repo_root}/.." && pwd)/.uv-cache}"
mkdir -p "${UV_CACHE_DIR}"
log "UV_CACHE_DIR=${UV_CACHE_DIR}"

log "Syncing Isaac Sim 5.1.0 env from uv.lock (large download; ~18 GB)…"
sync_args=(sync)
if [[ "${ROCO_SETUP_WITH_COLLECTION:-1}" == "1" ]]; then
  sync_args+=(--group collection)
fi
if [[ "${ROCO_SETUP_WITH_DEV:-0}" == "1" ]]; then
  sync_args+=(--group dev)
fi
uv "${sync_args[@]}"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
log "Smoke-importing isaacsim (accepts EULA via OMNI_KIT_ACCEPT_EULA)…"
uv run python -c "import isaacsim; print('isaacsim import ok')"

cat <<EOF

[setup] Done.

Next (single RTX 4090):
  export OMNI_KIT_ACCEPT_EULA=YES
  export ISAACSIM_HEADLESS=1
  export ISAACSIM_ACTIVE_GPU=0
  export ISAACSIM_PHYSICS_GPU=0
  ./scripts/run_roco.sh --max-sim-seconds 15 \\
      --record-video artifacts/smoke_head.mp4 \\
      --results-json artifacts/smoke_results.json

Full guide: docs/SETUP_UBUNTU22_RTX4090.md
EOF
