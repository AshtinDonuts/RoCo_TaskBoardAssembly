# Environment Setup — Ubuntu 22.04 + RTX 4090

Reproduce the RoCo Task Board Assembly Isaac Sim harness on a clean
**Ubuntu 22.04 (Jammy)** machine with a single **GeForce RTX 4090** (24 GB).

Pinned stack (from repo root):

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS, x86_64 |
| Python | CPython **3.11** (via `uv` / `.python-version`) |
| Isaac Sim | **5.1.0** (`isaacsim[all,extscache]==5.1.0.0`) |
| Env manager | [`uv`](https://docs.astral.sh/uv/) + `uv.lock` |
| NVIDIA driver (validated for Isaac 5.1) | **580.65.06** production branch |

Official Isaac Sim 5.1 requirements:
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html

---

## 0. Hardware / disk checklist

- RTX 4090 with RT cores (required; Ampere datacenter GPUs without RT cores are unsupported).
- **≥ 32 GB RAM** (64 GB preferred).
- **≥ 80 GB free SSD** near the clone (Isaac wheels ≈ 18 GB in `.venv/`; `uv` cache can hardlink another ~18 GB if cache and repo share a filesystem).
- Headless SSH is fine; GUI optional.

Quick bootstrap (installs apt packages + `uv`, then syncs the venv):

```bash
# after cloning this repo
bash scripts/setup_ubuntu22_rtx4090.sh
```

Or follow the manual steps below.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y \
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
```

Raise inotify watches (Isaac may spam `errno=28` otherwise; usually non-fatal in headless):

```bash
echo 'fs.inotify.max_user_watches=524288' | sudo tee /etc/sysctl.d/99-isaac-inotify.conf
sudo sysctl --system
```

---

## 2. NVIDIA driver (Isaac Sim 5.1)

Isaac Sim **5.1** documents Linux driver **580.65.06**. Prefer the **R580
production** branch. Newer feature branches (e.g. 595.x / 610.x) have caused
CUDA detection failures and `rtx.scenedb` crashes with Isaac 5.x.

### Option A — Ubuntu packages (simplest)

```bash
sudo apt update
# Prefer 580 if available in your apt sources:
sudo apt install -y nvidia-driver-580
# Fallback often available on 22.04: nvidia-driver-535
sudo reboot
```

### Option B — NVIDIA `.run` installer (closest to validated)

1. Download **Linux 64-bit Production Branch** driver **580.65.06** (or latest R580) from
   https://www.nvidia.com/Download/Find.aspx or the Unix Driver Archive.
2. Drop to text mode / stop display manager, then:

```bash
chmod +x NVIDIA-Linux-x86_64-580.65.06.run
sudo ./NVIDIA-Linux-x86_64-580.65.06.run
sudo reboot
```

### Verify

```bash
nvidia-smi
# Expect: RTX 4090, Driver Version ~580.x, CUDA Version reported by driver ≥ 12.x
vulkaninfo --summary | head -40   # should list NVIDIA ICD
```

Confirm Vulkan ICD path (used by `scripts/roco_isaac_env.sh`):

```bash
ls /etc/vulkan/icd.d/nvidia_icd.json
```

You do **not** need a full CUDA toolkit install for the Isaac Sim pip stack;
the driver + Isaac wheels are enough for simulation. Install a CUDA toolkit
only if you separately train/infer models in another env (e.g. LeRobot).

---

## 3. Clone the repo (Git LFS)

```bash
git lfs install
git clone <YOUR_REMOTE_URL> RoCo_TaskBoardAssembly
cd RoCo_TaskBoardAssembly
git lfs pull   # scene_init.usd / scene_base.usd / meshes / videos
```

Tracked LFS patterns: `*.usd`, `*.usdc`, `*.obj`, `*.mp4` (see `.gitattributes`).

Sanity-check large assets are real files, not pointer stubs:

```bash
# pointer stubs start with "version https://git-lfs.github.com/spec/v1"
head -c 80 scene_init.usd
ls -lh scene_init.usd scene_base.usd
```

---

## 4. Install `uv` and sync Isaac Sim

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure ~/.local/bin is on PATH (installer prints the line)
export PATH="$HOME/.local/bin:$PATH"

cd /path/to/RoCo_TaskBoardAssembly

# Optional: put the uv cache on the same filesystem as the repo so hardlinks work
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/../.uv-cache}"

uv sync                       # creates .venv/ from uv.lock (~18 GB)
# Optional extras used by dataset collection / splitting:
uv sync --group collection
# Optional: pytest / ruff / ty
uv sync --group collection --group dev
```

Accept the Omniverse EULA (or set before every run):

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv run python -c "import isaacsim; print('isaacsim import ok')"
```

First Kit import can take several minutes while extension caches warm up.

---

## 5. Single-GPU (RTX 4090) runtime env

`scripts/roco_isaac_env.sh` already sets EULA, Vulkan ICD, headless `DISPLAY`
cleanup, and UV cache defaults. For one GPU, pin Isaac to device **0**:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_HEADLESS=1          # omit or set 0 for GUI
export ISAACSIM_ACTIVE_GPU=0
export ISAACSIM_PHYSICS_GPU=0
# Prefer the launcher (sources roco_isaac_env.sh):
./scripts/run_roco.sh --help
```

Ubuntu 22.04 ships `libstdc++` with `GLIBCXX_3.4.29+`, so the Ubuntu 20.04
`LD_PRELOAD` workaround in `roco_isaac_env.sh` should not trigger.

---

## 6. Smoke tests

Baseline scripted policy (short cap):

```bash
cd /path/to/RoCo_TaskBoardAssembly
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_HEADLESS=1
export ISAACSIM_ACTIVE_GPU=0
export ISAACSIM_PHYSICS_GPU=0

./scripts/run_roco.sh \
  --max-sim-seconds 15 \
  --record-video artifacts/smoke_head.mp4 \
  --results-json artifacts/smoke_results.json
```

Or equivalently:

```bash
uv run python task/run_pick_place.py \
  --max-sim-seconds 15 \
  --record-video artifacts/smoke_head.mp4 \
  --results-json artifacts/smoke_results.json
```

Full baseline (no time cap):

```bash
./scripts/run_roco.sh \
  --record-video artifacts/baseline_head.mp4 \
  --results-json artifacts/baseline_results.json
```

Fairness XY randomization trial:

```bash
./scripts/run_roco.sh --random-seed 0 --max-parts 1
```

---

## 7. Optional: LeRobot collection group

In-process LeRobot v3 collection (same Isaac venv + `collection` group):

```bash
uv sync --group collection
OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_HEADLESS=1 \
  uv run --group collection python task/collect_lerobot_v3.py --help
```

---

## 8. Optional: pi0.5 / Diffusion sidecar (separate Python env)

Learned policies run in a **sidecar** process because Torch/LeRobot cannot
share Isaac’s pinned env. On a **single 4090**, Isaac and the model share GPU
0 — watch VRAM (pi0.5 + Isaac may need lower batch / headless / fewer cameras).

Default multi-GPU docs use separate IDs; remap for one GPU:

```bash
# Isaac
export ISAACSIM_ACTIVE_GPU=0
export ISAACSIM_PHYSICS_GPU=0
# Sidecar sees only GPU 0
export PI05_CUDA_VISIBLE_DEVICES=0
export PI05_DEVICE=cuda
```

See [`pi05_eval.md`](pi05_eval.md) and [`pi05_baseline_training.md`](pi05_baseline_training.md).
Expected layout:

```text
../lerobot_roco_pi05/          # LeRobot checkout + its own .venv
../.hf-cache/                  # Hugging Face cache (HF_HOME)
```

Eval launcher:

```bash
LEROBOT_ROOT=/path/to/lerobot_roco_pi05 \
HF_HOME=/path/to/.hf-cache \
ISAACSIM_HEADLESS=1 \
ISAACSIM_ACTIVE_GPU=0 \
ISAACSIM_PHYSICS_GPU=0 \
PI05_CUDA_VISIBLE_DEVICES=0 \
./scripts/eval_pi05_roco.sh /path/to/checkpoint/pretrained_model
```

---

## 9. Common failures

| Symptom | Fix |
|---------|-----|
| `nvidia-smi` missing / wrong GPU | Install R580 driver; reboot; confirm 4090 in `nvidia-smi` |
| Isaac segfault / no CUDA device on very new drivers | Stay on **R580**; avoid 595/610 feature branches for Isaac 5.1 |
| EULA prompt hangs in scripts | `export OMNI_KIT_ACCEPT_EULA=YES` |
| `scene_*.usd` tiny (~130 B) | `git lfs pull` |
| Disk doubled during `uv sync` | Put `UV_CACHE_DIR` on the **same filesystem** as the repo |
| `Failed to create change watch … errno=28` | Raise `fs.inotify.max_user_watches` (step 1) |
| Headless crash with SSH `-X` | `ISAACSIM_HEADLESS=1` and unset `DISPLAY` (done by `roco_isaac_env.sh`) |
| ffmpeg / video encode failures | `sudo apt install ffmpeg`; avoid putting conda `lib` on `LD_LIBRARY_PATH` |
| OOM on 24 GB with Isaac + large VLM | Run headless; close other GPU users; for sidecars prefer eval-only or sequential use |

---

## 10. What this machine does **not** need

- Omniverse Launcher / separate Isaac Sim `.zip` install (pip/`uv` is enough).
- Conda for the Isaac harness (optional only if you prefer it for unrelated tools).
- Multi-GPU FSDP configs (`scripts/accelerate_pi05_fsdp_*.yaml`) unless you add more GPUs for training.

Repo sources of truth: `pyproject.toml`, `uv.lock`, `.python-version`, `scripts/roco_isaac_env.sh`.
