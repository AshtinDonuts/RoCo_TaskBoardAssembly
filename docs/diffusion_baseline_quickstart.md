# Diffusion Baseline: Setup, Training, and Isaac Sim Evaluation

This is the condensed reproduction guide for the LeRobot Diffusion Policy
baseline prepared in this repository. Run commands from the repository root.

## 1. Machine requirements

- Linux x86-64 with an NVIDIA GPU and working Vulkan/CUDA drivers.
- CPython 3.11 and [`uv`](https://docs.astral.sh/uv/).
- Git LFS, FFmpeg, and enough local storage.
- Approximately 17 GB for the Isaac environment, 6.4 GB for the LeRobot
  environment, 1.1 GB for the dataset, and several GB per training run.
- A display/X server for rendered GUI runs. Headless evaluation can instead set
  `ISAACSIM_HEADLESS=1`.

The tested stack uses Isaac Sim 5.1.0, NumPy 1.26.4, LeRobot 0.4.4, and a
CUDA-enabled PyTorch build.

## 2. Clone and materialize large assets

Install Git LFS using the package manager for the new machine, then clone and
materialize the tracked USD, OBJ, and other large files:

```bash
git clone <repository-url> RoCo_TaskBoardAssembly
cd RoCo_TaskBoardAssembly
git lfs install
git lfs pull
```

If a USD file is only a short text file beginning with the Git LFS pointer
header, the LFS objects have not been materialized yet. Do not run Isaac until
`git lfs pull` completes.

Use the current repository revision/working tree. It includes the static-board,
policy-cadence, and video-timing fixes described below; an older checkout will
not reproduce the corrected rollout behavior.

## 3. Create the Isaac Sim environment

The checked-in lockfile installs the pip distribution of Isaac Sim, so a
separate Omniverse Launcher installation is not required:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv sync
uv run python -c "import isaacsim; print('Isaac Sim ready')"
```

`uv sync` creates `.venv/` from `uv.lock`. If the default uv cache filesystem
is too small, place the cache on a sufficiently large disk, preferably on the
same filesystem as the repository so uv can hardlink packages:

```bash
export UV_CACHE_DIR=/path/to/large-disk/.uv-cache
uv sync
```

Important pinned values are in `pyproject.toml`:

- Python `>=3.11,<3.12`
- `isaacsim[all,extscache]==5.1.0.0`
- NumPy overridden to `1.26.4`

## 4. Create the separate LeRobot environment

Training dependencies are intentionally isolated from Isaac's Python stack:

```bash
./scripts/setup_lerobot_env.sh
```

This recreates `.venv_lerobot/` and installs:

- CUDA PyTorch/torchvision (CUDA 12.4 wheels when available)
- `lerobot[diffusion]==0.4.4`
- SciPy, headless OpenCV, and pytest

Confirm both the package version and CUDA visibility:

```bash
.venv_lerobot/bin/python -c \
  "from importlib.metadata import version; import torch; print(version('lerobot'), torch.__version__, torch.cuda.is_available())"
```

## 5. Download and validate the training dataset

The baseline pins this Hugging Face dataset revision:

```text
repository: rocochallenge2025/rocochallenge2026_Industrial_Assembly
revision:   dc03b003f94d184b2b20465ed986456ee1bf2a3c
```

Authenticate with Hugging Face first if the dataset requires it. Validation
downloads/caches the dataset and checks its schema:

```bash
export HF_HOME="$PWD/.hf-cache"
export HF_LEROBOT_HOME="$PWD/.hf-cache/lerobot"
.venv_lerobot/bin/python scripts/validate_roco_dataset.py
```

Expected dataset properties:

- 200 episodes and 121,454 frames at 10 FPS
- Three 240x320 RGB streams: head, left hand, and right hand
- 44-dimensional `observation.state`
- 14-dimensional bimanual action

The normal local dataset location is:

```text
.hf-cache/lerobot/rocochallenge2025/rocochallenge2026_Industrial_Assembly
```

Pass a different local copy with `--root /path/to/dataset`, or set
`DP_DATASET_ROOT` when training.

## 6. Smoke-train before the production run

Run a 200-step, three-episode smoke job first:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/train_diffusion.sh smoke
```

Useful overrides are `DP_STEPS`, `DP_BATCH_SIZE`, `DP_NUM_WORKERS`,
`DP_SAVE_FREQ`, `DP_LOG_FREQ`, `DP_SEED`, `DP_DATASET_ROOT`, and
`DP_OUTPUT_DIR`.

Load the resulting model and verify that it emits a finite 14-D action:

```bash
.venv_lerobot/bin/python scripts/smoke_infer_dp.py \
  outputs/diffusion/<smoke-run>/checkpoints/last/pretrained_model
```

## 7. Production training used for the current checkpoint

The current production run was configured as follows:

```bash
CUDA_VISIBLE_DEVICES=0 \
DP_BATCH_SIZE=64 \
DP_SAVE_FREQ=15000 \
./scripts/train_diffusion.sh production
```

Its saved `train_config.json` records:

- Requested steps: 95,000; current usable checkpoint: step 75,000
- Batch size: 64; workers: 4; seed: 1000
- ResNet-18 visual backbone shared across three cameras
- Observation steps: 2; horizon: 16; action steps: 4
- `drop_n_last_frames=11`
- Adam at `1e-4`, cosine schedule, 500 warmup steps
- Checkpoint frequency: 15,000; W&B and Hub upload disabled
- Image transforms disabled; ImageNet statistics disabled

The checkpoint currently used for deployment is:

```text
outputs/diffusion/production_20260804_192033/checkpoints/last/pretrained_model
```

`last` is a symlink to `075000`. Training was configured for 95,000 steps but
the available run stopped at 75,000, so do not describe this checkpoint as a
completed 95,000-step model.

The repository ignores `outputs/`, so the checkpoint is **not transferred by
Git**. Copy the run directory separately to the same relative location on the
new machine, for example with `rsync`:

```bash
rsync -a --info=progress2 \
  user@training-machine:/path/to/RoCo_TaskBoardAssembly/outputs/diffusion/production_20260804_192033/ \
  outputs/diffusion/production_20260804_192033/
```

For deployment only, every file in `checkpoints/075000/pretrained_model/` is
required. Also preserve `training_state/` if training might be resumed.

Old optimizer/training-state directories consume substantial disk space. The
following keeps all exported `pretrained_model` directories while retaining
full resume state only for the newest checkpoint:

```bash
./scripts/prune_old_training_state.sh outputs/diffusion 1
```

## 8. CPU-side checkpoint and adapter checks

```bash
.venv_lerobot/bin/python scripts/smoke_infer_dp.py \
  outputs/diffusion/production_20260804_192033/checkpoints/last/pretrained_model

.venv_lerobot/bin/python -m pytest -q tests/test_diffusion_adapter.py
```

The adapter tests currently report seven passing tests.

The dataset revision encodes each 14-D action as left and right
`[x, y, z, intrinsic-XYZ Euler, gripper]` slices. Euler values are unwrapped and
may exceed `±pi`. The current Isaac runner executes the left slice and holds the
right arm at its initial pose.

## 9. Run the learned policy in rendered Isaac Sim

Use the low-quality preset for the most efficient rendered evaluation:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
ISAACSIM_HEADLESS=0 \
DP_CUDA_VISIBLE_DEVICES=0 \
DP_RENDER_QUALITY=low \
DP_EVAL_DIR="$PWD/artifacts/dp_eval_baseline" \
./scripts/eval_diffusion_roco.sh \
  "$PWD/outputs/diffusion/production_20260804_192033/checkpoints/last/pretrained_model"
```

The output directory receives:

- `head.mp4`: head-camera rollout at 10 FPS
- `results.json`: per-component grading results

The final stage snapshot is written to `task/scene_final.usd`.

For a bounded smoke rollout, add one or more environment variables:

```bash
DP_EVAL_MAX_STEPS=200       # 20 simulated seconds at the 10 Hz outer loop
DP_EVAL_MAX_PARTS=1
DP_EVAL_MAX_SIM_SECONDS=20
```

Do not append harness flags after the checkpoint argument: the evaluation
script currently builds its extra arguments from the `DP_EVAL_*` environment
variables.

## 10. Rendering and control fixes included in this revision

These fixes are required for meaningful evaluation:

1. `/World/task_board` is enforced as static collision geometry. Any
   `RigidBodyAPI` under the board is removed, its pose is checked again at
   shutdown, and all nine loose components are verified as dynamic.
2. The task no longer wraps the board in `SingleRigidPrim`, which previously
   applied `RigidBodyAPI` implicitly and allowed the arm to knock it away.
3. Diffusion policy queries are scheduled from the actual physics step index.
   One rendered outer-loop call advances 20 physics ticks; counting 20 calls
   had accidentally reduced the policy from 10 Hz to approximately 0.5 Hz.
4. Diffusion videos default to 10 FPS, matching the renderer and dataset
   cadence. The earlier 15 FPS encoding distorted playback timing.
5. `DP_RENDER_QUALITY=low` uses 320x240 camera sensors, disables three extra UI
   camera tiles, reduces the main viewport, and disables antialiasing. It does
   not change physics or policy timing.

The board fixation defaults to enabled. It can be controlled directly with
`--fixate-assembly-board` or `--no-fixate-assembly-board` when invoking
`task/run_pick_place.py` without the evaluation wrapper.

## 11. Runtime troubleshooting

### No GUI or Vulkan device

Verify `DISPLAY`, the NVIDIA driver, Vulkan ICD configuration, and GPU access.
On machines where auto-detection fails, this may help:

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

### `GLIBCXX_* not found` or C++ runtime mismatch

We encountered a host C++ runtime mismatch and temporarily launched Isaac with
a compatible `libgcc_s.so.1` and `libstdc++.so.6` through `LD_PRELOAD`. Those
paths were specific to that machine and must not be copied blindly. Prefer
installing a compatible system C++ runtime. If a preload is unavoidable,
resolve and validate the exact libraries on the new host before setting it.

### Simulator runs slower than real time

That is acceptable for synchronous evaluation: wall-clock stalls do not alter
simulation-time policy cadence. Confirm the output MP4 is 10 FPS and its frame
count corresponds to the simulated duration. Use `DP_RENDER_QUALITY=low` to
reduce rendering cost.

### Checkpoint loads but rollout fails

Check `task/dp_server.log`, confirm `.venv_lerobot` can see CUDA, and rerun
`scripts/smoke_infer_dp.py`. Also confirm the three observation images are
240x320, state is 44-D, and action is 14-D.

## 12. Current baseline status

- Checkpoint: production step 75,000
- Deployment path: LeRobot sidecar process plus Isaac-side adapter
- Static-board invariant: verified at startup and finalization
- Most recent full pre-cadence-fix rollout: 0/9 task components passed
- Corrected bounded rollout artifact:
  `artifacts/dp_eval_last_low_quality_cadence_fixed_20260805/head.mp4`

Because the full 0/9 run used the erroneous 0.5 Hz adapter cadence, it is not
a valid measure of the corrected policy. Run a new full evaluation before
using a success score for comparison.
