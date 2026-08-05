# pi0.5 Baseline: Setup, Fine-tuning, and Isaac Sim Evaluation

This guide reproduces the RoCo pi0.5 baseline with three supported training
profiles: full fine-tuning on one 2×A100 node, LoRA on one A100, and LoRA on a
local A6000. Run commands from the repository root.

## 1. Requirements

- Linux x86-64, NVIDIA CUDA/Vulkan drivers, Git LFS, FFmpeg, Git, and `uv`.
  The Isaac Sim 5.1 pip wheels require glibc 2.35 or newer; on older HPC login
  nodes, run evaluation in a supported container/host or an existing compatible
  Isaac environment.
- Python 3.11 for the pinned Isaac Sim environment and Python 3.12 for LeRobot.
- Isaac Sim 5.1.0 dependencies installed with `uv sync`.
- About 1.1 GB for the dataset, a separate LeRobot checkout/environment, the
  pi0.5 base weights, and checkpoint storage.
- Hugging Face access to
  [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) and
  `google/paligemma-3b-pt-224`. Accept any model terms before submitting a
  batch job, then provide `HF_TOKEN` or `$HF_HOME/token`.

Clone the repository and materialize its LFS assets:

```bash
git clone <repository-url> RoCo_TaskBoardAssembly
cd RoCo_TaskBoardAssembly
git lfs install
git lfs pull
```

## 2. Create the two isolated environments

Create the Isaac environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv sync
```

Create the pinned pi0.5 checkout and `.venv`:

```bash
./scripts/setup_pi05_env.sh
export LEROBOT_ROOT="$(cd .. && pwd)/lerobot_roco_pi05"
export PI05_SERVER_PY="$LEROBOT_ROOT/.venv/bin/python"
```

The setup script defaults to LeRobot commit `2f2b567` and the PyTorch 2.7.1
CUDA 11.8 wheels, which support the A100 cluster's NVIDIA 510 driver. Override
`LEROBOT_ROOT`, `LEROBOT_REPO`, `LEROBOT_REVISION`, `PI05_PYTHON`,
`PI05_TORCH_INDEX`, `PI05_TORCH_BUILD_SUFFIX`, or the
`PI05_TORCH*_VERSION` variables when needed. It will not overwrite a non-Git
directory or switch a dirty checkout.

Verify CUDA and model imports:

```bash
"$PI05_SERVER_PY" -c \
  "import torch; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print(torch.cuda.is_available(), PI05Policy.name)"
```

## 3. Authenticate and validate the dataset

```bash
export HF_HOME="$PWD/.hf-cache"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
# export HF_TOKEN=hf_...

"$PI05_SERVER_PY" scripts/validate_roco_dataset.py --require-pi05
```

The baseline pins:

```text
dataset:  rocochallenge2025/rocochallenge2026_Industrial_Assembly
revision: dc03b003f94d184b2b20465ed986456ee1bf2a3c
```

Expected data are 200 episodes / 121,454 frames at 10 Hz, three 240×320 RGB
cameras, a 44-D state, a 14-D action, and the task text `assemble parts onto
the task board`. Actions contain left and right
`[xyz, intrinsic-XYZ Euler, gripper]` slices. Euler values are unwrapped and
can exceed ±π.

The validator reports whether quantile statistics are present. The pinned
revision includes all pi0.5 quantiles, so the supplied recipe uses quantile
normalization by default. For another compatible dataset without quantiles,
set `PI05_NORMALIZATION=mean_std`.

## 4. Smoke training

Run a short LoRA smoke test first:

```bash
CUDA_VISIBLE_DEVICES=0 \
PI05_NUM_PROCESSES=1 \
PI05_BATCH_SIZE=1 \
./scripts/train_pi05.sh smoke lora
```

For the full two-GPU code path:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PI05_BATCH_SIZE=1 \
./scripts/train_pi05.sh smoke full
```

Smoke mode uses three episodes and 20 steps by default. Outputs are placed
under `outputs/pi05/<strategy>_smoke_<timestamp>/`. Each run records the exact
command, LeRobot revision, GPU inventory, and effective batch size in
`launch.txt`.

Verify the resulting checkpoint:

```bash
"$PI05_SERVER_PY" scripts/smoke_infer_pi05.py \
  outputs/pi05/<run>/checkpoints/last/pretrained_model
```

This supports both full checkpoints (`model.safetensors`) and LoRA adapters
(`adapter_config.json` plus adapter weights).

## 5. Production profile A: full fine-tuning on 2×A100 40 GB

The full model is replicated on both GPUs with Accelerate DDP. Start with
per-device batch size 2 and reduce it to 1 if the smoke job runs out of memory:

```bash
sbatch \
  --account=<account> \
  --partition=<a100-partition> \
  --export=ALL,HF_HOME="$PWD/.hf-cache",LEROBOT_ROOT="$(cd .. && pwd)/lerobot_roco_pi05",PI05_BATCH_SIZE=2 \
  scripts/slurm/train_pi05_full_2xa100.sbatch
```

`PI05_BATCH_SIZE` is per process, so batch 2 on two GPUs has effective batch
size 4. DDP does not make a model fit that cannot fit on one GPU; gradient
checkpointing and bfloat16 are enabled. Default production length is 10,000
steps, with a 10,000-step cosine schedule.

## 6. Production profile B: LoRA on one A100 40 GB

```bash
sbatch \
  --account=<account> \
  --partition=<a100-partition> \
  --export=ALL,HF_HOME="$PWD/.hf-cache",LEROBOT_ROOT="$(cd .. && pwd)/lerobot_roco_pi05",PI05_BATCH_SIZE=4 \
  scripts/slurm/train_pi05_lora_1xa100.sbatch
```

The default LoRA rank/alpha are 16/16. Override `PI05_LORA_R`,
`PI05_LORA_ALPHA`, `PI05_LR`, and `PI05_BATCH_SIZE` after smoke calibration.
LeRobot's pi0.5 PEFT target set adapts the action expert attention projections
and pi0.5 projection layers.

Both SLURM launchers enable Hugging Face/Transformers offline mode because
compute nodes may not have outbound network access. Run dataset validation and
one local smoke setup first so the dataset, base model, and tokenizer are fully
materialized in the shared `HF_HOME`.

## 7. Production profile C: LoRA on a local A6000

```bash
CUDA_VISIBLE_DEVICES=0 \
PI05_NUM_PROCESSES=1 \
PI05_BATCH_SIZE=4 \
PI05_NUM_WORKERS=4 \
./scripts/train_pi05.sh production lora
```

Increase batch size only after monitoring peak VRAM during a smoke run.

## 8. Common overrides and resume

Useful variables:

- `PI05_STEPS`, `PI05_BATCH_SIZE`, `PI05_NUM_PROCESSES`, `PI05_NUM_WORKERS`
- `PI05_LR`, `PI05_DECAY_LR`, `PI05_WARMUP_STEPS`, `PI05_DECAY_STEPS`
- `PI05_NORMALIZATION` (`quantiles` by default, or `mean_std`)
- `PI05_SAVE_FREQ`, `PI05_LOG_FREQ`, `PI05_SEED`
- `PI05_DATASET_ROOT`, `PI05_OUTPUT_DIR`, `PI05_BASE_MODEL`
- `PI05_LORA_R`, `PI05_LORA_ALPHA`, `PI05_N_ACTION_STEPS`

Resume from a saved training configuration into a new timestamped run:

```bash
PI05_RESUME_CONFIG=/path/to/checkpoints/last/pretrained_model/train_config.json \
./scripts/train_pi05.sh production lora
```

Keep `training_state/` for resume. Deployment needs the whole
`pretrained_model/` directory. Full checkpoints are standalone; LoRA
checkpoints remain small but require access to the base model recorded in
`adapter_config.json`. The bundled server loads either form directly. If an
external deployment requires one weight file, load the adapter with PEFT,
call `merge_and_unload()`, save the merged policy with `save_pretrained()`,
and copy the saved pre/postprocessor JSON files into that export.

Old optimizer state can be pruned after a run is no longer resumable:

```bash
./scripts/prune_old_training_state.sh outputs/pi05 1
```

## 9. CPU/checkpoint checks

```bash
"$PI05_SERVER_PY" -m pytest -q tests/test_pi05_adapter.py
"$PI05_SERVER_PY" scripts/smoke_infer_pi05.py \
  outputs/pi05/<run>/checkpoints/last/pretrained_model
```

The smoke inference uses the real 44-D state, all three camera keys, and task
text, then verifies a finite 14-D action.

## 10. Bounded Isaac Sim evaluation

Use separate GPUs for Isaac and model inference when available:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
ISAACSIM_HEADLESS=1 \
ISAACSIM_ACTIVE_GPU=0 \
ISAACSIM_PHYSICS_GPU=0 \
PI05_CUDA_VISIBLE_DEVICES=1 \
PI05_RENDER_QUALITY=low \
PI05_EVAL_MAX_SIM_SECONDS=20 \
PI05_EVAL_DIR="$PWD/artifacts/pi05_smoke" \
./scripts/eval_pi05_roco.sh \
  "$PWD/outputs/pi05/<run>/checkpoints/last/pretrained_model"
```

The directory receives `head.mp4` at 10 FPS and `results.json`. Other bounds
are `PI05_EVAL_MAX_STEPS` and `PI05_EVAL_MAX_PARTS`.

The adapter queries at 10 Hz from the physics step index and holds the latest
absolute IK target between queries. It executes only the left seven action
dimensions because the current runner holds the right arm at home.

For a legacy checkpoint trained with rotation-vector actions only:

```bash
PI05_ACTION_ROTATION=rotvec ./scripts/eval_pi05_roco.sh /path/to/checkpoint
```

Do not use that override for the pinned dataset baseline.

## 11. Full evaluation

Remove all `PI05_EVAL_MAX_*` limits:

```bash
ISAACSIM_HEADLESS=1 \
PI05_CUDA_VISIBLE_DEVICES=1 \
PI05_RENDER_QUALITY=low \
PI05_EVAL_DIR="$PWD/artifacts/pi05_full" \
./scripts/eval_pi05_roco.sh \
  "$PWD/outputs/pi05/<run>/checkpoints/last/pretrained_model"
```

Do not report a baseline score until this corrected Euler/10 Hz rollout
finishes. Record the strategy, checkpoint step, effective batch size,
`results.json`, and rollout video with every score.

## 12. Troubleshooting

- **401/403 while loading weights:** accept model terms and make `HF_TOKEN`
  visible inside the interactive or SLURM job.
- **Missing quantiles:** use `PI05_NORMALIZATION=mean_std` for that dataset.
- **CUDA OOM:** reduce per-device batch size. Two-GPU DDP does not shard model
  memory.
- **SLURM preemption:** periodic checkpoints remain valid. Resubmit with
  `PI05_RESUME_CONFIG` pointing to the latest `train_config.json`.
- **Sidecar exits:** inspect `task/pi05_server.log`, verify
  `PI05_SERVER_PY`, and run `smoke_infer_pi05.py`.
- **Wrong orientation:** the pinned baseline defaults to intrinsic XYZ Euler.
  Ensure `PI05_ACTION_ROTATION` is unset or `euler_xyz`.
- **Inotify errno 28:** noisy but non-fatal in headless Isaac runs if rollout
  continues and artifacts are written.
- **Isaac wheel reports “Didn't find wheel”:** check `ldd --version`. Isaac Sim
  5.1 Linux wheels are tagged `manylinux_2_35` and cannot be installed on the
  current RHEL 8/glibc 2.28 login node. Training remains supported there; run
  simulator evaluation on a glibc 2.35+ host/container.
