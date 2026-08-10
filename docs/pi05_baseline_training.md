# pi0.5 Baseline Training Config

This is the reference config for reproducing the RoCo Industrial Assembly
pi0.5 baseline. Evaluation is documented separately in
[`pi05_eval.md`](pi05_eval.md).

## Reproduction Command

Run from the RoCo repo root:

```bash
HF_HOME=/path/to/.hf-cache \
LEROBOT_ROOT=/path/to/lerobot_roco_pi05 \
PI05_ACCELERATE_MODE=fsdp \
PI05_ACCELERATE_CONFIG=$PWD/scripts/accelerate_pi05_fsdp_4gpu.yaml \
PI05_NUM_PROCESSES=4 \
PI05_GPU_IDS=0,1,2,3 \
PI05_DATASET_REVISION=dc03b003f94d184b2b20465ed986456ee1bf2a3c \
PI05_PRETRAINED_REVISION=7de663972b7817d2c4cf2d84c821153dfea772e9 \
PI05_STEPS=30000 \
PI05_BATCH_SIZE=2 \
PI05_SAVE_FREQ=10000 \
PI05_LOG_FREQ=50 \
PI05_NUM_WORKERS=0 \
PI05_TRAIN_EXPERT_ONLY=false \
PI05_FREEZE_VISION_ENCODER=false \
PI05_PRETRAINED_PATH=lerobot/pi05_base \
PI05_OUTPUT_DIR=/path/to/outputs/roco_pi05_fft_30k_4gpu_bs2 \
PI05_JOB_NAME=roco_pi05_fft_30k_4gpu_bs2 \
./scripts/train_pi05_roco.sh
```

Expected checkpoints:

```text
.../checkpoints/010000/pretrained_model
.../checkpoints/020000/pretrained_model
.../checkpoints/030000/pretrained_model
```

Use `PI05_SAVE_FREQ=5000` if 5k-step checkpoints are desired.

## Dataset Config

| Key | Value |
|-----|-------|
| `--dataset.repo_id` | `rocochallenge2025/rocochallenge2026_Industrial_Assembly` |
| `--dataset.revision` | `dc03b003f94d184b2b20465ed986456ee1bf2a3c` |
| `--dataset.video_backend` | `pyav` |
| `dataset.streaming` | `false` |
| `dataset.eval_split` | `0.0` |
| `dataset.use_imagenet_stats` | `true` |
| `dataset.image_transforms.enable` | `false` |
| Episodes / frames / fps | `200` / `121454` / `10` |
| Robot type | `vega_1u` |

Input/output schema:

| Feature | Shape | Type |
|---------|-------|------|
| `observation.images.head` | `[3, 240, 320]` | `VISUAL` |
| `observation.images.left_hand` | `[3, 240, 320]` | `VISUAL` |
| `observation.images.right_hand` | `[3, 240, 320]` | `VISUAL` |
| `observation.state` | `[44]` | `STATE` |
| `action` | `[14]` | `ACTION` |

Action names:

```text
left_ee_x, left_ee_y, left_ee_z,
left_ee_rx, left_ee_ry, left_ee_rz, left_gripper,
right_ee_x, right_ee_y, right_ee_z,
right_ee_rx, right_ee_ry, right_ee_rz, right_gripper
```

## Policy Config

| Key | Value |
|-----|-------|
| `--policy.type` | `pi05` |
| `--policy.pretrained_path` | `lerobot/pi05_base` |
| `--policy.pretrained_revision` | `7de663972b7817d2c4cf2d84c821153dfea772e9` |
| `--policy.device` | `cuda` |
| `--policy.dtype` | `bfloat16` |
| `--policy.train_expert_only` | `false` |
| `--policy.freeze_vision_encoder` | `false` |
| `--policy.gradient_checkpointing` | `true` |
| `--policy.compile_model` | `false` |
| `policy.compile_mode` | `max-autotune` |
| `policy.paligemma_variant` | `gemma_2b` |
| `policy.action_expert_variant` | `gemma_300m` |
| `policy.image_resolution` | `[224, 224]` |
| `policy.tokenizer_max_length` | `200` |
| `policy.n_obs_steps` | `1` |
| `policy.n_action_steps` | `50` |
| `policy.chunk_size` | `50` |
| `policy.num_inference_steps` | `10` |
| `--policy.max_state_dim` | `44` |
| `--policy.max_action_dim` | `32` |
| `policy.empty_cameras` | `0` |
| `policy.use_relative_actions` | `false` |
| `policy.relative_exclude_joints` | `["gripper"]` |
| `policy.use_amp` | `false` |
| `policy.use_peft` | `false` |
| `policy.push_to_hub` | `false` |

Normalization:

| Type | Mapping |
|------|---------|
| `ACTION` | `QUANTILES` |
| `STATE` | `QUANTILES` |
| `VISUAL` | `IDENTITY` |

Time sampling:

| Key | Value |
|-----|-------|
| `policy.time_sampling_beta_alpha` | `1.5` |
| `policy.time_sampling_beta_beta` | `1.0` |
| `policy.time_sampling_offset` | `0.001` |
| `policy.time_sampling_scale` | `0.999` |
| `policy.min_period` | `0.004` |
| `policy.max_period` | `4.0` |

Training objective:

| Key | Value |
|-----|-------|
| Loss type | flow-matching MSE |
| Reduction | mean over time and original action dims |
| Action dims used in loss | first 14 dims, after truncating pi0.5 padding |
| L1 loss | no |

## Optimizer And Schedule

| Key | Value |
|-----|-------|
| `optimizer.type` | `adamw` |
| `optimizer.lr` / `scheduler.peak_lr` | `2.5e-5` |
| `optimizer.weight_decay` | `0.01` |
| `optimizer.betas` | `[0.9, 0.95]` |
| `optimizer.eps` | `1e-8` |
| `optimizer.grad_clip_norm` | `1.0` |
| `scheduler.type` | `cosine_decay_with_warmup` |
| `scheduler.num_warmup_steps` | `1000` |
| `scheduler.num_decay_steps` | `30000` |
| `scheduler.decay_lr` | `2.5e-6` |

## Trainer Config

| Key | Value |
|-----|-------|
| `steps` | `30000` |
| `batch_size` | `2` per process |
| Global batch size | `8` on 4 GPUs |
| `num_workers` | `0` |
| `prefetch_factor` | `4` |
| `persistent_workers` | `true` |
| `seed` | `1000` |
| `cudnn_deterministic` | `false` |
| `save_checkpoint` | `true` |
| `save_freq` | `10000` |
| `log_freq` | `50` |
| `env_eval_freq` | `0` |
| `eval_steps` | `0` |
| `max_eval_samples` | `0` |
| `wandb.enable` | `false` |
| `policy.push_to_hub` | `false` |
| `save_checkpoint_to_hub` | `false` |
| `sample_weighting` | `null` |
| `use_policy_training_preset` | `true` |

The local 30k run was resumed from the 20k checkpoint, so its saved
`train_config.json` contains `resume=true` and `checkpoint_path=.../020000`.
For a fresh reproduction from `lerobot/pi05_base`, keep the command above.

## Accelerate FSDP Config

Use [`scripts/accelerate_pi05_fsdp_4gpu.yaml`](../scripts/accelerate_pi05_fsdp_4gpu.yaml):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_processes: 4
use_cpu: false
gpu_ids: "0,1,2,3"
machine_rank: 0
main_training_function: main
num_machines: 1
rdzv_backend: static
same_network: true
fsdp_config:
  fsdp_version: 1
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: "Linear,Embedding"
  fsdp_use_orig_params: true
  fsdp_state_dict_type: FULL_STATE_DICT
```

## Minimal Requirements

| Requirement | Value |
|-------------|-------|
| LeRobot commit used locally | `8a74e0ac6d01706d67fddfed682a09d694d9c8c0` |
| LeRobot command | `uv run python -m lerobot.scripts.lerobot_train` works |
| pi0.5 dependencies | installed in the LeRobot environment |
| Hugging Face auth | token can access `lerobot/pi05_base` and `google/paligemma-3b-pt-224` |
| RoCo launcher | `scripts/train_pi05_roco.sh` |

Apply the local pi0.5/FSDP patch before the 4-GPU FFT run:

```bash
cd /path/to/lerobot_roco_pi05
git checkout 8a74e0ac6d01706d67fddfed682a09d694d9c8c0
git apply /path/to/RoCo_TaskBoardAssembly/docs/lerobot_pi05_fsdp.patch
```
