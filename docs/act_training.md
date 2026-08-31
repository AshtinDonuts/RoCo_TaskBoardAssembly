# ACT Skill Training

[`scripts/train_act_roco.sh`](../scripts/train_act_roco.sh) ports the ACT
training setup used for the 30 Hz `aware_35ec027_timeout120_seeds100-199`
subtask dataset into a portable launcher. It trains one per-part LeRobot ACT
checkpoint at a time and defaults to the source setup: successful episodes
only, a 9-frame action chunk, AMP, batch size 64, and 15,000 optimization
steps.

## Prerequisites

- A LeRobot checkout with its training environment. By default the launcher
  uses `../lerobot_roco_pi05`; set `LEROBOT_ROOT` when it lives elsewhere.
- The derived subtask dataset. The default is produced by:

  ```bash
  ./scripts/run_aware_35ec027_seed_batch.sh --start-seed 100 --count 100
  ```

  The expected dataset directory is
  `artifacts/lerobot/aware_35ec027_timeout120_seeds100-199/derived_subtasks`.

## Train a Skill

Select by part name so training remains correct if the harness part order
changes:

```bash
ACT_PART=rod_16mm \
ACT_CUDA_VISIBLE_DEVICES=0 \
./scripts/train_act_roco.sh
```

Inspect the resolved episode list and command without starting training:

```bash
ACT_PART=rod_16mm ./scripts/train_act_roco.sh --dry-run
```

`ACT_TASK_ID=0` through `8` is also supported for compatibility with the
source script. Task-ID selection uses `episode_index % 9` and verifies that
the selected stride contains only one part. In the current aware dataset the
strides are:

| Task ID | Part |
|---:|---|
| 0 | `gear_60teeth` |
| 1 | `gear_20teeth` |
| 2 | `rod_16mm` |
| 3 | `bolt_8mm` |
| 4 | `usb_a` |
| 5 | `hdmi` |
| 6 | `pin` |
| 7 | `battery_size1` |
| 8 | `battery_size5` |

To train all nine skills sequentially:

```bash
for part in \
  gear_60teeth gear_20teeth rod_16mm bolt_8mm usb_a hdmi pin \
  battery_size1 battery_size5
do
  ACT_PART="${part}" ./scripts/train_act_roco.sh
done
```

## Configuration

The launcher accepts these environment overrides:

| Variable | Default |
|---|---|
| `ACT_PART` | unset; select `ACT_TASK_ID` instead |
| `ACT_TASK_ID` | `0` |
| `ACT_PASS_ONLY` | `1` |
| `ACT_DATASET_ROOT` | aware seeds 100–199 derived dataset in `artifacts/lerobot` |
| `ACT_SUBTASKS_MANIFEST` | `<dataset>/meta/roco_subtasks.jsonl` |
| `ACT_DATASET_REPO_ID` | `local/aware_35ec027_timeout120_seeds100-199_subtasks` |
| `ACT_CHUNK_SIZE` / `ACT_N_ACTION_STEPS` | `9` / `9` |
| `ACT_BATCH_SIZE` / `ACT_NUM_WORKERS` | `64` / `8` |
| `ACT_STEPS` | `15000` |
| `ACT_EVAL_SPLIT` / `ACT_EVAL_STEPS` | `0.1` / `5000` |
| `ACT_MAX_EVAL_SAMPLES` | `128` |
| `ACT_SAVE_FREQ` / `ACT_LOG_FREQ` | `5000` / `100` |
| `ACT_USE_AMP` | `true` |
| `ACT_CUDA_VISIBLE_DEVICES` | existing `CUDA_VISIBLE_DEVICES`, otherwise `0` |
| `ACT_OUTPUT_DIR` | timestamped directory under `<LEROBOT_ROOT>/outputs` |
| `ACT_PYTHON` | `<LEROBOT_ROOT>/.venv/bin/python`, then `uv run python` fallback |
| `ACT_WANDB_ENABLE` | `false` |
| `ACT_WANDB_PROJECT` / `ACT_WANDB_ENTITY` | `roco` / unset |

Additional arguments are appended to the LeRobot command, so an individual
run can override or extend its trainer configuration:

```bash
ACT_PART=usb_a ./scripts/train_act_roco.sh \
  --optimizer.lr=0.00005 --steps=20000
```

Checkpoints are written beneath `ACT_OUTPUT_DIR`, normally at steps 5,000,
10,000, and 15,000.
