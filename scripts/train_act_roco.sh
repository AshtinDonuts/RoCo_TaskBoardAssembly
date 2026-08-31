#!/usr/bin/env bash
# Train a per-part LeRobot ACT policy on the derived RoCo subtask dataset.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
models_root="$(cd -- "${repo_root}/.." && pwd)"

usage() {
  cat <<'EOF'
Train one RoCo ACT skill with LeRobot.

Usage: scripts/train_act_roco.sh [--dry-run] [LeRobot training overrides]

Select a subtask with ACT_PART (recommended) or ACT_TASK_ID (episode-index
stride, compatible with the original trainer). Configuration is supplied by
ACT_* environment variables; see docs/act_training.md for the complete list.

Examples:
  ACT_PART=gear_60teeth ./scripts/train_act_roco.sh --dry-run
  ACT_PART=rod_16mm ACT_CUDA_VISIBLE_DEVICES=1 ./scripts/train_act_roco.sh
EOF
}

dry_run="${ACT_DRY_RUN:-0}"
extra_args=()
for arg in "$@"; do
  case "${arg}" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      dry_run=1
      ;;
    *)
      extra_args+=("${arg}")
      ;;
  esac
done

lerobot_root="${LEROBOT_ROOT:-${models_root}/lerobot_roco_pi05}"
dataset_root="${ACT_DATASET_ROOT:-${repo_root}/artifacts/lerobot/aware_35ec027_timeout120_seeds100-199/derived_subtasks}"
manifest="${ACT_SUBTASKS_MANIFEST:-${dataset_root}/meta/roco_subtasks.jsonl}"
dataset_repo_id="${ACT_DATASET_REPO_ID:-local/aware_35ec027_timeout120_seeds100-199_subtasks}"
task_id="${ACT_TASK_ID:-0}"
part="${ACT_PART:-}"
pass_only="${ACT_PASS_ONLY:-1}"

if ! [[ "${task_id}" =~ ^[0-8]$ ]]; then
  echo "ACT_TASK_ID must be an integer from 0 through 8; got ${task_id}" >&2
  exit 2
fi
if [[ -n "${part}" && ! "${part}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "ACT_PART contains unsupported characters: ${part}" >&2
  exit 2
fi
case "${pass_only,,}" in
  1|true|yes) pass_only=1 ;;
  0|false|no) pass_only=0 ;;
  *) echo "ACT_PASS_ONLY must be true/false or 1/0; got ${pass_only}" >&2; exit 2 ;;
esac

if [[ ! -d "${lerobot_root}" ]]; then
  echo "LeRobot checkout not found: ${lerobot_root}" >&2
  echo "Set LEROBOT_ROOT to the checkout containing the ACT training environment." >&2
  exit 2
fi
if [[ ! -d "${dataset_root}" ]]; then
  echo "ACT dataset not found: ${dataset_root}" >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  echo "Subtask manifest not found: ${manifest}" >&2
  exit 2
fi

# Select by explicit part name when provided. The task-id path intentionally
# retains the source trainer's episode_index % 9 behaviour, while checking
# that the selected stride really represents only one part.
episodes="$({ python3 - "${manifest}" "${task_id}" "${part}" "${pass_only}" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
task_id = int(sys.argv[2])
requested_part = sys.argv[3]
pass_only = bool(int(sys.argv[4]))

rows = []
with manifest.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{manifest}:{line_number}: invalid JSON: {exc}") from exc
        if requested_part:
            selected = row.get("part") == requested_part
        else:
            selected = int(row["episode_index"]) % 9 == task_id
        if selected and (not pass_only or bool(row.get("pass"))):
            rows.append(row)

if not rows:
    selector = f"part={requested_part}" if requested_part else f"task_id={task_id}"
    suffix = " with pass=True" if pass_only else ""
    raise SystemExit(f"no episodes found for {selector}{suffix} in {manifest}")

parts = sorted({str(row.get("part", "")) for row in rows})
if len(parts) != 1:
    raise SystemExit(
        f"task_id={task_id} selects multiple parts {parts}; set ACT_PART explicitly"
    )

episodes = sorted({int(row["episode_index"]) for row in rows})
print(
    f"Selected {len(episodes)} episodes for {parts[0]} "
    f"({'pass only' if pass_only else 'pass and fail'})",
    file=sys.stderr,
)
print(json.dumps(episodes, separators=(",", ":")))
PY
} )"

selector="task_${task_id}"
if [[ -n "${part}" ]]; then
  selector="${part}"
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${ACT_OUTPUT_DIR:-${lerobot_root}/outputs/roco_act_aware100-199_${selector}_${timestamp}}"
job_name="${ACT_JOB_NAME:-roco_act_aware100-199_${selector}}"

chunk_size="${ACT_CHUNK_SIZE:-9}"
n_action_steps="${ACT_N_ACTION_STEPS:-${chunk_size}}"
batch_size="${ACT_BATCH_SIZE:-64}"
num_workers="${ACT_NUM_WORKERS:-8}"
steps="${ACT_STEPS:-15000}"
eval_split="${ACT_EVAL_SPLIT:-0.1}"
eval_steps="${ACT_EVAL_STEPS:-5000}"
max_eval_samples="${ACT_MAX_EVAL_SAMPLES:-128}"
save_freq="${ACT_SAVE_FREQ:-5000}"
log_freq="${ACT_LOG_FREQ:-100}"
use_amp="${ACT_USE_AMP:-true}"
wandb_enable="${ACT_WANDB_ENABLE:-false}"

export CUDA_VISIBLE_DEVICES="${ACT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${ACT_PYTHON:-}" ]]; then
  train_python=("${ACT_PYTHON}")
elif [[ -x "${lerobot_root}/.venv/bin/python" ]]; then
  train_python=("${lerobot_root}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  train_python=(uv run python)
else
  echo "No LeRobot Python found; set ACT_PYTHON or create ${lerobot_root}/.venv." >&2
  exit 2
fi

train_args=(
  --dataset.repo_id="${dataset_repo_id}"
  --dataset.root="${dataset_root}"
  --dataset.episodes="${episodes}"
  --dataset.eval_split="${eval_split}"
  --dataset.video_backend=pyav
  --policy.type=act
  --policy.device=cuda
  --policy.chunk_size="${chunk_size}"
  --policy.n_action_steps="${n_action_steps}"
  --policy.vision_backbone=resnet18
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.use_amp="${use_amp}"
  --policy.push_to_hub=false
  --batch_size="${batch_size}"
  --num_workers="${num_workers}"
  --steps="${steps}"
  --eval_steps="${eval_steps}"
  --max_eval_samples="${max_eval_samples}"
  --save_freq="${save_freq}"
  --log_freq="${log_freq}"
  --env_eval_freq=0
  --wandb.enable="${wandb_enable}"
  --wandb.project="${ACT_WANDB_PROJECT:-roco}"
  --job_name="${job_name}"
  --output_dir="${output_dir}"
)
if [[ -n "${ACT_WANDB_ENTITY:-}" ]]; then
  train_args+=(--wandb.entity="${ACT_WANDB_ENTITY}")
fi
train_args+=("${extra_args[@]}")

command=("${train_python[@]}" -m lerobot.scripts.lerobot_train "${train_args[@]}")
echo "LeRobot: ${lerobot_root}"
echo "Dataset: ${dataset_root}"
echo "Output:   ${output_dir}"
printf 'Command:  '
printf '%q ' "${command[@]}"
printf '\n'

if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

cd "${lerobot_root}"
exec "${command[@]}"
