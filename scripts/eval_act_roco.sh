#!/usr/bin/env bash
# Evaluate one LeRobot ACT skill on a single RoCo subtask with optional
# fairness XY randomization. Isolates the run via ROCO_PART_ORDER.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lerobot_root="${LEROBOT_ROOT:-/home/khw/lerobot}"

ckpt="${1:-${ACT_CKPT:-}}"
part="${2:-${ACT_PART:-}}"
if [[ -z "${ckpt}" || -z "${part}" ]]; then
  cat >&2 <<'EOF'
usage: scripts/eval_act_roco.sh <pretrained_model_dir> <part_name> [extra run_roco args...]

Env overrides:
  ACT_RANDOM_SEED              fairness XY seed (default: 101)
  ACT_BLIND_XY                 1/0; blind policy to offsets (default: 1)
  ACT_NOMINAL                  1/0; omit all XY randomization (default: 0)
  ACT_PREPLACE_PREVIOUS_SUCCESS
                                1/0; place all earlier canonical parts at their
                                successful endpoints before this rollout (default: 0)
  ACT_EVAL_RESULTS / ACT_EVAL_VIDEO
  ACT_CUDA_VISIBLE_DEVICES     GPU for ACT sidecar (default: 0)
  ISAACSIM_*                   passed through to run_roco.sh
EOF
  exit 2
fi
shift 2

export ACT_CKPT="${ckpt}"
export ACT_SERVER_PY="${ACT_SERVER_PY:-${lerobot_root}/.venv/bin/python}"
export ACT_CUDA_VISIBLE_DEVICES="${ACT_CUDA_VISIBLE_DEVICES:-0}"
export ACT_SERVER_LOG="${ACT_SERVER_LOG:-${repo_root}/artifacts/act_server_${part}.log}"
export ROCO_PART_ORDER="${part}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

random_seed="${ACT_RANDOM_SEED:-101}"
blind="${ACT_BLIND_XY:-1}"
nominal="${ACT_NOMINAL:-0}"
preplace_previous="${ACT_PREPLACE_PREVIOUS_SUCCESS:-0}"
results="${ACT_EVAL_RESULTS:-${repo_root}/artifacts/act_eval_${part}_seed${random_seed}_results.json}"
video="${ACT_EVAL_VIDEO:-${repo_root}/artifacts/act_eval_${part}_seed${random_seed}_head.mp4}"

extra_args=(
  --policy policies.act_lerobot.ACTLeRobotPolicy
  --results-json "${results}"
  --record-video "${video}"
  --record-video-camera "${ACT_EVAL_CAMERA:-head}"
  --record-video-fps "${ACT_EVAL_FPS:-15}"
  --max-parts "${ACT_EVAL_MAX_PARTS:-1}"
)
if [[ "${nominal}" != "1" && "${nominal}" != "true" ]]; then
  extra_args+=(--random-seed "${random_seed}")
  if [[ "${blind}" == "1" || "${blind}" == "true" ]]; then
    extra_args+=(--blind-to-xy-randomization)
  fi
fi
if [[ -n "${ACT_EVAL_MAX_STEPS:-}" ]]; then
  extra_args+=(--max-steps "${ACT_EVAL_MAX_STEPS}")
fi
if [[ -n "${ACT_EVAL_MAX_SIM_SECONDS:-}" ]]; then
  extra_args+=(--max-sim-seconds "${ACT_EVAL_MAX_SIM_SECONDS}")
fi
if [[ "${preplace_previous}" == "1" || "${preplace_previous}" == "true" ]]; then
  extra_args+=(--preplace-previous-success)
fi
extra_args+=("$@")

echo "[eval_act] part=${part}"
echo "[eval_act] ckpt=${ckpt}"
echo "[eval_act] nominal=${nominal} random_seed=${random_seed} blind=${blind} preplace_previous=${preplace_previous}"
echo "[eval_act] results=${results}"
echo "[eval_act] video=${video}"

cd "${repo_root}"
exec ./scripts/run_roco.sh "${extra_args[@]}"
