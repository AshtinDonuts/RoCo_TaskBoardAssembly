#!/usr/bin/env bash
# Collect randomized 35ec027 rollouts as LeRobot v3 episodes and derive
# successful-part-only subtask episodes.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
start_seed=0
seed_count=30
output_dir=""
sample_fps=10
repo_id="taskboard/aware_35ec027_full_assembly"
force=0

while (($#)); do
  case "$1" in
    --start-seed) start_seed="$2"; shift 2 ;;
    --count) seed_count="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --fps) sample_fps="$2"; shift 2 ;;
    --repo-id) repo_id="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help)
      cat <<'EOF'
Collect randomized 35ec027 rollouts and split successful part episodes.

Usage: scripts/run_aware_35ec027_seed_batch.sh [options]

  --start-seed N   First seed (default: 0)
  --count N        Number of seeds (default: 30)
  --output-dir DIR Output root; defaults to an artifacts/lerobot seed range
  --fps N          LeRobot recording FPS (default: 10)
  --repo-id ID     Local LeRobot repository ID
  --force          Delete and rebuild the entire output root
EOF
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! [[ "${start_seed}" =~ ^[0-9]+$ && "${seed_count}" =~ ^[1-9][0-9]*$ && "${sample_fps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "start seed must be nonnegative; count and fps must be positive integers" >&2
  exit 2
fi

end_seed=$((start_seed + seed_count - 1))
if [[ -z "${output_dir}" ]]; then
  output_dir="${repo_root}/artifacts/lerobot/aware_35ec027_timeout120_seeds${start_seed}-${end_seed}"
elif [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi
output_dir="$(realpath -m -- "${output_dir}")"
if [[ "${output_dir}" == "/" || "${output_dir}" == "${repo_root}" || "${output_dir}" == "$(dirname -- "${repo_root}")" ]]; then
  echo "refusing unsafe output directory: ${output_dir}" >&2
  exit 2
fi

source_dir="${output_dir}/source"
results_dir="${output_dir}/results"
logs_dir="${output_dir}/logs"
derived_dir="${output_dir}/derived_successful_subtasks"
batch_log="${output_dir}/batch.log"
summary_json="${output_dir}/summary.json"
batch_marker="${output_dir}/.roco_lerobot_batch"

if ((force)) && [[ -e "${output_dir}" ]]; then
  if [[ ! -f "${batch_marker}" ]]; then
    echo "refusing to delete an output directory without ${batch_marker}" >&2
    exit 2
  fi
  rm -rf -- "${output_dir}"
fi
mkdir -p "${results_dir}" "${logs_dir}"
touch "${batch_marker}"

cd "${repo_root}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export TASK_ENABLE_CAMERA_VIEWPORTS="${TASK_ENABLE_CAMERA_VIEWPORTS:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export TASK_BASELINE_MOTION_PROFILE="settings35ec027_timeout120"

{
  echo "===== BATCH START $(date -Is) ====="
  echo "start_seed=${start_seed} count=${seed_count} output_dir=${output_dir} fps=${sample_fps}"
  echo "profile=${TASK_BASELINE_MOTION_PROFILE} repo_id=${repo_id}"
} | tee -a "${batch_log}"

for ((i = 0; i < seed_count; i++)); do
  seed=$((start_seed + i))
  seed_tag="$(printf '%03d' "${seed}")"
  result_path="${results_dir}/seed-${seed_tag}.json"
  pending_path="${results_dir}/seed-${seed_tag}.pending.json"
  log_path="${logs_dir}/seed-${seed_tag}.log"

  if [[ -s "${result_path}" && ! -e "${pending_path}" ]]; then
    echo "[batch] seed ${seed}: finalized; skipping" | tee -a "${batch_log}"
    continue
  fi
  echo "[batch] seed ${seed}: collecting $(date -Is)" | tee -a "${batch_log}"
  set +e
  uv run --group collection python task/collect_lerobot_v3.py \
    --output-root "${source_dir}" --repo-id "${repo_id}" \
    --policy policies.baseline_scripted.BaselinePolicy \
    --random-seed "${seed}" --sample-hz "${sample_fps}" \
    --results-json "${result_path}" >"${log_path}" 2>&1
  ec=$?
  set -e
  echo "[batch] seed ${seed}: exit=${ec} $(date -Is)" | tee -a "${batch_log}"
  if ((ec != 0)); then
    echo "[batch] seed ${seed}: FAILED — continuing collection" | tee -a "${batch_log}"
  fi
done

set +e
python3 - "${source_dir}" "${results_dir}" "${summary_json}" "${start_seed}" "${seed_count}" <<'PY'
import json, os, pathlib, sys
source, results_dir, summary_path = map(pathlib.Path, sys.argv[1:4])
start, count = int(sys.argv[4]), int(sys.argv[5])
requested = set(range(start, start + count))
rows, errors = [], []
for path in sorted(results_dir.glob("seed-*.json")):
    if path.name.endswith(".pending.json"):
        continue
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("completion_reason") != "complete" or len(row.get("segments", [])) != 9:
            raise ValueError("result is not a complete nine-part episode")
        rows.append(row)
    except Exception as exc:
        errors.append(f"{path}: {exc}")
episodes = [int(row["episode_index"]) for row in rows]
seeds = [int(row["seed"]) for row in rows]
if len(episodes) != len(set(episodes)): errors.append("duplicate episode_index")
if len(seeds) != len(set(seeds)): errors.append("duplicate seed")
if sorted(episodes) != list(range(len(episodes))): errors.append(f"non-dense episodes: {sorted(episodes)}")
info_path = source / "meta/info.json"
source_count = int(json.loads(info_path.read_text())["total_episodes"]) if info_path.is_file() else None
if source_count != len(rows): errors.append(f"source has {source_count} episodes, results have {len(rows)}")
missing = sorted(requested - set(seeds))
if not errors:
    manifest = source / "meta/roco_rollouts.jsonl"
    temp = manifest.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(rows, key=lambda x: int(x["episode_index"]))), encoding="utf-8")
    os.replace(temp, manifest)
summary = {
    "status": "incomplete" if missing or errors else "collected",
    "start_seed": start, "seed_count": count, "source_episodes": source_count,
    "missing_requested_seeds": missing, "errors": errors,
    "n_pass_parts": sum(int(row.get("n_pass", 0)) for row in rows),
    "n_fail_parts": sum(int(row.get("n_fail", 0)) for row in rows),
    "n_missing_parts": sum(int(row.get("n_missing", 0)) for row in rows),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
raise SystemExit(1 if missing or errors else 0)
PY
validate_ec=$?
set -e
if ((validate_ec != 0)); then
  echo "[batch] collection incomplete; source is resumable and splitting was skipped" | tee -a "${batch_log}"
  exit 1
fi

echo "[batch] splitting successful parts $(date -Is)" | tee -a "${batch_log}"
uv run --group collection python split_lerobot_subtasks.py \
  "${source_dir}" "${derived_dir}" \
  --rollout-manifest "${source_dir}/meta/roco_rollouts.jsonl" \
  --successful-parts-only --replace | tee "${output_dir}/split_summary.json"

python3 - "${summary_json}" "${output_dir}/split_summary.json" <<'PY'
import json, os, pathlib, sys
summary_path, split_path = map(pathlib.Path, sys.argv[1:3])
summary = json.loads(summary_path.read_text())
summary["status"] = "complete"
summary["derived"] = json.loads(split_path.read_text())
temp = summary_path.with_suffix(".tmp")
temp.write_text(json.dumps(summary, indent=2) + "\n")
os.replace(temp, summary_path)
PY

echo "===== BATCH END $(date -Is) =====" | tee -a "${batch_log}"
echo "[batch] source: ${source_dir}"
echo "[batch] successful subtasks: ${derived_dir}"
echo "[batch] summary: ${summary_json}"
