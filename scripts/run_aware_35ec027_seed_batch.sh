#!/usr/bin/env bash
# Offset-aware baseline rollouts with 35ec027 motion settings and a
# 2-minute per-subtask timeout (PER_PART_TIMEOUT_STEPS=24000).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

start_seed=0
seed_count=30
output_dir="${repo_root}/artifacts/randomized_rollouts/aware_settings35ec027_timeout120_seeds0-29"
camera="head"
video_fps=15
force=0

while (($#)); do
  case "$1" in
    --start-seed) start_seed="$2"; shift 2 ;;
    --count) seed_count="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --camera) camera="$2"; shift 2 ;;
    --fps) video_fps="$2"; shift 2 ;;
    --force) force=1; shift ;;
    -h|--help)
      cat <<'EOF'
Run offset-aware seed0..N-1 rollouts with 35ec027 baseline settings.

Usage:
  scripts/run_aware_35ec027_seed_batch.sh [options]

Options:
  --start-seed N   First seed (default: 0)
  --count N        Number of seeds (default: 30)
  --output-dir DIR Output root
  --camera NAME    head, L_wrist, or R_wrist (default: head)
  --fps N          Recorded video FPS (default: 15)
  --force          Rerun and overwrite completed seed outputs
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi

videos_dir="${output_dir}/videos"
results_dir="${output_dir}/results"
traj_dir="${output_dir}/trajectories"
logs_dir="${output_dir}/logs"
mkdir -p "${videos_dir}" "${results_dir}" "${traj_dir}" "${logs_dir}"

batch_log="${output_dir}/batch.log"
summary_json="${output_dir}/summary.json"

cd "${repo_root}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export TASK_ENABLE_CAMERA_VIEWPORTS="${TASK_ENABLE_CAMERA_VIEWPORTS:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# Enforce 35ec027-matching motion flags + 120s subtask timeout.
python3 - task/param_config.py <<'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
repls = {
    "ENABLE_SMOOTH_RETURN_HOME": False,
    "ENABLE_BASELINE_JOINT_RATE_LIMIT": False,
    "ENABLE_SAFE_RETRACT": False,
}
for key, val in repls.items():
    text2, n = re.subn(
        rf"^({key}\s*=\s*)(True|False)\s*$",
        rf"\g<1>{val}",
        text,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"failed to set {key} (n={n})")
    text = text2
text2, n = re.subn(
    r"^(PER_PART_TIMEOUT_STEPS\s*=\s*)\d+(\s*.*)$",
    r"\g<1>24000\2",
    text,
    count=1,
    flags=re.M,
)
if n != 1:
    raise SystemExit("failed to set PER_PART_TIMEOUT_STEPS")
path.write_text(text2)
print("flags: SMOOTH=False RATE=False SAFE=False TIMEOUT=24000")
PY

{
  echo "===== BATCH START $(date -Is) ====="
  echo "start_seed=${start_seed} count=${seed_count} output_dir=${output_dir}"
  echo "settings: ENABLE_SMOOTH_RETURN_HOME=False ENABLE_BASELINE_JOINT_RATE_LIMIT=False ENABLE_SAFE_RETRACT=False PER_PART_TIMEOUT_STEPS=24000 CARTESIAN_MAX_EE_SPEED_M_S=0.30"
} | tee -a "${batch_log}"

for ((i = 0; i < seed_count; i++)); do
  seed=$((start_seed + i))
  seed_tag="$(printf '%03d' "${seed}")"
  video_path="${videos_dir}/seed-${seed_tag}.mp4"
  results_path="${results_dir}/seed-${seed_tag}_results.json"
  traj_path="${traj_dir}/seed-${seed_tag}_trajectory.csv"
  log_path="${logs_dir}/seed-${seed_tag}.log"

  if ((force == 0)) && [[ -s "${results_path}" && -s "${video_path}" ]]; then
    echo "[batch] seed ${seed}: complete; skipping" | tee -a "${batch_log}"
    continue
  fi

  echo "[batch] seed ${seed}: running $(date -Is)" | tee -a "${batch_log}"
  set +e
  "${repo_root}/scripts/run_roco.sh" \
    --policy policies.baseline_scripted.BaselinePolicy \
    --random-seed "${seed}" \
    --record-video "${video_path}" \
    --record-video-camera "${camera}" \
    --record-video-fps "${video_fps}" \
    --results-json "${results_path}" \
    --trajectory-csv "${traj_path}" \
    >"${log_path}" 2>&1
  ec=$?
  set -e
  echo "[batch] seed ${seed}: exit=${ec} $(date -Is)" | tee -a "${batch_log}"
  if ((ec != 0)); then
    echo "[batch] seed ${seed}: FAILED — continuing" | tee -a "${batch_log}"
  fi
done

python3 - "${output_dir}" "${start_seed}" "${seed_count}" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
start = int(sys.argv[2])
count = int(sys.argv[3])
rows = []
for i in range(count):
    seed = start + i
    tag = f"{seed:03d}"
    p = out / "results" / f"seed-{tag}_results.json"
    row = {"seed": seed, "results_path": str(p), "status": "missing"}
    if p.exists() and p.stat().st_size > 0:
        try:
            d = json.loads(p.read_text())
            m = d.get("metadata", {})
            row.update({
                "status": "ok",
                "n_pass": d.get("n_pass"),
                "n_fail": d.get("n_fail"),
                "n_missing": d.get("n_missing"),
                "completed_parts": m.get("completed_parts"),
                "total_task_steps": m.get("total_task_steps"),
                "sim_time_s": m.get("sim_time_s"),
                "completion_reason": m.get("completion_reason"),
                "blind_to_xy_randomization": m.get("blind_to_xy_randomization"),
                "snap_fired_parts": m.get("snap_fired_parts"),
                "failed_parts": [
                    part["name"] for part in d.get("per_part", [])
                    if not part.get("pass")
                ],
            })
        except Exception as e:
            row["status"] = f"parse_error:{e}"
    rows.append(row)
ok = [r for r in rows if r["status"] == "ok"]
summary = {
    "experiment": "aware_settings35ec027_timeout120",
    "settings": {
        "ENABLE_SMOOTH_RETURN_HOME": False,
        "ENABLE_BASELINE_JOINT_RATE_LIMIT": False,
        "ENABLE_SAFE_RETRACT": False,
        "PER_PART_TIMEOUT_STEPS": 24000,
        "CARTESIAN_MAX_EE_SPEED_M_S": 0.30,
        "blind_to_xy_randomization": False,
    },
    "start_seed": start,
    "seed_count": count,
    "n_ok": len(ok),
    "n_missing_or_failed": count - len(ok),
    "mean_n_pass": (sum(r["n_pass"] for r in ok) / len(ok)) if ok else None,
    "seeds": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({
    "n_ok": summary["n_ok"],
    "n_missing_or_failed": summary["n_missing_or_failed"],
    "mean_n_pass": summary["mean_n_pass"],
}, indent=2))
PY

echo "===== BATCH END $(date -Is) =====" | tee -a "${batch_log}"
echo "[batch] summary: ${summary_json}"
echo "[batch] output: ${output_dir}"
