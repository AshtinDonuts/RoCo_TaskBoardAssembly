#!/usr/bin/env bash
# Capture nominal + seeded fairness RGB-D snapshots for offline offset eval.
#
# IMPORTANT: each seed relaunches Isaac Sim. Back-to-back Kit processes will
# OOM/crash this machine (swap fills). Prefer:
#   --no-video --cooldown-sec 45 --retries 2
# or the chunked wrapper:
#   scripts/capture_randomization_chunked.sh --start-seed 0 --count 100
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

start_seed=0
seed_count=10
output_dir="${repo_root}/artifacts/randomization-final-frames"
camera="head"
video_fps=15
max_steps=3
policy="policies.baseline_scripted.BaselinePolicy"
include_reference=1
force=0
record_video=1
cooldown_sec=30
retries=2
continue_on_error=1
min_free_mem_mb=12000

while (($#)); do
  case "$1" in
    --start-seed)
      start_seed="$2"
      shift 2
      ;;
    --count)
      seed_count="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --camera)
      camera="$2"
      shift 2
      ;;
    --fps)
      video_fps="$2"
      shift 2
      ;;
    --max-steps)
      max_steps="$2"
      shift 2
      ;;
    --policy)
      policy="$2"
      shift 2
      ;;
    --no-reference)
      include_reference=0
      shift
      ;;
    --no-video)
      record_video=0
      shift
      ;;
    --cooldown-sec)
      cooldown_sec="$2"
      shift 2
      ;;
    --retries)
      retries="$2"
      shift 2
      ;;
    --min-free-mem-mb)
      min_free_mem_mb="$2"
      shift 2
      ;;
    --fail-fast)
      continue_on_error=0
      shift
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Generate a nominal camera reference and seeded fairness-randomization frames.

Usage:
  scripts/generate_randomization_final_frames.sh [options]

Options:
  --start-seed N        First seed (default: 0)
  --count N             Number of seeds (default: 10)
  --output-dir DIR      Output root (default: artifacts/randomization-final-frames)
  --camera NAME         head, L_wrist, or R_wrist (default: head)
  --fps N               Recorded video FPS (default: 15)
  --max-steps N         Control steps before snapshot (default: 3)
  --policy PATH         Dotted policy class used during capture
  --no-reference        Do not capture/rebuild the nominal reference
  --no-video            Skip MP4/ffmpeg (recommended; NPZ+JSON are enough
                        for estimator accuracy). Much less likely to OOM.
  --cooldown-sec N      Sleep between Isaac launches (default: 30)
  --retries N           Retries per seed after a failed launch (default: 2)
  --min-free-mem-mb N   Wait until this much MemAvailable (default: 12000)
  --fail-fast           Abort the batch on the first failed seed
  --force               Rerun and overwrite completed seed outputs
  -h, --help            Show this help

Do NOT fire 100 Kit relaunches with no cooldown — Isaac will crash once
RAM/swap is exhausted. For large sweeps use:
  scripts/capture_randomization_chunked.sh --start-seed 0 --count 100
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "use --help for usage" >&2
      exit 2
      ;;
  esac
done

if ! [[ "${start_seed}" =~ ^[0-9]+$ ]]; then
  echo "--start-seed must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${seed_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--count must be a positive integer" >&2
  exit 2
fi
if ! [[ "${video_fps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--fps must be a positive integer" >&2
  exit 2
fi
if ! [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-steps must be a positive integer" >&2
  exit 2
fi
if ! [[ "${cooldown_sec}" =~ ^[0-9]+$ ]]; then
  echo "--cooldown-sec must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${retries}" =~ ^[0-9]+$ ]]; then
  echo "--retries must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${min_free_mem_mb}" =~ ^[0-9]+$ ]]; then
  echo "--min-free-mem-mb must be a non-negative integer" >&2
  exit 2
fi
case "${camera}" in
  head|L_wrist|R_wrist) ;;
  *)
    echo "--camera must be head, L_wrist, or R_wrist" >&2
    exit 2
    ;;
esac

if [[ "${output_dir}" != /* ]]; then
  output_dir="${repo_root}/${output_dir}"
fi

if ((record_video == 1)); then
  command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg is required unless --no-video is set" >&2
    exit 1
  }
fi

videos_dir="${output_dir}/videos"
frames_dir="${output_dir}/frames"
results_dir="${output_dir}/results"
logs_dir="${output_dir}/logs"
observations_dir="${output_dir}/observations"
mkdir -p "${videos_dir}" "${frames_dir}" "${results_dir}" "${logs_dir}" \
  "${observations_dir}"

cd "${repo_root}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
export TASK_ENABLE_CAMERA_OUTPUT=1

_available_mem_mb() {
  # Prefer MemAvailable (accounts for reclaimable cache).
  awk '/MemAvailable:/ {print int($2/1024); exit}' /proc/meminfo
}

_wait_for_memory() {
  local free_mb
  free_mb="$(_available_mem_mb)"
  if ((free_mb >= min_free_mem_mb)); then
    return 0
  fi
  echo "[batch] waiting for memory: available=${free_mb} MiB < ${min_free_mem_mb} MiB"
  local waited=0
  while ((free_mb < min_free_mem_mb)); do
    sleep 10
    waited=$((waited + 10))
    free_mb="$(_available_mem_mb)"
    if ((waited % 30 == 0)); then
      echo "[batch] still waiting: available=${free_mb} MiB (${waited}s)"
    fi
    # Cap wait so a permanently low-memory machine still progresses.
    if ((waited >= 300)); then
      echo "[batch] WARN: proceeding with available=${free_mb} MiB after ${waited}s"
      break
    fi
  done
}

_cleanup_isaac_leftovers() {
  # Best-effort: reap orphaned Kit/python children from a prior crash.
  pkill -f "task/run_pick_place.py" 2>/dev/null || true
  pkill -f "isaacsim" 2>/dev/null || true
  sleep 2
}

_seed_complete() {
  local observation_path="$1"
  local results_path="$2"
  local frame_path="$3"
  if [[ ! -s "${observation_path}" || ! -s "${results_path}" ]]; then
    return 1
  fi
  if ((record_video == 1)) && [[ ! -s "${frame_path}" ]]; then
    return 1
  fi
  return 0
}

if ((include_reference == 1)); then
  reference_dir="${output_dir}/reference"
  mkdir -p "${reference_dir}"
  reference_video="${reference_dir}/${camera}-nominal.mp4"
  reference_frame="${reference_dir}/${camera}-nominal.png"
  reference_results="${reference_dir}/nominal.json"
  reference_log="${reference_dir}/nominal.log"
  reference_observation="${reference_dir}/nominal-observation.npz"

  if ((force == 1)) || [[ ! -s "${reference_results}" \
      || ! -s "${reference_observation}" ]]; then
    echo "[batch] nominal layout: capturing ${camera} reference"
    _wait_for_memory
    ref_cmd=(
      "${repo_root}/scripts/run_roco.sh"
      --policy "${policy}"
      --max-steps "${max_steps}"
      --observation-snapshot "${reference_observation}"
      --results-json "${reference_results}"
    )
    if ((record_video == 1)); then
      ref_cmd+=(
        --record-video "${reference_video}"
        --record-video-camera "${camera}"
        --record-video-fps "${video_fps}"
      )
    fi
    "${ref_cmd[@]}" 2>&1 | tee "${reference_log}"
    if ((record_video == 1)) && [[ -s "${reference_video}" ]]; then
      ffmpeg -y -loglevel error \
        -sseof -5 -i "${reference_video}" \
        -vf reverse -frames:v 1 "${reference_frame}"
    fi
    _cleanup_isaac_leftovers
    if ((cooldown_sec > 0)); then
      sleep "${cooldown_sec}"
    fi
  else
    echo "[batch] nominal layout: complete; skipping"
  fi
  if [[ -s "${reference_observation}" ]]; then
    echo "[batch] nominal layout: building camera reference bundle"
    python3 "${repo_root}/scripts/build_camera_reference.py" \
      --camera-axes world \
      --observation "${reference_observation}" \
      --output "${repo_root}/task/policies/camera_reference" \
      || echo "[batch] WARN: camera reference build failed (capture pose missing?)"
  fi
fi

failed_seeds=()
for ((i = 0; i < seed_count; i++)); do
  seed=$((start_seed + i))
  seed_tag="$(printf '%03d' "${seed}")"
  video_path="${videos_dir}/seed-${seed_tag}.mp4"
  frame_path="${frames_dir}/seed-${seed_tag}-final.png"
  results_path="${results_dir}/seed-${seed_tag}.json"
  log_path="${logs_dir}/seed-${seed_tag}.log"
  observation_path="${observations_dir}/seed-${seed_tag}.npz"

  if ((force == 0)) && _seed_complete "${observation_path}" "${results_path}" "${frame_path}"; then
    echo "[batch] seed ${seed}: complete; skipping"
    continue
  fi

  attempt=0
  ok=0
  while ((attempt <= retries)); do
    if ((attempt > 0)); then
      echo "[batch] seed ${seed}: retry ${attempt}/${retries}"
      _cleanup_isaac_leftovers
      if ((cooldown_sec > 0)); then
        sleep "${cooldown_sec}"
      fi
    fi
    _wait_for_memory
    echo "[batch] seed ${seed}: running rollout (attempt $((attempt + 1)))"
    cmd=(
      "${repo_root}/scripts/run_roco.sh"
      --policy "${policy}"
      --random-seed "${seed}"
      --max-steps "${max_steps}"
      --observation-snapshot "${observation_path}"
      --results-json "${results_path}"
    )
    if ((record_video == 1)); then
      cmd+=(
        --record-video "${video_path}"
        --record-video-camera "${camera}"
        --record-video-fps "${video_fps}"
      )
    fi
    set +e
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    rc=${PIPESTATUS[0]}
    set -e
    if ((rc == 0)) && [[ -s "${observation_path}" && -s "${results_path}" ]]; then
      ok=1
      break
    fi
    echo "[batch] seed ${seed}: launch failed rc=${rc}" >&2
    attempt=$((attempt + 1))
  done

  _cleanup_isaac_leftovers

  if ((ok == 0)); then
    failed_seeds+=("${seed}")
    echo "[batch] seed ${seed}: FAILED after $((retries + 1)) attempts" >&2
    if ((continue_on_error == 0)); then
      exit 1
    fi
    if ((cooldown_sec > 0)); then
      sleep "${cooldown_sec}"
    fi
    continue
  fi

  if ((record_video == 1)); then
    if [[ ! -s "${video_path}" ]]; then
      echo "[batch] seed ${seed}: rollout produced no video" >&2
      failed_seeds+=("${seed}")
      if ((continue_on_error == 0)); then
        exit 1
      fi
      continue
    fi
    echo "[batch] seed ${seed}: extracting final frame"
    ffmpeg -y -loglevel error \
      -sseof -5 -i "${video_path}" \
      -vf reverse -frames:v 1 "${frame_path}"
  else
    echo "[batch] seed ${seed}: wrote observation+results (no video)"
  fi

  if ((cooldown_sec > 0)); then
    echo "[batch] cooldown ${cooldown_sec}s before next seed"
    sleep "${cooldown_sec}"
  fi
done

if ((record_video == 1)) && compgen -G "${frames_dir}/seed-*-final.png" >/dev/null; then
  contact_sheet="${output_dir}/contact-sheet.png"
  echo "[batch] building contact sheet"
  ffmpeg -y -loglevel error \
    -pattern_type glob -framerate 1 -i "${frames_dir}/seed-*-final.png" \
    -vf "scale=320:-1,tile=5x6:padding=4:margin=4" \
    -frames:v 1 "${contact_sheet}" || true
  echo "[batch] contact sheet: ${contact_sheet}"
fi

echo "[batch] done"
echo "[batch] RGB-D observations: ${observations_dir}"
echo "[batch] results JSON: ${results_dir}"
if ((include_reference == 1)); then
  echo "[batch] nominal RGB-D observation: ${reference_observation}"
fi
if ((${#failed_seeds[@]} > 0)); then
  echo "[batch] FAILED seeds: ${failed_seeds[*]}" >&2
  exit 1
fi
