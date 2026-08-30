#!/usr/bin/env bash
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
  --start-seed N   First seed (default: 0)
  --count N        Number of seeds (default: 10)
  --output-dir DIR Output root (default: artifacts/randomization-final-frames)
  --camera NAME    head, L_wrist, or R_wrist (default: head)
  --fps N          Recorded video FPS (default: 15)
  --max-steps N    Control steps before snapshot (default: 3)
  --policy PATH    Dotted policy class used during capture
  --no-reference   Do not capture the nominal, unrandomized reference frame
  --force          Rerun and overwrite completed seed outputs
  -h, --help       Show this help

Randomized runs are competition-faithful: the policy receives nominal config,
camera output is enabled, and sampled offsets are not printed. Exact offsets
remain in the post-run results JSON for offline estimator-error measurement.
Each capture also writes RGB, depth, intrinsics, and camera pose to NPZ.
The nominal capture is converted into task/policies/camera_reference for
CameraOffsetScriptedPolicy. Completed captures are skipped unless --force
is supplied.
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

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required to extract final frames" >&2
  exit 1
}

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

if ((include_reference == 1)); then
  reference_dir="${output_dir}/reference"
  mkdir -p "${reference_dir}"
  reference_video="${reference_dir}/${camera}-nominal.mp4"
  reference_frame="${reference_dir}/${camera}-nominal.png"
  reference_results="${reference_dir}/nominal.json"
  reference_log="${reference_dir}/nominal.log"
  reference_observation="${reference_dir}/nominal-observation.npz"

  if ((force == 1)) || [[ ! -s "${reference_frame}" \
      || ! -s "${reference_results}" || ! -s "${reference_observation}" ]]; then
    echo "[batch] nominal layout: capturing ${camera} reference"
    "${repo_root}/scripts/run_roco.sh" \
      --policy "${policy}" \
      --max-steps "${max_steps}" \
      --record-video "${reference_video}" \
      --record-video-camera "${camera}" \
      --record-video-fps "${video_fps}" \
      --observation-snapshot "${reference_observation}" \
      --results-json "${reference_results}" \
      2>&1 | tee "${reference_log}"
    ffmpeg -y -loglevel error \
      -sseof -5 -i "${reference_video}" \
      -vf reverse -frames:v 1 "${reference_frame}"
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

for ((i = 0; i < seed_count; i++)); do
  seed=$((start_seed + i))
  seed_tag="$(printf '%03d' "${seed}")"
  video_path="${videos_dir}/seed-${seed_tag}.mp4"
  frame_path="${frames_dir}/seed-${seed_tag}-final.png"
  results_path="${results_dir}/seed-${seed_tag}.json"
  log_path="${logs_dir}/seed-${seed_tag}.log"
  observation_path="${observations_dir}/seed-${seed_tag}.npz"

  if ((force == 0)) && [[ -s "${frame_path}" && -s "${results_path}" \
      && -s "${observation_path}" ]]; then
    echo "[batch] seed ${seed}: complete; skipping"
    continue
  fi

  if ((force == 1)) || [[ ! -s "${video_path}" || ! -s "${results_path}" \
      || ! -s "${observation_path}" ]]; then
    echo "[batch] seed ${seed}: running rollout"
    "${repo_root}/scripts/run_roco.sh" \
      --policy "${policy}" \
      --random-seed "${seed}" \
      --max-steps "${max_steps}" \
      --record-video "${video_path}" \
      --record-video-camera "${camera}" \
      --record-video-fps "${video_fps}" \
      --observation-snapshot "${observation_path}" \
      --results-json "${results_path}" \
      2>&1 | tee "${log_path}"
  else
    echo "[batch] seed ${seed}: reusing existing rollout video"
  fi

  if [[ ! -s "${video_path}" ]]; then
    echo "[batch] seed ${seed}: rollout produced no video" >&2
    exit 1
  fi

  echo "[batch] seed ${seed}: extracting final frame"
  # Read only the final five seconds, reverse that short segment, then keep
  # its first frame. This yields the last decodable frame without buffering
  # the entire rollout in ffmpeg's reverse filter.
  ffmpeg -y -loglevel error \
    -sseof -5 -i "${video_path}" \
    -vf reverse -frames:v 1 "${frame_path}"
done

contact_sheet="${output_dir}/contact-sheet.png"
echo "[batch] building contact sheet"
ffmpeg -y -loglevel error \
  -pattern_type glob -framerate 1 -i "${frames_dir}/seed-*-final.png" \
  -vf "scale=320:-1,tile=5x6:padding=4:margin=4" \
  -frames:v 1 "${contact_sheet}"

echo "[batch] done"
echo "[batch] final frames: ${frames_dir}"
echo "[batch] RGB-D observations: ${observations_dir}"
if ((include_reference == 1)); then
  echo "[batch] nominal reference: ${reference_frame}"
  echo "[batch] nominal RGB-D observation: ${reference_observation}"
fi
echo "[batch] contact sheet: ${contact_sheet}"
