#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

start_seed=0
seed_count=10
output_dir="${repo_root}/artifacts/randomization-final-frames"
camera="head"
video_fps=15
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
    --force)
      force=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Generate seeded fairness-randomization rollouts and extract their final frames.

Usage:
  scripts/generate_randomization_final_frames.sh [options]

Options:
  --start-seed N   First seed (default: 0)
  --count N        Number of seeds (default: )
  --output-dir DIR Output root (default: artifacts/randomization-final-frames)
  --camera NAME    head, L_wrist, or R_wrist (default: head)
  --fps N          Recorded video FPS (default: 15)
  --force          Rerun and overwrite completed seed outputs
  -h, --help       Show this help

Completed seeds are skipped unless --force is supplied. Each seed produces an
MP4, results JSON, log, and final-frame PNG. A 5-column contact sheet is built
after all requested seeds complete.
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
mkdir -p "${videos_dir}" "${frames_dir}" "${results_dir}" "${logs_dir}"

cd "${repo_root}"
export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"

for ((i = 0; i < seed_count; i++)); do
  seed=$((start_seed + i))
  seed_tag="$(printf '%03d' "${seed}")"
  video_path="${videos_dir}/seed-${seed_tag}.mp4"
  frame_path="${frames_dir}/seed-${seed_tag}-final.png"
  results_path="${results_dir}/seed-${seed_tag}.json"
  log_path="${logs_dir}/seed-${seed_tag}.log"

  if ((force == 0)) && [[ -s "${frame_path}" && -s "${results_path}" ]]; then
    echo "[batch] seed ${seed}: complete; skipping"
    continue
  fi

  if ((force == 1)) || [[ ! -s "${video_path}" || ! -s "${results_path}" ]]; then
    echo "[batch] seed ${seed}: running rollout"
    "${repo_root}/scripts/run_roco.sh" \
      --random-seed "${seed}" \
      --max-steps 3 \
      --record-video "${video_path}" \
      --record-video-camera "${camera}" \
      --record-video-fps "${video_fps}" \
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
echo "[batch] contact sheet: ${contact_sheet}"
