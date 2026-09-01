#!/usr/bin/env bash
# Chunked wrapper around generate_randomization_final_frames.sh.
#
# Each Isaac Kit relaunch leaks host RAM into swap. Running 100 seeds in one
# tight loop crashes. This script captures in small chunks with a long pause
# between chunks so the OS can reclaim memory.
#
# Example (resume after a partial run that reached seed 34):
#   scripts/capture_randomization_chunked.sh --start-seed 0 --count 100
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

start_seed=0
seed_count=100
chunk_size=5
chunk_pause_sec=90
cooldown_sec=45
retries=2
min_free_mem_mb=12000
extra_args=()

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
    --chunk-size)
      chunk_size="$2"
      shift 2
      ;;
    --chunk-pause-sec)
      chunk_pause_sec="$2"
      shift 2
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
    -h|--help)
      cat <<'EOF'
Capture fairness-randomization RGB-D snapshots in small Isaac chunks.

Usage:
  scripts/capture_randomization_chunked.sh [options]

Options:
  --start-seed N         First seed (default: 0)
  --count N              Total seeds (default: 100)
  --chunk-size N         Seeds per Isaac batch (default: 5)
  --chunk-pause-sec N    Pause between chunks (default: 90)
  --cooldown-sec N       Pause between seeds inside a chunk (default: 45)
  --retries N            Per-seed retries (default: 2)
  --min-free-mem-mb N    Min MemAvailable before launch (default: 12000)

Always passes --no-video --no-reference to the inner script (NPZ+JSON only;
nominal reference is assumed already built). Completed seeds are skipped.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

log_dir="${repo_root}/artifacts/randomization-final-frames/logs"
mkdir -p "${log_dir}"
master_log="${log_dir}/chunked-$(date +%Y%m%d-%H%M%S).log"

{
  echo "[chunked] start_seed=${start_seed} count=${seed_count} chunk_size=${chunk_size}"
  echo "[chunked] cooldown=${cooldown_sec}s chunk_pause=${chunk_pause_sec}s"
  end_seed=$((start_seed + seed_count))
  seed=${start_seed}
  while ((seed < end_seed)); do
    remaining=$((end_seed - seed))
    this_count=${chunk_size}
    if ((this_count > remaining)); then
      this_count=${remaining}
    fi
    echo "[chunked] === seeds ${seed}..$((seed + this_count - 1)) ==="
    set +e
    "${repo_root}/scripts/generate_randomization_final_frames.sh" \
      --start-seed "${seed}" \
      --count "${this_count}" \
      --no-reference \
      --no-video \
      --cooldown-sec "${cooldown_sec}" \
      --retries "${retries}" \
      --min-free-mem-mb "${min_free_mem_mb}"
    rc=$?
    set -e
    if ((rc != 0)); then
      echo "[chunked] WARN: chunk starting at ${seed} exited rc=${rc} (continuing)"
    fi
    seed=$((seed + this_count))
    if ((seed < end_seed)); then
      echo "[chunked] pausing ${chunk_pause_sec}s between chunks"
      # Reap leftovers and let swap breathe.
      pkill -f "task/run_pick_place.py" 2>/dev/null || true
      pkill -f "isaacsim" 2>/dev/null || true
      sleep "${chunk_pause_sec}"
    fi
  done
  n_obs=$(ls "${repo_root}/artifacts/randomization-final-frames/observations"/seed-*.npz 2>/dev/null | wc -l)
  n_json=$(ls "${repo_root}/artifacts/randomization-final-frames/results"/seed-*.json 2>/dev/null | wc -l)
  echo "[chunked] done observations=${n_obs} results=${n_json}"
} 2>&1 | tee -a "${master_log}"

echo "[chunked] log: ${master_log}"
