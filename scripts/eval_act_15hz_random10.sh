#!/usr/bin/env bash
# Evaluate each trained 15Hz ACT subtask policy on 10 randomized boards.
# Each rollout is limited to 100 task steps and uses ACT temporal aggregation.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tools_dir="${repo_root}/artifacts/act_15hz_grid/eval_tools"
lerobot_root="${LEROBOT_ROOT:-$(cd -- "${repo_root}/.." && pwd)/lerobot_roco_pi05}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/roco-act-uv-cache}"
act_gpu="${ACT_CUDA_VISIBLE_DEVICES:-1}"
isaac_gpu="${ISAACSIM_ACTIVE_GPU:-0}"
physics_gpu="${ISAACSIM_PHYSICS_GPU:-0}"
coeff="${ACT_TEMPORAL_ENSEMBLE_COEFF:-0.01}"
max_steps="${ACT_EVAL_MAX_STEPS:-100}"
output_suffix="evals_ckpt8000_random10_max${max_steps}_tempagg"
gcc_lib="/tmp/roco-gcc12-runtime/lib/x86_64-linux-gnu:/tmp/roco-gcc12-runtime/usr/lib/x86_64-linux-gnu"

parts=(usb_a rod_16mm battery_size5 gear_60teeth)
seeds=(0 1 2 3 4 5 6 7 8 9)

if [[ ! -x "${lerobot_root}/.venv/bin/python" ]]; then
  echo "missing lerobot venv: ${lerobot_root}/.venv/bin/python" >&2
  exit 2
fi

fail=0
for part in "${parts[@]}"; do
  run_root="${repo_root}/artifacts/act_15hz_grid/roco_act_15hz_${part}_chunk10_b32"
  ckpt="${run_root}/checkpoints/008000/pretrained_model"
  output_root="${run_root}/${output_suffix}"
  queue_log="${output_root}/queue.log"

  if [[ ! -f "${ckpt}/model.safetensors" ]]; then
    echo "missing checkpoint: ${ckpt}" >&2
    fail=1
    continue
  fi
  mkdir -p "${output_root}"
  printf '%s START part=%s seeds=%s (max-steps=%s, temporal_agg=%s)\n' \
    "$(date --iso-8601=seconds)" "${part}" "${seeds[*]}" "${max_steps}" "${coeff}" \
    | tee -a "${queue_log}"

  for seed in "${seeds[@]}"; do
    out_dir="${output_root}/seed_${seed}"
    video="${out_dir}/${part}_head.mp4"
    results="${out_dir}/${part}_results.json"
    log="${out_dir}/eval.log"
    server_log="${out_dir}/server.log"
    mkdir -p "${out_dir}"

    if [[ -s "${video}" && -s "${results}" ]]; then
      printf '%s SKIP part=%s seed=%s (results and video already exist)\n' \
        "$(date --iso-8601=seconds)" "${part}" "${seed}" | tee -a "${queue_log}"
      continue
    fi

    printf '%s START part=%s seed=%s\n' \
      "$(date --iso-8601=seconds)" "${part}" "${seed}" | tee -a "${queue_log}"
    cd "${repo_root}"
    set +e
    LD_LIBRARY_PATH="${gcc_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${tools_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    ACT_CKPT="${ckpt}" \
    ACT_SERVER_PY="${lerobot_root}/.venv/bin/python" \
    ACT_SERVER="${tools_dir}/roco_act_server.py" \
    ACT_SERVER_PYTHONPATH="${lerobot_root}/src" \
    ACT_SERVER_LOG="${server_log}" \
    ACT_CUDA_VISIBLE_DEVICES="${act_gpu}" \
    ACT_TEMPORAL_ENSEMBLE_COEFF="${coeff}" \
    ROCO_PART_ORDER="${part}" \
    TASK_ENABLE_CAMERA_VIEWPORTS=0 \
    ISAACSIM_HEADLESS=1 \
    ISAACSIM_ACTIVE_GPU="${isaac_gpu}" \
    ISAACSIM_PHYSICS_GPU="${physics_gpu}" \
    ./scripts/run_roco.sh \
      --policy act_lerobot.ACTLeRobotPolicy \
      --record-video "${video}" \
      --record-video-camera head \
      --record-video-fps 15 \
      --results-json "${results}" \
      --max-parts 1 \
      --max-steps "${max_steps}" \
      --random-seed "${seed}" \
      > "${log}" 2>&1
    status=$?
    set -e
    printf '%s END part=%s seed=%s exit=%s results=%s\n' \
      "$(date --iso-8601=seconds)" "${part}" "${seed}" "${status}" "${results}" \
      | tee -a "${queue_log}"
    if [[ "${status}" -ne 0 ]]; then
      fail=1
    fi
  done

  printf '%s DONE part=%s fail=%s\n' "$(date --iso-8601=seconds)" "${part}" "${fail}" \
    | tee -a "${queue_log}"
done

exit "${fail}"
