#!/usr/bin/env bash
set -euo pipefail

export TASK_ENABLE_CAMERA_OUTPUT="${TASK_ENABLE_CAMERA_OUTPUT:-1}"

exec uv run python task/run_pick_place.py \
  --policy policies.camera_offset_scripted.CameraOffsetScriptedPolicy \
  "$@"
