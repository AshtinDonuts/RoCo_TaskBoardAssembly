#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

archive_name="roco_iros2026_camera_offset_policy.zip"
output_path="${1:-$repo_root/dist/$archive_name}"
if [[ "$output_path" != /* ]]; then
  output_path="$repo_root/$output_path"
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "error: zip is required" >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip is required for archive verification" >&2
  exit 1
fi

staging_dir="$(mktemp -d)"
trap 'rm -rf -- "$staging_dir"' EXIT
bundle_name="RoCo_TaskBoardAssembly_submission"
bundle_root="$staging_dir/$bundle_name"
mkdir -p "$bundle_root" "$(dirname "$output_path")"

include_file() {
  case "$1" in
    .python-version|pyproject.toml|uv.lock|snap_attach.py|scene_base.usd|scene_init.usd|SUBMISSION.md|run_submission.sh)
      return 0
      ;;
    parts/*|robot/*|table/*|task/controllers/*|task/policies/camera_offset/*|task/policies/camera_reference/*)
      return 0
      ;;
    task/policy_api.py|task/param_config.py|task/part_init_poses.json|task/eval_randomization.py|task/run_pick_place.py)
      return 0
      ;;
    task/policies/__init__.py|task/policies/_joint_rate_limiter.py|task/policies/baseline_scripted.py|task/policies/camera_offset_scripted.py)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

copied=0
while IFS= read -r -d '' file; do
  if ! include_file "$file"; then
    continue
  fi
  if [[ ! -f "$file" ]]; then
    echo "error: selected submission file is missing: $file" >&2
    exit 1
  fi
  case "$file" in
    *.usd|*.usda|*.usdc|*.obj)
      if head -c 256 -- "$file" | grep -q 'git-lfs.github.com/spec'; then
        echo "error: $file is an unresolved Git LFS pointer" >&2
        exit 1
      fi
      ;;
  esac
  cp -a --parents -- "$file" "$bundle_root/"
  copied=$((copied + 1))
done < <(git ls-files --cached --others --exclude-standard -z)

required=(
  SUBMISSION.md
  run_submission.sh
  scene_init.usd
  task/policy_api.py
  task/run_pick_place.py
  task/param_config.py
  task/policies/camera_offset_scripted.py
  task/policies/baseline_scripted.py
  task/policies/_joint_rate_limiter.py
  task/policies/camera_offset/estimator.py
  task/policies/camera_reference/manifest.json
  task/policies/camera_reference/head_rgb.npy
  task/controllers/ee_pose_controller.py
  task/controllers/vega_1u_setup.py
)
for file in "${required[@]}"; do
  if [[ ! -f "$bundle_root/$file" ]]; then
    echo "error: required runtime file was not packaged: $file" >&2
    exit 1
  fi
done

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "policy=policies.camera_offset_scripted.CameraOffsetScriptedPolicy"
  echo "camera_output=TASK_ENABLE_CAMERA_OUTPUT=1"
} > "$bundle_root/SUBMISSION_COMMIT.txt"

(
  cd "$bundle_root"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > MANIFEST.sha256
)

uncompressed_bytes="$(du -sb "$bundle_root" | awk '{print $1}')"
if (( uncompressed_bytes >= 1000000000 )); then
  echo "error: uncompressed bundle is ${uncompressed_bytes} bytes (limit: 1000000000)" >&2
  exit 1
fi

rm -f -- "$output_path"
(
  cd "$staging_dir"
  zip -q -9 -r "$output_path" "$bundle_name"
)

unzip -tq "$output_path" >/dev/null
archive_bytes="$(stat -c %s "$output_path")"
if (( archive_bytes >= 1000000000 )); then
  echo "error: ZIP is ${archive_bytes} bytes (limit: 1000000000)" >&2
  exit 1
fi

echo "created: $output_path"
echo "files: $copied source files + submission metadata"
echo "uncompressed_bytes: $uncompressed_bytes"
echo "zip_bytes: $archive_bytes"
sha256sum "$output_path"
