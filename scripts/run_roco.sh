#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

cd "${repo_root}"
# shellcheck source=scripts/roco_isaac_env.sh
source "${script_dir}/roco_isaac_env.sh"

exec uv run python task/run_pick_place.py "$@"
