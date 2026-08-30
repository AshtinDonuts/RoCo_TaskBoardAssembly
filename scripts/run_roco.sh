#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cache_root="$(cd -- "${repo_root}/.." && pwd)"

cd "${repo_root}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${cache_root}/.uv-cache}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

if [[ -z "${DISPLAY:-}" ]]; then
  export ISAACSIM_HEADLESS="${ISAACSIM_HEADLESS:-1}"
fi

# Headless + forwarded DISPLAY (e.g. SSH -X) makes Kit probe X11/Xrandr and
# can crash during startup. Prefer a true headless session.
if [[ "${ISAACSIM_HEADLESS:-}" == "1" ]]; then
  unset DISPLAY
fi

if [[ -z "${VK_ICD_FILENAMES:-}" && -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

# Isaac Sim 5.x needs GLIBCXX_3.4.29+. Ubuntu 20.04's system libstdc++ stops
# at 3.4.28. Prefer LD_PRELOAD of a newer libstdc++ only — putting a full
# conda lib dir on LD_LIBRARY_PATH breaks system ffmpeg / X11.
_has_glibcxx_3429() {
  local lib="$1"
  [[ -f "$lib" ]] && grep -aq 'GLIBCXX_3.4.29' "$lib"
}

if ! _has_glibcxx_3429 /lib/x86_64-linux-gnu/libstdc++.so.6 \
  && ! _has_glibcxx_3429 /usr/lib/x86_64-linux-gnu/libstdc++.so.6; then
  _libstdcxx=""
  for cand in \
    "${ROCO_LIBSTDCXX:-}" \
    "${CONDA_PREFIX:+${CONDA_PREFIX}/lib/libstdc++.so.6}" \
    "${HOME}/miniconda3/lib/libstdc++.so.6" \
    "${HOME}/anaconda3/lib/libstdc++.so.6" \
    "${HOME}/mambaforge/lib/libstdc++.so.6" \
    "${HOME}/miniforge3/lib/libstdc++.so.6"
  do
    if [[ -n "$cand" ]] && _has_glibcxx_3429 "$cand"; then
      _libstdcxx="$cand"
      break
    fi
  done
  if [[ -z "$_libstdcxx" ]]; then
    echo "error: system libstdc++ lacks GLIBCXX_3.4.29 (needed by Isaac Sim)." >&2
    echo "Install a newer libstdc++ (e.g. conda) and set ROCO_LIBSTDCXX=/path/to/libstdc++.so.6" >&2
    exit 1
  fi
  # Keep a single libstdc++ preload; preserve any other preloads.
  _preload_rest=""
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    IFS=':' read -ra _pre_parts <<< "${LD_PRELOAD}"
    for _p in "${_pre_parts[@]}"; do
      [[ -z "$_p" || "$_p" == "$_libstdcxx" || "$_p" == *"/libstdc++.so"* ]] && continue
      _preload_rest="${_preload_rest:+${_preload_rest}:}${_p}"
    done
  fi
  export LD_PRELOAD="${_libstdcxx}${_preload_rest:+:${_preload_rest}}"
fi

# Drop conda lib dirs from LD_LIBRARY_PATH if present (common shell pollution).
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  _filtered=""
  IFS=':' read -ra _parts <<< "${LD_LIBRARY_PATH}"
  for _p in "${_parts[@]}"; do
    [[ -z "$_p" ]] && continue
    case "$_p" in
      *miniconda*|*anaconda*|*mambaforge*|*miniforge*|*conda/envs*|*conda/pkgs*) continue ;;
      *) _filtered="${_filtered:+${_filtered}:}${_p}" ;;
    esac
  done
  if [[ -n "$_filtered" ]]; then
    export LD_LIBRARY_PATH="${_filtered}"
  else
    unset LD_LIBRARY_PATH
  fi
fi

exec uv run python task/run_pick_place.py "$@"
