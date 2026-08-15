#!/usr/bin/env python3
"""Host preflight for ALOHA Solo -> Isaac Sim Task Board collection."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

MIN_FREE_RAM_GB = 12.0
MIN_FREE_DISK_GB = 40.0
REPORT_PATH = Path.home() / "RoCo_TaskBoardAssembly" / "artifacts" / "preflight.json"


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"ERROR: {exc}"


def _mem_available_gb():
    info = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        info[key] = value.strip()
    kb = float(info.get("MemAvailable", "0 kB").split()[0])
    return kb / (1024 * 1024)


def _disk_free_gb(path: Path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def main() -> int:
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uname": _run(["uname", "-a"]).strip(),
        "os_release": Path("/etc/os-release").read_text(encoding="utf-8"),
        "python": _run(["python3", "--version"]).strip(),
        "ros": os.environ.get("ROS_DISTRO"),
        "display": os.environ.get("DISPLAY"),
        "nvidia_smi": _run(["nvidia-smi"]),
        "mem_available_gb": round(_mem_available_gb(), 2),
        "disk_free_gb": round(_disk_free_gb(Path.home()), 2),
        "uv": shutil.which("uv") or str(Path.home() / ".local/bin/uv"),
        "git_lfs": shutil.which("git-lfs"),
        "ffmpeg": shutil.which("ffmpeg"),
        "conda": shutil.which("conda"),
        "isaac_venv": str(Path.home() / "RoCo_TaskBoardAssembly" / ".venv"),
        "isaac_venv_exists": (Path.home() / "RoCo_TaskBoardAssembly" / ".venv").exists(),
        "lerobot_env": str(Path.home() / "miniconda3" / "envs" / "lerobot"),
        "lerobot_env_exists": (Path.home() / "miniconda3" / "envs" / "lerobot").exists(),
        "driver_strategy": "smoke-test 595.84 first; fall back to 580 if Kit/Vulkan crashes",
        "notes": [
            "Isaac Sim 5.1 minimum is 32 GB RAM; keep at least 12 GB free before launch.",
            "Do not run LeRobot training and Isaac Sim at the same time on 16 GB VRAM.",
            "Physical follower must stay unlaunched (launch_followers:=false).",
        ],
    }
    ok = True
    reasons = []
    if report["mem_available_gb"] < MIN_FREE_RAM_GB:
        ok = False
        reasons.append(f"RAM available {report['mem_available_gb']} GB < {MIN_FREE_RAM_GB}")
    if report["disk_free_gb"] < MIN_FREE_DISK_GB:
        ok = False
        reasons.append(f"disk free {report['disk_free_gb']} GB < {MIN_FREE_DISK_GB}")
    if "ERROR" in report["nvidia_smi"] or "failed" in report["nvidia_smi"].lower():
        ok = False
        reasons.append("nvidia-smi failed")
    report["ok"] = ok
    report["blockers"] = reasons
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ok", "blockers", "mem_available_gb", "disk_free_gb", "driver_strategy")}, indent=2))
    print(f"wrote {REPORT_PATH}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
