#!/usr/bin/env python3
"""Launch one ALOHA-to-DexMate collection episode.

Does not import Isaac Sim or LeRobot. Starts the challenge harness under
the uv/Isaac environment and points it at the isolated LeRobot sidecar.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv" / "bin" / "python"
LEROBOT_PYTHON = Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "python"
MIN_FREE_RAM_GB = 12.0


def _mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024 * 1024)
    return 0.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _isaac_env(extra: dict) -> dict:
    env = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "DISPLAY": os.environ.get("DISPLAY", ""),
        "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": "/usr/local/bin:/usr/bin:/bin:" + str(ROOT / ".venv" / "bin"),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "TASK_ENABLE_CAMERA_OUTPUT": "1",
        "TASK_ENABLE_CAMERA_VIEWPORTS": os.environ.get("TASK_ENABLE_CAMERA_VIEWPORTS", "1"),
        "ROCO_PER_PART_TIMEOUT_STEPS": os.environ.get("ROCO_PER_PART_TIMEOUT_STEPS", "120000"),
        "ALOHA_LEADER_ENDPOINT": os.environ.get("ALOHA_LEADER_ENDPOINT", "127.0.0.1:19850"),
        "ALOHA_TELEOP_CONFIG": str(ROOT / "config" / "aloha_solo_to_vega_1u.yaml"),
        "LEROBOT_SERVER_PY": str(LEROBOT_PYTHON),
        "LEROBOT_SERVER_SCRIPT": str(ROOT / "tools" / "lerobot_recorder" / "server.py"),
        "ROCO_COMMIT": _git_commit(),
    }
    for key in (
        "XDG_RUNTIME_DIR",
        "DBUS_SESSION_BUS_ADDRESS",
        "WAYLAND_DISPLAY",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "ISAACSIM_HEADLESS",
        "ISAACSIM_ACTIVE_GPU",
        "ISAACSIM_PHYSICS_GPU",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    env.update(extra)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="local/roco_aloha_teleop")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-parts", type=int, default=0)
    parser.add_argument("--max-sim-seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Print the synthetic leader command; does not start it.")
    args = parser.parse_args()

    if not args.skip_preflight:
        mem = _mem_available_gb()
        if mem < MIN_FREE_RAM_GB:
            print(f"Need >= {MIN_FREE_RAM_GB} GB RAM free; have {mem:.1f} GB. "
                  "Close browsers/IDEs or pass --skip-preflight.", file=sys.stderr)
            return 2
    if not ISAAC_PYTHON.exists():
        print(f"Isaac venv missing: {ISAAC_PYTHON}. Run `uv sync` in {ROOT}.", file=sys.stderr)
        return 2
    if not LEROBOT_PYTHON.exists():
        print(f"LeRobot env missing: {LEROBOT_PYTHON}.", file=sys.stderr)
        return 2

    results = args.output_root / args.run_id / "results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "LEROBOT_REPO_ID": args.repo_id,
        "LEROBOT_OUTPUT_ROOT": str(args.output_root),
        "ALOHA_TELEOP_RUN_ID": args.run_id,
        "LEROBOT_SERVER_LOG": str(results.parent / "recorder.log"),
    }
    if args.headless:
        extra["ISAACSIM_HEADLESS"] = "1"

    cmd = [
        str(ISAAC_PYTHON),
        str(ROOT / "task" / "run_pick_place.py"),
        "--policy",
        "policies.aloha_teleop.AlohaTeleopPolicy",
        "--results-json",
        str(results),
    ]
    if args.max_parts:
        cmd += ["--max-parts", str(args.max_parts)]
    if args.max_sim_seconds:
        cmd += ["--max-sim-seconds", str(args.max_sim_seconds)]

    print("Leader (hardware):")
    print("  source /opt/ros/humble/setup.bash")
    print("  source /home/khw/interbotix_ws/install/setup.bash")
    print("  ros2 launch aloha_isaac_teleop leader_only.launch.py robot:=aloha_solo")
    if args.synthetic:
        print("Synthetic leader:")
        print(f"  {sys.executable} {ROOT / 'scripts' / 'synthetic_leader.py'}")
    print("Isaac command:", " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd=str(ROOT / "task"), env=_isaac_env(extra))
    print(f"harness exit={proc.returncode} results={results}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
