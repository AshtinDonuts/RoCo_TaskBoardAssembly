#!/usr/bin/env python3
"""Launch one ALOHA-to-DexMate collection episode.

Does not import Isaac Sim or LeRobot. Starts the challenge harness under
the uv/Isaac environment and points it at the isolated LeRobot sidecar.

``--synthetic`` and ``--keyboard`` start a virtual leader in this process
so you do not need a second blocking terminal before Isaac launches.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task"))

from teleop.export_config import (  # noqa: E402
    DEFAULT_EXPORT_CONFIG,
    load_export_config,
)
from teleop.protocol import parse_endpoint  # noqa: E402

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
    env = os.environ.copy()
    env.update({
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "PYTHONUNBUFFERED": "1",
        "TASK_ENABLE_CAMERA_OUTPUT": "1",
        "TASK_ENABLE_CAMERA_VIEWPORTS": os.environ.get("TASK_ENABLE_CAMERA_VIEWPORTS", "1"),
        "ROCO_PER_PART_TIMEOUT_STEPS": os.environ.get("ROCO_PER_PART_TIMEOUT_STEPS", "120000"),
        "LEROBOT_SERVER_PY": str(LEROBOT_PYTHON),
        "LEROBOT_SERVER_SCRIPT": str(ROOT / "tools" / "lerobot_recorder" / "server.py"),
        "ROCO_COMMIT": _git_commit(),
    })
    # Only forward when set so an empty default does not clear all tiles.
    if "TASK_CAMERA_VIEWPORTS" in os.environ:
        env["TASK_CAMERA_VIEWPORTS"] = os.environ["TASK_CAMERA_VIEWPORTS"]
    env["PATH"] = str(ROOT / ".venv" / "bin") + os.pathsep + env.get("PATH", "")
    env.update(extra)
    return env


def wait_for_listen(host: str, port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(0.25)
            sock.connect((host, port))
            sock.close()
            return
        except OSError as exc:
            last_err = exc
            try:
                sock.close()
            except OSError:
                pass
            time.sleep(0.05)
    raise RuntimeError(f"leader not listening on {host}:{port}: {last_err}")


def _start_synthetic(host: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts" / "synthetic_leader.py"),
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
    )
    print(
        f"[collect] started synthetic leader pid={proc.pid} on {host}:{port}",
        flush=True,
    )
    return proc


def _terminate_leaders(procs: List[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--export-config",
        type=Path,
        default=Path(os.environ.get("ALOHA_EXPORT_CONFIG", str(DEFAULT_EXPORT_CONFIG))),
    )
    pre_args, _ = pre.parse_known_args()
    try:
        export_cfg = load_export_config(pre_args.export_config)
    except Exception as exc:
        print(f"Failed to load export config {pre_args.export_config}: {exc}", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-config",
        type=Path,
        default=pre_args.export_config,
        help="Teleop/data-export JSON (fps, image size, session defaults, control.arms).",
    )
    parser.add_argument("--repo-id", default=export_cfg.export.dataset.repo_id)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=export_cfg.paths.output_root,
    )
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-parts", type=int, default=0)
    parser.add_argument("--max-sim-seconds", type=float, default=0.0)
    parser.add_argument(
        "--episode-time-s",
        type=float,
        default=float(
            os.environ.get("ROCO_EPISODE_TIME_S", str(export_cfg.session.episode_time_s))
        ),
        help="Wall-clock recording duration per episode. Right arrow saves earlier.",
    )
    parser.add_argument(
        "--warmup-time-s",
        type=float,
        default=float(
            os.environ.get("ROCO_WARMUP_TIME_S", str(export_cfg.session.warmup_time_s))
        ),
        help="Wall-clock warmup before each episode. No frames during warmup.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=int(
            os.environ.get("ROCO_NUM_EPISODES", str(export_cfg.session.num_episodes))
        ),
        help="Episodes to save in this session.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--camera-viewports",
        default=None,
        help=(
            "Kit camera viewport tiles to open (comma-separated). "
            "Choices: head, l_wrist, r_wrist (or all / none). "
            "Passed through to run_pick_place.py."
        ),
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Start sine-wave virtual leader(s) in the background, then launch Isaac.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Enable in-Isaac keyboard Cartesian teleop (no ALOHA leader TCP).",
    )
    args = parser.parse_args()
    if args.synthetic and args.keyboard:
        print("Use only one of --synthetic or --keyboard.", file=sys.stderr)
        return 2
    if args.export_config.resolve() != export_cfg.source_path:
        try:
            export_cfg = load_export_config(args.export_config)
        except Exception as exc:
            print(f"Failed to load export config {args.export_config}: {exc}", file=sys.stderr)
            return 2

    mem = _mem_available_gb()
    if not args.skip_preflight and mem < MIN_FREE_RAM_GB:
        msg = (
            f"Need >= {MIN_FREE_RAM_GB} GB RAM free; have {mem:.1f} GB. "
            "Close browsers/IDEs or pass --skip-preflight."
        )
        if args.synthetic or args.keyboard:
            print(msg + " Continuing because this is a virtual-leader run.", flush=True)
        else:
            print(msg, file=sys.stderr)
            return 2
    if not ISAAC_PYTHON.exists():
        print(f"Isaac venv missing: {ISAAC_PYTHON}. Run `uv sync` in {ROOT}.", file=sys.stderr)
        return 2
    if not LEROBOT_PYTHON.exists():
        print(f"LeRobot env missing: {LEROBOT_PYTHON}.", file=sys.stderr)
        return 2

    right_ep = export_cfg.leader_endpoint_for("right")
    left_ep = export_cfg.leader_endpoint_for("left")
    right_host, right_port = parse_endpoint(right_ep)
    left_host, left_port = parse_endpoint(left_ep)

    results = args.output_root / args.run_id / "results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    extra = {
        "ALOHA_EXPORT_CONFIG": str(export_cfg.source_path),
        "ALOHA_TELEOP_CONFIG": str(export_cfg.paths.teleop_yaml),
        "ALOHA_LEADER_ENDPOINT": right_ep,
        "ALOHA_LEADER_ENDPOINT_RIGHT": right_ep,
        "ALOHA_LEADER_ENDPOINT_LEFT": left_ep,
        "LEROBOT_REPO_ID": args.repo_id,
        "LEROBOT_OUTPUT_ROOT": str(args.output_root),
        "ALOHA_TELEOP_RUN_ID": args.run_id,
        "LEROBOT_SERVER_LOG": str(results.parent / "recorder.log"),
        "LEROBOT_RESULTS_JSON": str(results),
        "ROCO_EPISODE_TIME_S": str(args.episode_time_s),
        "ROCO_WARMUP_TIME_S": str(args.warmup_time_s),
        "ROCO_NUM_EPISODES": str(args.num_episodes),
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
    if args.camera_viewports:
        cmd += ["--camera-viewports", args.camera_viewports]

    stop = threading.Event()
    leader_procs: List[subprocess.Popen] = []
    if args.keyboard:
        extra["ALOHA_KEYBOARD_TELEOP"] = "1"
        print(
            "[collect] keyboard teleop runs inside Isaac (carb.input). "
            "Focus the Isaac viewport and hold i/k/j/l/t/g after launch. "
            f"control.arms={export_cfg.control_arms} (kbd drives right).",
            flush=True,
        )
    elif args.synthetic:
        if export_cfg.control_arms == "dual":
            leader_procs.append(_start_synthetic(left_host, left_port))
            leader_procs.append(_start_synthetic(right_host, right_port))
        else:
            leader_procs.append(_start_synthetic(right_host, right_port))
    else:
        print("Leader (hardware) must already be running:")
        print("  source /opt/ros/humble/setup.bash")
        print("  source /home/khw/interbotix_ws/install/setup.bash")
        if export_cfg.control_arms == "dual":
            print(
                f"  # two leader bridges on {left_ep} (left) and {right_ep} (right)"
            )
            print("  ros2 launch aloha_isaac_teleop leader_only.launch.py robot:=aloha_solo")
            print("  # plus a second bridge process bound to the other port")
        else:
            print("  ros2 launch aloha_isaac_teleop leader_only.launch.py robot:=aloha_solo")
            print(f"  # endpoint {right_ep} → DexMate right arm")

    if args.synthetic:
        try:
            wait_for_listen(right_host, right_port, timeout_s=5.0)
            if export_cfg.control_arms == "dual":
                wait_for_listen(left_host, left_port, timeout_s=5.0)
            time.sleep(0.2)
        except RuntimeError as exc:
            print(f"[collect] {exc}", file=sys.stderr)
            stop.set()
            _terminate_leaders(leader_procs)
            return 2

    print("Isaac command:", " ".join(cmd), flush=True)
    print(
        f"Export config: {export_cfg.source_path} "
        f"control.arms={export_cfg.control_arms} "
        f"fps={export_cfg.fps:g} clock={export_cfg.export.playback_clock} "
        f"image={export_cfg.img_w}x{export_cfg.img_h}",
        flush=True,
    )
    print(
        f"Recording: episode_time={args.episode_time_s:g}s "
        f"warmup={args.warmup_time_s:g}s num_episodes={args.num_episodes} "
        "(Right=save  Left=rerecord  Esc=stop)",
        flush=True,
    )
    print("[collect] launching Isaac Sim...", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT / "task"),
        env=_isaac_env(extra),
    )

    def _shutdown(*_args):
        stop.set()
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    try:
        code = proc.wait()
    finally:
        stop.set()
        _terminate_leaders(leader_procs)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            code = proc.returncode if proc.returncode is not None else 1
    print(f"harness exit={code} results={results}")
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
