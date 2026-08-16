#!/usr/bin/env python3
"""Emit synthetic ALOHA leader samples so Isaac can be tested without hardware.

This process blocks until killed. Prefer:

    python3 scripts/collect_aloha_episode.py --synthetic ...

which starts this leader in the background and then launches Isaac.
"""
from __future__ import annotations

import argparse
import math
import time

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "task"))

from teleop.protocol import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    bind_server,
    make_leader_sample,
    send_message,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--amplitude", type=float, default=0.04)
    parser.add_argument("--clutch", action="store_true", default=True)
    args = parser.parse_args()

    server = bind_server(args.host, args.port)
    print(f"[synthetic_leader] listening on {args.host}:{args.port}", flush=True)
    conn = None
    seq = 0
    period = 1.0 / args.hz
    t0 = time.time()
    while True:
        if conn is None:
            try:
                conn, addr = server.accept()
                conn.setsockopt(__import__("socket").IPPROTO_TCP, __import__("socket").TCP_NODELAY, 1)
                print(f"[synthetic_leader] client {addr}", flush=True)
            except TimeoutError:
                continue
            except OSError:
                continue
        t = time.time() - t0
        pos = [
            args.amplitude * math.sin(0.4 * t),
            args.amplitude * math.sin(0.5 * t + 0.3),
            0.02 * math.sin(0.2 * t),
        ]
        sample = make_leader_sample(
            seq=seq,
            joints=[0.0, -0.6, 0.5, 0.0, 1.2, 0.0],
            ee_pos=pos,
            ee_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            gripper_norm=0.5 + 0.5 * math.sin(0.3 * t),
            clutch=args.clutch,
            deadman=True,
            cmd="start" if seq == 0 else "none",
        )
        try:
            send_message(conn, sample)
        except OSError:
            print("[synthetic_leader] client disconnected", flush=True)
            conn.close()
            conn = None
            continue
        seq += 1
        if seq % int(args.hz) == 0:
            print(f"[synthetic_leader] seq={seq} pos={pos}", flush=True)
        time.sleep(period)


if __name__ == "__main__":
    main()
