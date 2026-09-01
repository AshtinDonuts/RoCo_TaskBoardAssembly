#!/usr/bin/env python3
"""Keyboard teleop is in-process now.

Use:

    python3 scripts/collect_aloha_episode.py --keyboard ...

Keys are read inside Isaac (carb.input) so the viewport can keep focus.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "keyboard_leader.py is obsolete.\n"
        "Run:\n"
        "  python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py "
        "--keyboard --max-parts 1 --episode-time-s 20 --warmup-time-s 2\n"
        "Then focus the Isaac window and hold i/k/j/l/t/g.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
