#!/usr/bin/env python3
"""LeRobot v3 recorder sidecar. Runs in the Python 3.12 conda env.

Length-prefixed pickle protocol on stdin/stdout. Never inherit Isaac's
PYTHONPATH. All numpy arrays from Isaac are converted to lists/bytes first.
"""
from __future__ import annotations

import pickle
import struct
import sys
import traceback
from typing import Any, Dict, Optional

from writer import EpisodeWriter

HEADER = struct.Struct(">I")

_in = sys.stdin.buffer
_out = sys.stdout.buffer


def _read() -> Optional[Dict[str, Any]]:
    header = _in.read(HEADER.size)
    if len(header) < HEADER.size:
        return None
    (n,) = HEADER.unpack(header)
    buf = b""
    while len(buf) < n:
        chunk = _in.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return pickle.loads(buf)


def _write(obj: Dict[str, Any]) -> None:
    payload = pickle.dumps(obj, protocol=4)
    _out.write(HEADER.pack(len(payload)) + payload)
    _out.flush()


def main() -> None:
    writer = EpisodeWriter()
    sys.stderr.write("[recorder] ready\n")
    sys.stderr.flush()
    while True:
        msg = _read()
        if msg is None:
            if not writer._finalized:
                try:
                    writer.finalize_session({"reason": "eof"})
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            break
        cmd = msg.get("cmd")
        try:
            if cmd == "init":
                _write(writer.init(msg))
            elif cmd == "begin_episode":
                _write(writer.begin(msg))
            elif cmd == "frame":
                _write(writer.add_frame(msg))
            elif cmd == "save_episode":
                _write(writer.save(msg))
            elif cmd == "discard_episode":
                _write(writer.discard(msg))
            elif cmd == "finalize_session":
                _write(writer.finalize_session(msg))
            elif cmd == "end_episode":
                _write(writer.end(msg))
            elif cmd == "shutdown":
                if not writer._finalized:
                    _write(writer.finalize_session({"reason": "shutdown"}))
                else:
                    _write({"ok": True})
                break
            else:
                _write({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _write({"ok": False, "error": str(exc)})


if __name__ == "__main__":
    main()
