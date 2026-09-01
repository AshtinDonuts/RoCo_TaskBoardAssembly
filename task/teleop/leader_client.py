"""Thread-safe TCP client for ALOHA leader samples."""
from __future__ import annotations

import socket
import threading
import time
from collections import deque
from typing import Deque, Optional

from .protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ProtocolError,
    connect_with_retry,
    recv_message,
    validate_leader_sample,
)


class LeaderClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout_s: float = 120.0,
        cmd_queue_max: int = 32,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self._lock = threading.Lock()
        self._latest = None
        # Full samples for edge cmds so resume/pause keep the leader pose
        # that accompanied the keypress, not a later pose-only frame.
        self._cmd_queue: Deque[dict] = deque(maxlen=max(1, int(cmd_queue_max)))
        self._last_recv_s = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._rate_n = 0
        self._rate_t0 = time.time()
        self._hz = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="leader-client", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def hz(self) -> float:
        return self._hz

    @property
    def error(self) -> Optional[str]:
        return self._error

    def age_s(self) -> float:
        with self._lock:
            if self._last_recv_s <= 0.0:
                return float("inf")
            return time.time() - self._last_recv_s

    def latest(self):
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def pop_cmd(self) -> str:
        """Drain one queued edge command string (compat). Prefer pop_cmd_event()."""
        event = self.pop_cmd_event()
        if event is None:
            return "none"
        return str(event.get("cmd", "none"))

    def pop_cmd_event(self) -> Optional[dict]:
        """Pop one queued command-event sample (cmd + leader pose at that edge)."""
        with self._lock:
            if not self._cmd_queue:
                return None
            return dict(self._cmd_queue.popleft())

    def _ingest_sample_locked(self, sample: dict, now: float) -> None:
        """Store latest pose sample and queue any non-none edge command event."""
        cmd = sample.get("cmd", "none")
        stored = dict(sample)
        if cmd not in (None, "", "none"):
            event = dict(sample)
            event["cmd"] = str(cmd)
            self._cmd_queue.append(event)
        stored["cmd"] = "none"
        self._latest = stored
        self._last_recv_s = now

    def _loop(self) -> None:
        sock: Optional[socket.socket] = None
        while not self._stop.is_set():
            try:
                if sock is None:
                    sock = connect_with_retry(
                        self.host, self.port, timeout_s=self.connect_timeout_s
                    )
                    self._connected = True
                    self._error = None
                sock.settimeout(0.25)
                try:
                    msg = recv_message(sock)
                except socket.timeout:
                    continue
                sample = validate_leader_sample(msg)
                now = time.time()
                with self._lock:
                    self._ingest_sample_locked(sample, now)
                self._rate_n += 1
                elapsed = now - self._rate_t0
                if elapsed >= 1.0:
                    self._hz = self._rate_n / elapsed
                    self._rate_n = 0
                    self._rate_t0 = now
            except (OSError, ConnectionError, ProtocolError) as exc:
                self._connected = False
                self._error = str(exc)
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    sock = None
                time.sleep(0.2)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._connected = False
