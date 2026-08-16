"""Length-prefixed JSON IPC for ALOHA leader samples and operator commands.

The same module is duplicated in the ROS package
``aloha_isaac_teleop.protocol`` so Isaac Sim (Python 3.11) and ROS 2 Humble
(Python 3.10) never share an interpreter. Keep the two files byte-identical
except for this docstring's path.

Wire format: big-endian uint32 payload length followed by UTF-8 JSON.
No numpy objects are ever serialized.
"""
from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 19850
HEADER = struct.Struct(">I")
MAX_PAYLOAD_BYTES = 1_000_000

COMMANDS = (
    "none",
    "start",
    "pause",
    "resume",
    "recenter",
    "reset",
    "abort",
    "part_done",
    "estop",
    "save_episode",
    "rerecord_episode",
    "stop_recording",
)

# Local to the leader bridge; never sent on the wire.
LOCAL_COMMANDS = ("clutch_toggle",)

CHAR_TO_CMD = {
    " ": "clutch_toggle",
    "r": "recenter",
    "p": "pause",
    "u": "resume",
    "n": "part_done",
    "x": "abort",
    "e": "estop",
    "s": "start",
}

LINE_ALIASES = {
    "save": "save_episode",
    "right": "save_episode",
    "rerecord": "rerecord_episode",
    "left": "rerecord_episode",
    "stop": "stop_recording",
    "esc": "stop_recording",
}

_ARROW_CMD = {
    ord("C"): "save_episode",
    ord("D"): "rerecord_episode",
}

REQUIRED_SAMPLE_KEYS = (
    "type",
    "version",
    "timestamp_ns",
    "seq",
    "joints",
    "ee_pos",
    "ee_quat_wxyz",
    "gripper_norm",
    "clutch",
    "deadman",
    "cmd",
)


def map_operator_token(token: str) -> Optional[str]:
    """Map a typed line or single character to a command name."""
    if token is None:
        return None
    if token[:1] == " " and token.strip() == "":
        return "clutch_toggle"
    raw = token.strip()
    if not raw:
        return None
    low = raw.lower()
    if low in LINE_ALIASES:
        return LINE_ALIASES[low]
    if low in COMMANDS or low in LOCAL_COMMANDS:
        return low
    if len(raw) == 1:
        return CHAR_TO_CMD.get(raw, CHAR_TO_CMD.get(raw.lower()))
    return None


class KeyDecoder:
    """Decode operator keys from raw TTY bytes (cbreak / non-canonical).

    LeRobot-compatible recording keys:
      right arrow -> save_episode
      left arrow  -> rerecord_episode
      Esc         -> stop_recording
    """

    def __init__(self, esc_timeout_s: float = 0.05, extra_chars: Optional[Dict[str, str]] = None) -> None:
        self.esc_timeout_s = float(esc_timeout_s)
        self._state = "idle"
        self._esc_t0 = 0.0
        self._extra_chars = dict(extra_chars or {})

    def feed(self, data: bytes, now: Optional[float] = None) -> List[str]:
        now = time.monotonic() if now is None else now
        out: List[str] = []
        for byte in data:
            out.extend(self._feed_byte(byte, now))
        return out

    def poll_timeout(self, now: Optional[float] = None) -> List[str]:
        now = time.monotonic() if now is None else now
        if self._state == "esc" and (now - self._esc_t0) >= self.esc_timeout_s:
            self._state = "idle"
            return ["stop_recording"]
        return []

    def _feed_byte(self, byte: int, now: float) -> List[str]:
        if self._state == "idle":
            if byte == 0x1B:
                self._state = "esc"
                self._esc_t0 = now
                return []
            if byte in (0x00, 0x0A, 0x0D):
                return []
            try:
                char = chr(byte)
            except ValueError:
                return []
            mapped = CHAR_TO_CMD.get(char, CHAR_TO_CMD.get(char.lower()))
            if mapped:
                return [mapped]
            extra = self._extra_chars.get(char, self._extra_chars.get(char.lower()))
            return [extra] if extra else []

        if self._state == "esc":
            if byte == ord("["):
                self._state = "csi"
                return []
            if byte == ord("O"):
                self._state = "ss3"
                return []
            if byte == 0x1B:
                self._esc_t0 = now
                return ["stop_recording"]
            self._state = "idle"
            rest = self._feed_byte(byte, now)
            return ["stop_recording"] + rest

        if self._state in ("csi", "ss3"):
            if byte in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"),
                        ord("5"), ord("6"), ord("7"), ord("8"), ord("9"),
                        ord(";")):
                return []
            self._state = "idle"
            cmd = _ARROW_CMD.get(byte)
            return [cmd] if cmd else []

        self._state = "idle"
        return []


class ProtocolError(ValueError):
    """Raised when a framed message cannot be decoded or fails validation."""


def now_ns() -> int:
    return time.time_ns()


def make_leader_sample(
    *,
    seq: int,
    joints: Iterable[float],
    ee_pos: Iterable[float],
    ee_quat_wxyz: Iterable[float],
    gripper_norm: float,
    clutch: bool,
    deadman: bool,
    cmd: str = "none",
    timestamp_ns: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample = {
        "type": "leader_sample",
        "version": PROTOCOL_VERSION,
        "timestamp_ns": int(now_ns() if timestamp_ns is None else timestamp_ns),
        "seq": int(seq),
        "joints": [float(v) for v in joints],
        "ee_pos": [float(v) for v in ee_pos],
        "ee_quat_wxyz": [float(v) for v in ee_quat_wxyz],
        "gripper_norm": float(gripper_norm),
        "clutch": bool(clutch),
        "deadman": bool(deadman),
        "cmd": cmd if cmd in COMMANDS else "none",
    }
    if extra:
        sample["extra"] = extra
    validate_leader_sample(sample)
    return sample


def validate_leader_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise ProtocolError("sample must be a dict")
    missing = [k for k in REQUIRED_SAMPLE_KEYS if k not in sample]
    if missing:
        raise ProtocolError(f"missing keys: {missing}")
    if sample.get("type") != "leader_sample":
        raise ProtocolError(f"unexpected type {sample.get('type')!r}")
    if int(sample.get("version", -1)) != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {sample.get('version')!r}")
    joints = list(sample["joints"])
    if len(joints) != 6:
        raise ProtocolError(f"joints must have length 6, got {len(joints)}")
    if len(sample["ee_pos"]) != 3:
        raise ProtocolError("ee_pos must have length 3")
    if len(sample["ee_quat_wxyz"]) != 4:
        raise ProtocolError("ee_quat_wxyz must have length 4")
    grip = float(sample["gripper_norm"])
    if not (-0.05 <= grip <= 1.05):
        raise ProtocolError(f"gripper_norm out of range: {grip}")
    sample["gripper_norm"] = float(min(1.0, max(0.0, grip)))
    cmd = sample.get("cmd") or "none"
    if cmd not in COMMANDS:
        raise ProtocolError(f"unknown cmd {cmd!r}")
    sample["cmd"] = cmd
    return sample


def encode_message(obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload too large: {len(payload)} bytes")
    return HEADER.pack(len(payload)) + payload


def decode_payload(payload: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("JSON root must be an object")
    return obj


def send_message(sock: socket.socket, obj: Dict[str, Any]) -> None:
    data = encode_message(obj)
    sock.sendall(data)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(sock: socket.socket) -> Dict[str, Any]:
    header = recv_exact(sock, HEADER.size)
    (length,) = HEADER.unpack(header)
    if length <= 0 or length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"invalid payload length {length}")
    payload = recv_exact(sock, length)
    return decode_payload(payload)


def recv_message_from_file(fh) -> Optional[Dict[str, Any]]:
    header = fh.read(HEADER.size)
    if not header:
        return None
    if len(header) < HEADER.size:
        raise ProtocolError("truncated frame header")
    (length,) = HEADER.unpack(header)
    if length <= 0 or length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"invalid payload length {length}")
    payload = fh.read(length)
    if len(payload) < length:
        raise ProtocolError("truncated frame payload")
    return decode_payload(payload)


def write_message_to_file(fh, obj: Dict[str, Any]) -> None:
    fh.write(encode_message(obj))
    fh.flush()


def connect_with_retry(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout_s: float = 60.0,
    retry_s: float = 0.25,
) -> socket.socket:
    deadline = time.time() + timeout_s
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(None)
            return sock
        except OSError as exc:
            last_err = exc
            sock.close()
            time.sleep(retry_s)
    raise ConnectionError(f"could not connect to {host}:{port}: {last_err}")


def bind_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    backlog: int = 8,
) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.settimeout(0.2)
    return sock


def parse_endpoint(value: str) -> Tuple[str, int]:
    if ":" not in value:
        return DEFAULT_HOST, int(value)
    host, port = value.rsplit(":", 1)
    return host or DEFAULT_HOST, int(port)
