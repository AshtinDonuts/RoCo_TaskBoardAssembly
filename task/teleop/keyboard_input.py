"""Keyboard input backends for in-process Cartesian teleop.

Preference order:
1. ``carb.input`` (Isaac viewport has focus — the usual case)
2. ``pynput`` global listener (optional)
3. controlling TTY ``/dev/tty`` (terminal must keep focus)
"""
from __future__ import annotations

import os
import select
import sys
import threading
import time
from typing import Dict, Optional, Set

# Motion / gripper are level-triggered (held). Task/record are edge-triggered.
HOLD_KEYS = {
    "i": "ee+y",
    "k": "ee-y",
    "j": "ee-x",
    "l": "ee+x",
    "t": "ee+z",
    "g": "ee-z",
    "q": "ee+yaw",
    "a": "ee-yaw",
    "w": "ee+pitch",
    "d": "ee-pitch",
    "z": "ee+roll",
    "c": "ee-roll",
    "f": "grip_close",
    "v": "grip_open",
}

EDGE_CHARS = {
    " ": "clutch_toggle",
    "r": "recenter",
    "p": "pause",
    "u": "resume",
    "n": "part_done",
    "x": "abort",
    "e": "estop",
    "s": "start",
}

ARROW_EDGE = {
    "right": "save_episode",
    "left": "rerecord_episode",
    "esc": "stop_recording",
}


class KeyboardInput:
    """Thread-safe held-key + edge-command source."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: Set[str] = set()
        self._edges: list[str] = []
        self._backend = "none"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._carb = None
        self._carb_keyboard = None
        self._carb_map: Dict[str, object] = {}

    @property
    def backend(self) -> str:
        return self._backend

    def start(self) -> str:
        if self._try_start_carb():
            return self._backend
        if self._try_start_pynput():
            return self._backend
        if self._try_start_tty():
            return self._backend
        self._backend = "none"
        return self._backend

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def poll_held(self) -> Set[str]:
        if self._backend == "carb":
            self._poll_carb()
        with self._lock:
            return set(self._held)

    def pop_edges(self) -> list[str]:
        with self._lock:
            edges = list(self._edges)
            self._edges.clear()
            return edges

    def _push_edge(self, token: str) -> None:
        with self._lock:
            self._edges.append(token)

    def _set_held_char(self, char: str, pressed: bool) -> None:
        token = HOLD_KEYS.get(char)
        if token is None:
            return
        with self._lock:
            if pressed:
                self._held.add(token)
            else:
                self._held.discard(token)

    def _try_start_carb(self) -> bool:
        try:
            import carb
            import carb.input
            import omni.appwindow
        except Exception:
            return False
        try:
            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            iface = carb.input.acquire_input_interface()
        except Exception:
            return False
        keymap = {}
        for name in list(HOLD_KEYS) + [k for k in EDGE_CHARS if k != " "]:
            key_name = name.upper()
            key = getattr(carb.input.KeyboardInput, key_name, None)
            if key is not None:
                keymap[name] = key
        space = getattr(carb.input.KeyboardInput, "SPACE", None)
        if space is not None:
            keymap[" "] = space
        # Arrow / escape for recording
        for label, attr in (
            ("right", "RIGHT"),
            ("left", "LEFT"),
            ("esc", "ESCAPE"),
        ):
            key = getattr(carb.input.KeyboardInput, attr, None)
            if key is not None:
                keymap[label] = key
        if len(keymap) < 5:
            return False
        self._carb = iface
        self._carb_keyboard = keyboard
        self._carb_map = keymap
        self._backend = "carb"
        self._prev_edge_down: Set[str] = set()
        print("[keyboard_input] using Isaac carb.input (focus the viewport)", flush=True)
        return True

    def _poll_carb(self) -> None:
        iface = self._carb
        keyboard = self._carb_keyboard
        if iface is None or keyboard is None:
            return
        held: Set[str] = set()
        down_edges: Set[str] = set()
        for name, key in self._carb_map.items():
            try:
                pressed = bool(iface.get_keyboard_value(keyboard, key))
            except Exception:
                pressed = False
            if name in HOLD_KEYS and pressed:
                held.add(HOLD_KEYS[name])
            if name in EDGE_CHARS and pressed:
                down_edges.add(EDGE_CHARS[name])
            if name in ARROW_EDGE and pressed:
                down_edges.add(ARROW_EDGE[name])
        with self._lock:
            self._held = held
            prev = getattr(self, "_prev_edge_down", set())
            for token in down_edges - prev:
                self._edges.append(token)
            self._prev_edge_down = down_edges

    def _try_start_pynput(self) -> bool:
        try:
            from pynput import keyboard
        except Exception:
            return False

        def on_press(key):
            char = self._pynput_char(key)
            if char in HOLD_KEYS:
                self._set_held_char(char, True)
            elif char in EDGE_CHARS:
                self._push_edge(EDGE_CHARS[char])
            else:
                edge = self._pynput_special(key)
                if edge:
                    self._push_edge(edge)

        def on_release(key):
            char = self._pynput_char(key)
            if char in HOLD_KEYS:
                self._set_held_char(char, False)

        try:
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.daemon = True
            listener.start()
            self._thread = listener  # type: ignore[assignment]
            self._backend = "pynput"
            return True
        except Exception:
            return False

    @staticmethod
    def _pynput_char(key) -> Optional[str]:
        try:
            if hasattr(key, "char") and key.char:
                return key.char.lower()
        except Exception:
            return None
        return None

    @staticmethod
    def _pynput_special(key) -> Optional[str]:
        name = getattr(key, "name", None)
        if name == "space":
            return "clutch_toggle"
        if name == "right":
            return "save_episode"
        if name == "left":
            return "rerecord_episode"
        if name == "esc":
            return "stop_recording"
        return None

    def _try_start_tty(self) -> bool:
        try:
            fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            if not sys.stdin.isatty():
                return False
            fd = sys.stdin.fileno()
        self._backend = "tty"
        self._thread = threading.Thread(
            target=self._tty_loop, args=(fd,), name="keyboard-tty", daemon=True
        )
        self._thread.start()
        return True

    def _tty_loop(self, fd: int) -> None:
        import termios
        import tty

        from .protocol import KeyDecoder

        decoder = KeyDecoder(extra_chars=HOLD_KEYS)
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            old = None
        try:
            if old is not None:
                tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.05)
                now = time.monotonic()
                if ready:
                    try:
                        data = os.read(fd, 32)
                    except OSError:
                        data = b""
                    if not data:
                        continue
                    for token in decoder.feed(data, now):
                        self._handle_tty_token(token, pressed=True)
                for token in decoder.poll_timeout(now):
                    self._handle_tty_token(token, pressed=True)
                # TTY cannot see key-up; clear holds shortly after edge.
                with self._lock:
                    # Keep last hold alive for a short pulse window via timestamps.
                    pass
        finally:
            if old is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
                except termios.error:
                    pass
            if fd != sys.stdin.fileno():
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _handle_tty_token(self, token: str, pressed: bool) -> None:
        if token in HOLD_KEYS.values():
            # Pulse hold for one poll cycle by recording expiry.
            with self._lock:
                self._held.add(token)
            self._schedule_release(token, 0.12)
            return
        if token in EDGE_CHARS.values() or token in ARROW_EDGE.values():
            self._push_edge(token)
            return
        # KeyDecoder may return CHAR_TO_CMD names already.
        if token in (
            "clutch_toggle",
            "recenter",
            "pause",
            "resume",
            "part_done",
            "abort",
            "estop",
            "start",
            "save_episode",
            "rerecord_episode",
            "stop_recording",
        ):
            self._push_edge(token)

    def _schedule_release(self, token: str, delay_s: float) -> None:
        def _clear():
            time.sleep(delay_s)
            with self._lock:
                self._held.discard(token)

        threading.Thread(target=_clear, daemon=True).start()
