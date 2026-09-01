"""Isaac-side client for the isolated LeRobot recorder sidecar."""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import time
from typing import Any, Dict, Optional

HEADER = struct.Struct(">I")


class RecorderClient:
    def __init__(
        self,
        server_py: str,
        server_script: str,
        log_path: str,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        keep = (
            "HOME",
            "USER",
            "LANG",
            "LC_ALL",
            "HF_HOME",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "CUDA_VISIBLE_DEVICES",
        )
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin:"
            + os.path.dirname(os.path.abspath(server_py)),
            "PYTHONUNBUFFERED": "1",
        }
        for key in keep:
            if key in os.environ:
                env[key] = os.environ[key]
        if extra_env:
            env.update(extra_env)
        # Force CPU recording so Isaac keeps the GPU.
        env["CUDA_VISIBLE_DEVICES"] = extra_env.get("CUDA_VISIBLE_DEVICES", "") if extra_env else ""
        self._log = open(log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [server_py, server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=env,
            cwd=os.path.dirname(os.path.abspath(server_script)),
        )
        time.sleep(0.5)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"recorder sidecar died on startup (exit {self._proc.returncode}); see {log_path}"
            )

    def send(self, obj: Dict[str, Any], wait: bool = True) -> Optional[Dict[str, Any]]:
        payload = pickle.dumps(obj, protocol=4)
        self._proc.stdin.write(HEADER.pack(len(payload)) + payload)
        self._proc.stdin.flush()
        if not wait:
            return None
        return self._recv()

    def _recv(self) -> Dict[str, Any]:
        header = self._proc.stdout.read(HEADER.size)
        if len(header) < HEADER.size:
            raise RuntimeError("recorder sidecar closed")
        (n,) = HEADER.unpack(header)
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("recorder sidecar closed during payload")
            buf += chunk
        return pickle.loads(buf)

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                try:
                    self.send({"cmd": "shutdown"}, wait=False)
                except Exception:
                    pass
                self._proc.terminate()
                self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        try:
            self._log.close()
        except Exception:
            pass
