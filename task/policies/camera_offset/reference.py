"""Nominal head-camera assets for competition-faithful offset estimation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .constants import BUNDLE_VERSION, DEFAULT_BUFFER_FRAMES


def _sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


@dataclass
class PartTemplate:
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    search_center_uv: np.ndarray  # template centre in the nominal image
    jacobian_xy_per_px: Optional[np.ndarray] = None  # local world-per-pixel

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[-1] < 3:
            raise ValueError(f"{self.name}: rgb template must be HxWx3")
        if self.depth.shape[:2] != self.rgb.shape[:2]:
            raise ValueError(f"{self.name}: depth/rgb template shape mismatch")
        if self.mask.shape != self.rgb.shape[:2]:
            raise ValueError(f"{self.name}: mask shape mismatch")
        if self.search_center_uv.shape != (2,):
            raise ValueError(f"{self.name}: search_center_uv must be length-2")
        if (self.jacobian_xy_per_px is not None
                and np.asarray(self.jacobian_xy_per_px).shape != (2, 2)):
            raise ValueError(f"{self.name}: jacobian must be 2x2")


@dataclass
class ReferenceBundle:
    """Submission-legal nominal reference. No seeds or shifted ground truth."""

    version: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    board_mask: np.ndarray
    jacobian_xy_per_px: np.ndarray  # 2x2, world metres per pixel
    board_center_uv: np.ndarray
    parts: Dict[str, PartTemplate]
    expected_hw: Tuple[int, int]
    content_hash: str
    buffer_frames: int = DEFAULT_BUFFER_FRAMES
    camera_R_world_from_cam: Optional[np.ndarray] = None
    camera_t_world: Optional[np.ndarray] = None
    plane_z: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.version) != int(BUNDLE_VERSION):
            raise ValueError(
                f"unsupported camera reference version {self.version}; "
                f"expected {BUNDLE_VERSION}"
            )
        h, w = int(self.expected_hw[0]), int(self.expected_hw[1])
        if self.rgb.shape[:2] != (h, w):
            raise ValueError(
                f"reference rgb shape {self.rgb.shape[:2]} != expected {(h, w)}"
            )
        if self.depth.shape[:2] != (h, w):
            raise ValueError("reference depth shape mismatch")
        if self.board_mask.shape != (h, w):
            raise ValueError("board mask shape mismatch")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if self.jacobian_xy_per_px.shape != (2, 2):
            raise ValueError("jacobian must be 2x2")
        if not np.all(np.isfinite(self.jacobian_xy_per_px)):
            raise ValueError("jacobian has non-finite values")
        for part in self.parts.values():
            part.validate()
        if self.content_hash != self.compute_hash():
            raise ValueError("camera reference hash mismatch")

    def compute_hash(self) -> str:
        parts_blob = []
        for name in sorted(self.parts):
            p = self.parts[name]
            parts_blob.extend(
                [
                    name.encode(),
                    np.ascontiguousarray(p.rgb).tobytes(),
                    np.ascontiguousarray(p.depth).tobytes(),
                    np.ascontiguousarray(p.mask).tobytes(),
                    np.ascontiguousarray(p.search_center_uv).tobytes(),
                ]
            )
        return _sha256_bytes(
            np.ascontiguousarray(self.rgb).tobytes(),
            np.ascontiguousarray(self.depth).tobytes(),
            np.ascontiguousarray(self.intrinsics).tobytes(),
            np.ascontiguousarray(self.board_mask).tobytes(),
            np.ascontiguousarray(self.jacobian_xy_per_px).tobytes(),
            *parts_blob,
        )

    def assert_observation_shape(self, rgb, depth=None, intrinsics=None) -> None:
        if rgb is None:
            raise RuntimeError(
                "CameraOffsetScriptedPolicy requires Observation.rgb['head']; "
                "set TASK_ENABLE_CAMERA_OUTPUT=1"
            )
        arr = np.asarray(rgb)
        if arr.shape[:2] != tuple(self.expected_hw):
            raise RuntimeError(
                f"head rgb shape {arr.shape[:2]} != reference {self.expected_hw}"
            )
        if depth is not None and np.asarray(depth).shape[:2] != tuple(self.expected_hw):
            raise RuntimeError("head depth shape does not match reference")
        if intrinsics is not None:
            K = np.asarray(intrinsics, dtype=np.float64)
            if K.shape != (3, 3):
                raise RuntimeError("head intrinsics must be 3x3")
            if not np.allclose(K, self.intrinsics, rtol=1e-3, atol=5e-2):
                raise RuntimeError(
                    "head intrinsics differ from the packaged camera reference"
                )

    @classmethod
    def default_dir(cls) -> Path:
        return Path(__file__).resolve().parents[1] / "camera_reference"

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "head_rgb.npy", self.rgb)
        np.save(directory / "head_depth.npy", self.depth)
        np.save(directory / "head_intrinsics.npy", self.intrinsics)
        np.save(directory / "board_mask.npy", self.board_mask.astype(np.uint8))
        parts_dir = directory / "parts"
        parts_dir.mkdir(exist_ok=True)
        part_meta = {}
        for name, part in self.parts.items():
            np.save(parts_dir / f"{name}_rgb.npy", part.rgb)
            np.save(parts_dir / f"{name}_depth.npy", part.depth)
            np.save(parts_dir / f"{name}_mask.npy", part.mask.astype(np.uint8))
            part_meta[name] = {
                "search_center_uv": [float(x) for x in part.search_center_uv],
            }
            if part.jacobian_xy_per_px is not None:
                part_meta[name]["jacobian_xy_per_px"] = (
                    np.asarray(part.jacobian_xy_per_px).tolist()
                )
        payload = {
            "version": int(self.version),
            "expected_hw": [int(self.expected_hw[0]), int(self.expected_hw[1])],
            "board_center_uv": [float(x) for x in self.board_center_uv],
            "jacobian_xy_per_px": self.jacobian_xy_per_px.tolist(),
            "content_hash": self.content_hash,
            "buffer_frames": int(self.buffer_frames),
            "parts": part_meta,
            "plane_z": None if self.plane_z is None else float(self.plane_z),
        }
        if self.camera_R_world_from_cam is not None:
            payload["camera_R_world_from_cam"] = (
                np.asarray(self.camera_R_world_from_cam).tolist()
            )
        if self.camera_t_world is not None:
            payload["camera_t_world"] = np.asarray(self.camera_t_world).tolist()
        (directory / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return directory

    @classmethod
    def load(cls, directory) -> "ReferenceBundle":
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"camera reference missing at {manifest_path}. "
                "Capture a nominal head frame and run "
                "scripts/build_camera_reference.py"
            )
        manifest = json.loads(manifest_path.read_text())
        rgb = np.load(directory / "head_rgb.npy")
        depth = np.load(directory / "head_depth.npy")
        K = np.load(directory / "head_intrinsics.npy")
        board_mask = np.load(directory / "board_mask.npy").astype(bool)
        parts = {}
        for name, meta in (manifest.get("parts") or {}).items():
            part_jac = meta.get("jacobian_xy_per_px")
            parts[name] = PartTemplate(
                name=name,
                rgb=np.load(directory / "parts" / f"{name}_rgb.npy"),
                depth=np.load(directory / "parts" / f"{name}_depth.npy"),
                mask=np.load(directory / "parts" / f"{name}_mask.npy").astype(bool),
                search_center_uv=np.asarray(
                    meta["search_center_uv"], dtype=np.float64
                ),
                jacobian_xy_per_px=(
                    None if part_jac is None
                    else np.asarray(part_jac, dtype=np.float64)
                ),
            )
        R = manifest.get("camera_R_world_from_cam")
        t = manifest.get("camera_t_world")
        bundle = cls(
            version=int(manifest["version"]),
            rgb=rgb,
            depth=depth,
            intrinsics=np.asarray(K, dtype=np.float64),
            board_mask=board_mask,
            jacobian_xy_per_px=np.asarray(
                manifest["jacobian_xy_per_px"], dtype=np.float64
            ),
            board_center_uv=np.asarray(
                manifest["board_center_uv"], dtype=np.float64
            ),
            parts=parts,
            expected_hw=tuple(int(x) for x in manifest["expected_hw"]),
            content_hash=str(manifest["content_hash"]),
            buffer_frames=int(
                manifest.get("buffer_frames", DEFAULT_BUFFER_FRAMES)
            ),
            camera_R_world_from_cam=(
                None if R is None else np.asarray(R, dtype=np.float64)
            ),
            camera_t_world=(
                None if t is None else np.asarray(t, dtype=np.float64)
            ),
            plane_z=manifest.get("plane_z"),
        )
        bundle.validate()
        return bundle


def make_bundle(*, rgb, depth, intrinsics, board_mask, jacobian_xy_per_px,
                board_center_uv, parts: Mapping[str, PartTemplate],
                buffer_frames=DEFAULT_BUFFER_FRAMES,
                camera_R_world_from_cam=None, camera_t_world=None,
                plane_z=None, diagnostics=None) -> ReferenceBundle:
    rgb = np.asarray(rgb)
    bundle = ReferenceBundle(
        version=BUNDLE_VERSION,
        rgb=rgb,
        depth=np.asarray(depth),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        board_mask=np.asarray(board_mask, dtype=bool),
        jacobian_xy_per_px=np.asarray(jacobian_xy_per_px, dtype=np.float64),
        board_center_uv=np.asarray(board_center_uv, dtype=np.float64).reshape(2),
        parts=dict(parts),
        expected_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
        content_hash="",
        buffer_frames=int(buffer_frames),
        camera_R_world_from_cam=camera_R_world_from_cam,
        camera_t_world=camera_t_world,
        plane_z=plane_z,
        diagnostics=dict(diagnostics or {}),
    )
    bundle.content_hash = bundle.compute_hash()
    bundle.validate()
    return bundle
