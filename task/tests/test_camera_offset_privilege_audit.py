"""Static privilege audit for the camera-only submission policy."""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_TASK_DIR = Path(__file__).resolve().parents[1]
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from policies.camera_offset.constants import (  # noqa: E402
    BUNDLE_VERSION,
    XY_MAX_M,
    XY_MIN_M,
)
from policies.camera_offset.reference import ReferenceBundle  # noqa: E402

POLICY_DIRS = [
    _TASK_DIR / "policies" / "camera_offset",
    _TASK_DIR / "policies" / "camera_offset_scripted.py",
]
REF_DIR = _TASK_DIR / "policies" / "camera_reference"

FORBIDDEN_MODULE_SUBSTR = (
    "eval_randomization",
    "XYRandomization",
    "param_config",
    "pxr",
    "UsdGeom",
    "isaacsim",
)
FORBIDDEN_NAMES = {
    "XYRandomization",
    "get_part_config",
    "get_world_pose",
    "GetLocalTransformation",
    "GetWorldTransformation",
    "PART_INIT_POSES",
}
ALLOWED_MANIFEST_KEYS = {
    "version",
    "expected_hw",
    "board_center_uv",
    "jacobian_xy_per_px",
    "content_hash",
    "buffer_frames",
    "parts",
    "plane_z",
    "camera_R_world_from_cam",
    "camera_t_world",
}
ALLOWED_PART_KEYS = {"search_center_uv", "jacobian_xy_per_px"}
ALLOWED_ACTION_MODULES = {
    "isaacsim.core.utils.types",
    "omni.isaac.core.utils.types",
}


def _iter_policy_py():
    for path in POLICY_DIRS:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.glob("*.py"))


class PrivilegeAuditTests(unittest.TestCase):
    def test_no_privileged_imports_or_calls(self):
        for path in _iter_policy_py():
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        for bad in FORBIDDEN_MODULE_SUBSTR:
                            self.assertNotIn(
                                bad, mod,
                                f"{path.name} imports forbidden module {mod}",
                            )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    # ArticulationAction typing only is allowed.
                    if mod in ALLOWED_ACTION_MODULES:
                        continue
                    for bad in FORBIDDEN_MODULE_SUBSTR:
                        self.assertNotIn(
                            bad, mod,
                            f"{path.name} imports forbidden module {mod}",
                        )
                elif isinstance(node, ast.Name):
                    self.assertNotIn(
                        node.id, FORBIDDEN_NAMES,
                        f"{path.name} references forbidden name {node.id}",
                    )
                elif isinstance(node, ast.Attribute):
                    self.assertNotIn(
                        node.attr, FORBIDDEN_NAMES,
                        f"{path.name} references forbidden attr {node.attr}",
                    )

    def test_camera_reference_is_nominal_only(self):
        self.assertTrue(REF_DIR.is_dir(), "camera_reference bundle missing")
        man = json.loads((REF_DIR / "manifest.json").read_text())
        self.assertTrue(set(man.keys()) <= ALLOWED_MANIFEST_KEYS)
        for key in man:
            self.assertNotRegex(key, r"(seed|trial|offset|shift|pick_pos|privileged)")
        for meta in man["parts"].values():
            self.assertTrue(set(meta.keys()) <= ALLOWED_PART_KEYS)
        for path in REF_DIR.rglob("*.npy"):
            arr = np.load(path, allow_pickle=False)
            self.assertNotEqual(arr.dtype, object)
        bundle = ReferenceBundle.load(REF_DIR)
        self.assertEqual(bundle.version, BUNDLE_VERSION)
        self.assertEqual(bundle.content_hash, bundle.compute_hash())

    def test_fairness_domain_duplicated_locally(self):
        self.assertEqual(XY_MIN_M, -0.01)
        self.assertEqual(XY_MAX_M, 0.01)
        # Policy package must not import evaluator sampling.
        text = "\n".join(p.read_text() for p in _iter_policy_py())
        self.assertNotIn("eval_randomization", text)
        self.assertNotIn("XYRandomization", text)


if __name__ == "__main__":
    unittest.main()
