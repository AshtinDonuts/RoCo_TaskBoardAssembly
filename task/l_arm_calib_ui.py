"""omni.ui panel + config writers for L-arm init joint calibration.

Sliders are labeled in degrees; internal targets are radians.
Save updates param_config.INIT_JOINT_TARGETS and Lula descriptor YAMLs.
"""
from __future__ import annotations

import math
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# URDF limits for L_arm_j1..j7 (radians) from robot/vega_1u_gripper.urdf.
# Note j2/j3/j7 differ from the mirrored R-arm limits.
L_ARM_JOINTS: Tuple[str, ...] = tuple(f"L_arm_j{i}" for i in range(1, 8))

L_ARM_LIMITS_RAD: Dict[str, Tuple[float, float]] = {
    "L_arm_j1": (-3.071, 3.071),
    "L_arm_j2": (-0.453, 1.5529999),
    "L_arm_j3": (-3.15, 3.15),
    "L_arm_j4": (-3.071, 0.244),
    "L_arm_j5": (-3.071, 3.071),
    "L_arm_j6": (-1.396, 1.396),
    "L_arm_j7": (-1.378, 1.1169999),
}

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAM_CONFIG_PATH = os.path.join(_TASK_DIR, "param_config.py")
_CONTROLLERS_DIR = os.path.join(_TASK_DIR, "controllers")

_L_DESC_FILES = (
    "vega_1u_L_arm_description_armonly.yaml",
    "vega_1u_L_arm_description_liftonly.yaml",
    "vega_1u_L_arm_description.yaml",
)
_R_DESC_FILES = (
    "vega_1u_R_arm_description_armonly.yaml",
    "vega_1u_R_arm_description_liftonly.yaml",
    "vega_1u_R_arm_description.yaml",
)


def rad_to_deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def _format_rad_fixed(rad: float) -> str:
    # Match repo style: -0.52359878 / 1.04719755 (8 decimal places).
    return f"{float(rad):.8f}"


def _deg_comment(rad: float) -> str:
    deg = rad_to_deg(rad)
    sign = "+" if deg >= 0 else ""
    if abs(deg - round(deg)) < 1e-4:
        return f"{sign}{int(round(deg))} deg"
    return f"{sign}{deg:.2f} deg"


def _deg_list_comment(targets_rad: Dict[str, float]) -> str:
    parts = []
    for j in L_ARM_JOINTS:
        deg = rad_to_deg(targets_rad[j])
        if abs(deg - round(deg)) < 1e-4:
            parts.append(f"{'+' if deg >= 0 else ''}{int(round(deg))}")
        else:
            parts.append(f"{deg:+.2f}")
    return ", ".join(parts)


def print_targets(targets_rad: Dict[str, float], prefix: str = "[l_arm_calib]") -> None:
    print(f"{prefix} L-arm joint targets:", flush=True)
    for jname in L_ARM_JOINTS:
        rad = float(targets_rad[jname])
        print(
            f"{prefix}   {jname}: {_format_rad_fixed(rad)} rad  "
            f"({_deg_comment(rad)})",
            flush=True,
        )
    print(f"{prefix} INIT_JOINT_TARGETS snippet:", flush=True)
    for jname in L_ARM_JOINTS:
        rad = float(targets_rad[jname])
        print(
            f'{prefix}     "{jname}": {_format_rad_fixed(rad)},   #  {_deg_comment(rad)}',
            flush=True,
        )


def save_l_arm_init_targets(targets_rad: Dict[str, float]) -> List[str]:
    """Write L_arm_j* into param_config + Lula YAMLs. Returns paths touched."""
    for jname in L_ARM_JOINTS:
        if jname not in targets_rad:
            raise KeyError(f"missing joint target: {jname}")

    touched: List[str] = []
    touched.append(_update_param_config(targets_rad))
    for fname in _L_DESC_FILES:
        path = os.path.join(_CONTROLLERS_DIR, fname)
        _update_l_descriptor_default_q(path, targets_rad)
        touched.append(path)
    for fname in _R_DESC_FILES:
        path = os.path.join(_CONTROLLERS_DIR, fname)
        _update_r_descriptor_fixed_l(path, targets_rad)
        touched.append(path)
    return touched


def _update_param_config(targets_rad: Dict[str, float]) -> str:
    path = _PARAM_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Rewrite L_arm_j* in the main INIT_JOINT_TARGETS dict (before the
    # R_ARM_TUCKED if/else block that only owns R_arm_j*).
    for jname in L_ARM_JOINTS:
        rad = float(targets_rad[jname])
        new_line = (
            f'    "{jname}": {_format_rad_fixed(rad)},   #  {_deg_comment(rad)}'
        )
        pattern = rf'^(\s*)"{re.escape(jname)}"\s*:\s*[^,\n]+,.*$'
        text2, n = re.subn(pattern, new_line, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"could not update {jname} in {path}")
        text = text2

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _replace_default_q_arm_slice(text: str, arm_vals: Sequence[float]) -> str:
    """Replace the last 7 entries of the first default_q: [...] block."""
    match = re.search(
        r"(default_q:\s*\[)(.*?)(\])",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("default_q block not found")
    inner = match.group(2)
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", inner)
    if len(nums) < 7:
        raise RuntimeError(
            f"default_q has {len(nums)} numbers; need >= 7 to replace arm slice"
        )
    prefix = nums[:-7]
    new_nums = list(prefix) + [_format_rad_fixed(v) for v in arm_vals]
    indent = "    "
    new_inner = f"\n{indent}{', '.join(new_nums)}\n"
    return (
        text[: match.start()]
        + match.group(1)
        + new_inner
        + match.group(3)
        + text[match.end() :]
    )


def _update_l_descriptor_default_q(path: str, targets_rad: Dict[str, float]) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    arm_vals = [float(targets_rad[j]) for j in L_ARM_JOINTS]
    text = _replace_default_q_arm_slice(text, arm_vals)
    deg_list = _deg_list_comment(targets_rad)
    text2, n = re.subn(
        r"^(\s*#\s*.*L_arm_j1\.\.j7\s*=\s*).*$",
        rf"\g<1>[{deg_list}] deg",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n:
        text = text2
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _update_r_descriptor_fixed_l(path: str, targets_rad: Dict[str, float]) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for jname in L_ARM_JOINTS:
        rad = float(targets_rad[jname])
        pattern = (
            rf"^(\s*-\s*\{{name:\s*{re.escape(jname)},\s*rule:\s*fixed,\s*value:\s*)"
            rf"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
            rf"(\s*\}}.*)$"
        )
        repl = rf"\g<1>{_format_rad_fixed(rad)}\g<2>"
        text2, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"could not update fixed {jname} in {path}")
        text = text2
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class LArmCalibUI:
    """Kit window with L_arm_j1..j7 sliders (degrees) and Reset/Print/Save."""

    def __init__(
        self,
        initial_targets_rad: Dict[str, float],
        on_save: Optional[Callable[[Dict[str, float]], None]] = None,
        on_print: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        import omni.ui as ui

        self._ui = ui
        self._initial = {j: float(initial_targets_rad[j]) for j in L_ARM_JOINTS}
        self._targets = dict(self._initial)
        self._on_save = on_save
        self._on_print = on_print
        self._teleport_requested = True
        self._models: Dict[str, object] = {}
        self._status_label = None

        self._window = ui.Window(
            "L Arm Init Calibration",
            width=420,
            height=420,
            visible=True,
        )
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label(
                    "Adjust L_arm_j1..j7 (degrees). Save writes param_config + Lula YAMLs.",
                    word_wrap=True,
                    height=0,
                )
                ui.Spacer(height=4)
                for jname in L_ARM_JOINTS:
                    lo_rad, hi_rad = L_ARM_LIMITS_RAD[jname]
                    with ui.HStack(height=0):
                        ui.Label(jname, width=90)
                        model = ui.SimpleFloatModel(rad_to_deg(self._targets[jname]))
                        self._models[jname] = model
                        model.add_value_changed_fn(
                            lambda m, name=jname: self._on_slider(name, m)
                        )
                        ui.FloatSlider(
                            model,
                            min=rad_to_deg(lo_rad),
                            max=rad_to_deg(hi_rad),
                            step=0.1,
                        )
                        ui.FloatDrag(
                            model,
                            min=rad_to_deg(lo_rad),
                            max=rad_to_deg(hi_rad),
                            step=0.1,
                            width=70,
                        )
                ui.Spacer(height=8)
                with ui.HStack(spacing=8, height=0):
                    ui.Button("Reset", clicked_fn=self._on_reset, height=28)
                    ui.Button("Print", clicked_fn=self._on_print_clicked, height=28)
                    ui.Button("Save", clicked_fn=self._on_save_clicked, height=28)
                self._status_label = ui.Label("", word_wrap=True, height=0)

    def _on_slider(self, jname: str, model) -> None:
        deg = float(model.get_value_as_float())
        lo, hi = L_ARM_LIMITS_RAD[jname]
        rad = max(lo, min(hi, deg_to_rad(deg)))
        prev = self._targets[jname]
        self._targets[jname] = rad
        if abs(rad - prev) > math.radians(15.0):
            self._teleport_requested = True

    def _on_reset(self) -> None:
        for jname in L_ARM_JOINTS:
            self._targets[jname] = float(self._initial[jname])
            self._models[jname].set_value(rad_to_deg(self._targets[jname]))
        self._teleport_requested = True
        self._set_status("Reset to startup INIT_JOINT_TARGETS.")

    def _on_print_clicked(self) -> None:
        targets = self.get_targets_rad()
        if self._on_print is not None:
            self._on_print(targets)
        else:
            print_targets(targets)
        self._set_status("Printed targets to console.")

    def _on_save_clicked(self) -> None:
        targets = self.get_targets_rad()
        if self._on_save is not None:
            self._on_save(targets)
        else:
            touched = save_l_arm_init_targets(targets)
            print_targets(targets)
            print("[l_arm_calib] saved to:", flush=True)
            for p in touched:
                print(f"[l_arm_calib]   {p}", flush=True)
        self._initial = dict(targets)
        self._set_status("Saved INIT_JOINT_TARGETS + Lula YAMLs.")

    def _set_status(self, msg: str) -> None:
        if self._status_label is not None:
            self._status_label.text = msg
        print(f"[l_arm_calib] {msg}", flush=True)

    def get_targets_rad(self) -> Dict[str, float]:
        return {j: float(self._targets[j]) for j in L_ARM_JOINTS}

    def request_teleport(self) -> None:
        self._teleport_requested = True

    def consume_teleport_request(self) -> bool:
        flag = bool(self._teleport_requested)
        self._teleport_requested = False
        return flag
