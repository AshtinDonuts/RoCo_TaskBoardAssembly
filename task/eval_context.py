"""Pure helpers for contextual evaluation trials.

The simulator-specific placement code lives with the scene setup. Keeping
the ordered-list lookup and endpoint selection here makes the behavior easy
to test without importing Isaac Sim.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def previous_part(current_part: str, ordered_parts: Sequence[str]):
    """Return the immediate predecessor of ``current_part``.

    ``None`` is returned for the first item. A missing item is an error rather
    than silently treating it as a first item, since that would make a
    contextual evaluation appear valid while using the wrong scene.
    """
    parts = tuple(ordered_parts)
    try:
        index = parts.index(current_part)
    except ValueError as exc:
        raise ValueError(
            f"current part {current_part!r} is not present in the ordered "
            f"task list {parts!r}"
        ) from exc
    return parts[index - 1] if index else None


def _first_present(mapping: Mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def successful_target_spec(config: Mapping) -> dict:
    """Select the configured endpoint representing a successful placement.

    Snap parts use their final ``connect_*`` pose when available, because
    that is the pose held by the fixed joint after a successful snap. Open
    parts use ``grade_pos`` when configured (the post-release settled target)
    and otherwise fall back to ``place_pos``.
    """
    release_mode = config.get("release_mode", "open")
    if release_mode == "snap":
        snap = config.get("snap")
        if not isinstance(snap, Mapping):
            raise ValueError(
                "release_mode='snap' requires a snap configuration dict"
            )
        position = _first_present(snap, "connect_pos", "target_pos")
        rotation = _first_present(snap, "connect_rot", "target_rot")
        if position is None or rotation is None:
            raise ValueError(
                "snap contextual placement requires target_pos/target_rot "
                "or connect_pos/connect_rot"
            )
        return {
            "position": position,
            "rotation": rotation,
            "measure": "mesh_pose",
            "fixed_joint": True,
            "parent_body_path": snap.get("parent_body_path", ""),
            "mesh_path": snap.get("mesh_path"),
        }

    position = _first_present(config, "grade_pos", "place_pos")
    if position is None:
        raise ValueError(
            "open contextual placement requires grade_pos or place_pos"
        )
    return {
        "position": position,
        "rotation": None,
        "measure": (
            "aabb_midpoint"
            if config.get("grade_use_aabb", False)
            else "mesh_translation"
        ),
        "fixed_joint": False,
        "parent_body_path": "",
        "mesh_path": None,
    }
