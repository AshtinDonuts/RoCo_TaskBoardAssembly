# RoCo_TaskBoardAssembly — Fork Notes

Fork-specific setup and workflow notes that override or supplement the
upstream [`README.md`](README.md).

## Python environment

**Do not run `uv sync`.** The upstream README describes downloading a
self-contained `.venv/` (~18 GB) via `uv sync`. On this machine Isaac Sim
5.1.0.0 is already installed inside the Play2Perfect venv:

```
/home/khw/RoCoIROS26/play2perfect/.venv_isaacsim/
```

That venv contains `isaacsim==5.1.0.0` (plus `isaaclab==2.3.2.post1`),
which is the exact version pinned in `pyproject.toml`. Use it directly —
no separate install needed.

## Running the harness

From the repo root, always set `OMNI_KIT_ACCEPT_EULA` first, then invoke
the shared venv's Python **from inside `task/`** so that `param_config`,
`controllers`, and `policies` resolve as bare imports:

```bash
export OMNI_KIT_ACCEPT_EULA=YES

cd /home/khw/RoCoIROS26/RoCo_TaskBoardAssembly/task

/home/khw/RoCoIROS26/play2perfect/.venv_isaacsim/bin/python \
    run_pick_place.py
```

With a custom policy:

```bash
/home/khw/RoCoIROS26/play2perfect/.venv_isaacsim/bin/python \
    run_pick_place.py --policy policies.my_team.MyPolicy
```

Dump per-part pass/fail to JSON:

```bash
/home/khw/RoCoIROS26/play2perfect/.venv_isaacsim/bin/python \
    run_pick_place.py --results-json out/results.json
```

## Local fixes applied in this fork

### Observation prim paths (`scene_prims.py`)

`param_config.py` previously set `L_object_prim_path` and
`R_object_prim_path` to `/World/parts/rod_16mm`. After `rod_16mm` snaps
(FixedJoint child), the PhysX tensor view for that prim becomes invalid and
the harness crashes mid-sequence.

These paths have been moved to [`task/scene_prims.py`](task/scene_prims.py)
and now point at `/World/task_board/task_board_color` — the snap *anchor*,
never a joint-locked movable. Its tensor view stays valid across the entire
9-part sequence.

### URDF mesh paths

`robot/vega_1u_gripper.urdf` contained absolute paths from the authoring
machine (`file:///Yejun_Files/Works/Robotics/Results/Roco2026/...`). All 90
mesh references have been rewritten to relative paths (`meshes/<file>.obj`),
so the URDF resolves correctly from any machine and any URDF visualiser.

## Isaac Sim / venv status (this machine)

| Location | Isaac Sim | Notes |
|---|---|---|
| `RoCo_TaskBoardAssembly/.venv/` | **not created** | `uv sync` not run; not needed |
| `play2perfect/.venv_isaacsim/` | **5.1.0.0** | Use this — matches `pyproject.toml` pin |
| `~/isaacsim/` (`python.sh`) | **6.0.0-rc.59** | Newer major version; API mismatches likely |

GPU: NVIDIA RTX 4090, driver 570.169.
