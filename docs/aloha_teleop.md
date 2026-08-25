# ALOHA Solo to DexMate Vega 1U teleoperation

This pipeline uses the physical ALOHA Solo **leader** as a Cartesian input device
for the RoCo Industrial Task Board scene in Isaac Sim 5.1. By default the DexMate
`vega_1u_gripper` **right** arm is driven through Lula IK (`control.arms: "right"`
in [`config/teleop_export.json`](../config/teleop_export.json)). Set
`control.arms` to `"dual"` to drive both virtual arms from two leader TCP
streams. The physical follower is never launched. Demonstrations are written as
LeRobot v3 episodes with the challenge 44-D state / 14-D action contract.

## Environments

Keep these interpreters isolated. Never import ROS, Isaac Sim, and current
LeRobot into the same process.

| Role | Interpreter | How it is created |
| --- | --- | --- |
| ALOHA leader | ROS 2 Humble / Python 3.10 | existing `/home/khw/interbotix_ws` |
| Isaac Sim + RoCo harness | Python 3.11 via `uv` | `cd ~/RoCo_TaskBoardAssembly && uv sync` |
| LeRobot recorder | conda `lerobot` / Python 3.12, LeRobot 0.6.1 | `conda create -n lerobot python=3.12` then `pip install 'lerobot[dataset,training,viz]'` |

Pinned challenge commit is recorded in [`run_manifest.json`](../run_manifest.json).

## Machine notes

This host: Ubuntu 22.04, RTX 4080 16 GB, 32 GB RAM (Isaac Sim minimum),
NVIDIA driver **595.84**.

**Driver status on this machine (2026-08-15):** `import isaacsim` succeeds, but
launching the Kit/RTX harness (`task/run_pick_place.py`) segfaults in
`librtx.scenedb.plugin.so` under driver 595.84. That matches the known Isaac
Sim 5.1 incompatibility. Before collecting data, install a validated 580
production driver (needs sudo) and reboot:

```bash
# after confirming you have sudo
sudo apt install nvidia-driver-580
sudo reboot
# then:
cd /home/khw/RoCo_TaskBoardAssembly
OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_HEADLESS=1 \
  ./.venv/bin/python task/run_pick_place.py --max-parts 1 --max-sim-seconds 20 \
  --results-json /tmp/roco_smoke.json
```

Do not launch Isaac with less than about 12 GB RAM free. Do not train LeRobot
and run Isaac at the same time on 16 GB VRAM. The recorder sidecar is CPU-only.

```bash
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/preflight.py
```

## Install

```bash
# uv (already in ~/.local/bin) and git-lfs (already in ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES

cd /home/khw/RoCo_TaskBoardAssembly
uv sync    # ~18 GB Isaac Sim 5.1 stack

# Isolated LeRobot
conda activate lerobot
pip install 'lerobot[dataset,training,viz]'

# ROS package
cd /home/khw/interbotix_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select aloha_isaac_teleop --symlink-install
source install/setup.bash
```

Isaac smoke test after `uv sync`:

```bash
cd /home/khw/RoCo_TaskBoardAssembly
OMNI_KIT_ACCEPT_EULA=YES uv run python -c "import isaacsim; print('isaacsim ok')"
OMNI_KIT_ACCEPT_EULA=YES TASK_ENABLE_CAMERA_OUTPUT=1 \
  uv run python task/run_pick_place.py --max-parts 1 --max-sim-seconds 20 \
  --results-json /tmp/roco_smoke.json
```

If that crashes, switch to driver 580 and repeat.

## Normal collection

Terminal 1, hardware. Follower USB can stay disconnected.

```bash
source /opt/ros/humble/setup.bash
source /home/khw/interbotix_ws/install/setup.bash
ros2 launch aloha_isaac_teleop leader_only.launch.py robot:=aloha_solo
```

Default `control.arms` is `"right"`: that Solo leader maps onto DexMate **right**.
For `"dual"`, run two leader bridges on the ports in
`control.leader_endpoints` (left + right); Isaac connects to both.

The opening ceremony (drive to start pose) is **off by default**. Close the
leader gripper (or press `s`) to start; the arm stays where it is. Backdrive
then enables. **Clutch starts in track** (DexMate EE follows leader deltas).
To restore the start-pose motion:

```bash
ros2 launch aloha_isaac_teleop leader_only.launch.py robot:=aloha_solo opening_ceremony:=true
```

### WIP: Space pause / reanchor (not ready)

**Status: work in progress — do not rely on this for collection.** The intended
mechanism is relative clutching so you can park DexMate, reposition the physical
leader into a comfortable pose, then resume without yanking the virtual arm or
stitching a stop into the recording. Hardware tests after the current
implementation still failed to capture new origins on Space-2 / resume.

Intended (not yet working end-to-end):

1. **Space (pause)** — freeze physics, recording, and the episode timer; DexMate
   stays put. Reposition the real arm into a comfortable mid-range pose.
2. **Space again (track)** — atomically pair the frozen virtual EE with the new
   leader pose. DexMate should not move until you move the leader again; only
   subsequent physical deltas should apply.

Until this is fixed, prefer continuous tracking without Space clutch
repositioning. `p` / `u` share the same unfinished path. Gravity compensation
after gripper close is **off by default**; set
`control.gravity_compensation: true` in
[`config/teleop_export.json`](../config/teleop_export.json) (or launch with
`gravity_compensation:=true`) to enable it.

Terminal 2, Isaac + recorder:

```bash
export PATH="$HOME/.local/bin:$PATH"
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py
# optional:
#   --export-config config/teleop_export.json
#   --episode-time-s 600 --warmup-time-s 5 --num-episodes 1
```
### Operator keys (leader terminal)

Task / robot keys do **not** save or discard data:

| Key | Action |
| --- | --- |
| space | **WIP** clutch toggle: **track** ↔ **pause** (intended: freeze sim+recording then reanchor; not reliable yet) |
| r | recenter / recapture origins |
| p | **WIP** jump to pause (same unfinished path as Space) |
| u | **WIP** resume / reanchor (same unfinished path as Space) |
| n | mark current gravity-settled (`open`) part done |
| x | abort remaining parts (hold the robot; recording continues) |
| e | emergency hold (deadman off) |
| s | start |

LeRobot-style recording keys. Task success is logged only; it never decides whether an episode is kept:

| Key | Action |
| --- | --- |
| Right arrow | save the current episode early |
| Left arrow | discard this attempt and rerecord after warmup |
| Esc | save (if recording) and stop the session |
| episode timer | auto-save at `--episode-time-s` (default **600 s**) |

There is a **5 s warmup** (`--warmup-time-s`) before every episode, including rerecords. No dataset frames are written during warmup. Completing all 9 parts does not save or end recording; the robot holds until you save, rerecord, stop, or hit the episode timer.

ROS command topic: `/aloha_isaac_teleop/command` (`std_msgs/String`). Recording names: `save_episode`, `rerecord_episode`, `stop_recording` (aliases `save`, `rerecord`, `stop`).

Snap parts advance automatically when the harness snap gate fires. `open`
parts (gears, batteries) wait for `n`.

## Synthetic leader (no hardware)

`synthetic_leader.py` is a blocking TCP server. Do **not** run it in the same
terminal before collect — that is why Isaac never launched and you had to
Ctrl-C. Pass `--synthetic` and collect starts the sine-wave leader itself,
waits until port(s) are listening, then launches Isaac. With
`control.arms: "dual"`, collect starts **two** synthetic leaders on the
configured left/right ports:

```bash
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py \
  --synthetic --max-parts 1 --episode-time-s 20 --warmup-time-s 2
```

## Keyboard Cartesian teleop (no hardware)

Keyboard teleop runs **inside Isaac** (`ALOHA_KEYBOARD_TELEOP=1`) so keys work
while the viewport has focus. Hold keys for continuous motion (~0.12 m/s).

```bash
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py \
  --keyboard --max-parts 1 --episode-time-s 20 --warmup-time-s 2
```

After Isaac opens, click the viewport, wait for warmup, then hold:

| Key | Action |
| --- | --- |
| i / k | EE into / out of headcam view |
| j / l | EE left / right in headcam view |
| t / g | EE up / down |
| q / a | yaw (no-op when orientation is locked) |
| w / d | pitch (no-op when orientation is locked) |
| z / c | roll (no-op when orientation is locked) |
| f | gripper close |
| v | gripper open |

You should see `[aloha_teleop] kbd move held=...` in the Isaac log when keys
register. Task/recording keys are unchanged (`n`, `x`, arrows, Esc).
Keyboard always drives the **right** arm; with `control.arms: "dual"` the left
arm stays held (no second keyboard stream).

## Calibration

**Fixed right EE orientation:** Solo→Vega defaults to a **claw-machine**
mode: preferred world top-down via `retarget.fixed_orientation_wxyz: [0, 1, 0, 0]`
(same wxyz as scripted left `PART_DEFAULTS["ee_orientation"]`) plus
`retarget.orientation_cone_rad: 0.40` (~23°). Teleop tracks **XYZ + gripper**;
leader/keyboard wrist tilts are ignored. Lula tries the preferred quat first,
then in-cone free tilts, so translation is **not** blocked when exact
top-down is IK-infeasible. Dual-mode left clears both keys.

- Harden to legacy hard lock: `orientation_cone_rad: 0` (or omit / null).
- Full 6DoF wrist: `fixed_orientation_wxyz: null` (cone unused).

If the arm starts off top-down, retarget still **slews** toward the preferred
quat at `max_ang_vel`.
Alternatively, `retarget.fix_orientation: true` (default **false** in the
Solo→Vega YAML) holds the quaternion captured at clutch engage / reanchor
instead of a hardcoded world quat. Prefer `fixed_orientation_wxyz` when the
desired lock is a known world orientation (e.g. top-down grasp).

Leader deltas are mapped through `retarget.axes_map` in
[`config/aloha_solo_to_vega_1u.yaml`](../config/aloha_solo_to_vega_1u.yaml)
so motion matches the **head camera view** (into-image / image-left-right /
image-up), including the INIT `head_j1` pitch. When orientation is **not**
locked, wrist mapping uses the same map as a **space-fixed** conjugation
(tilt the leader in the image → DexMate tilts the same way in the image).
If the robot root is rotated in the stage, recompute that 3×3 (or fall back
to `axes_perm` / `axes_sign`). Keep rotation gains ≤ 1 until the map feels
natural.

**Distance units:** both the physical ALOHA leader EE (Interbotix FK) and the
DexMate stage use **meters**. `retarget.translation_gain` scales leader meter
deltas onto DexMate (`1.0` = 1:1). Default is **2.0** so a small leader hand
motion covers more of the task-board workspace; raise/lower that gain (and
keep `max_lin_vel` high enough that rate limiting does not cancel it) if
DexMate still feels short or overshoots. Workspace bounds and velocity limits
are the software safety layer; stale packets hold the last target at 100 ms and
pause tracking at 500 ms.

**Proximity fine control** (R-wrist laser `last_length`):

| Mode | YAML key | Effect |
| --- | --- | --- |
| Rate limit (default **off**) | `retarget.proximity_rate_limit` | Scales `max_lin_vel` / `max_ang_vel` / `max_lin_acc`. Absolute map unchanged → EEF **catches up**. |
| Delta gain (default **on** in this YAML) | `retarget.proximity_delta_gain` | Scales **per-frame** leader deltas into the DexMate target (translation + rotation). **No catch-up**, no snap when scale changes. Path-dependent; clutch/`r` resets. |

Both use a linear band: scale `1` at/above `depth_outer_m`, `scale_min` at/below `depth_inner_m`. Status logs include `prox_delta=` / `prox_rate=`. Old key `proximity_slowdown` is still accepted as an alias for `proximity_rate_limit`.

## Dataset contract

Export settings live in [`config/teleop_export.json`](../config/teleop_export.json)
(`--export-config` / `ALOHA_EXPORT_CONFIG`). Sample rate and mp4 fps are the same
value (`export.fps`, default **10**). With `export.playback_clock: wall` (default),
frames are wall-clock gated so **1 s of teleop ≈ 1 s of mp4 replay**, matching
what the operator saw. Use `sim` only when you want physics-time gating.

Arm targeting (`control` block):

| `control.arms` | Leaders | DexMate |
| --- | --- | --- |
| `right` (default) | one TCP stream (`control.leader_endpoint`, env `ALOHA_LEADER_ENDPOINT`) | right live; left held / action home |
| `dual` | two streams (`control.leader_endpoints.left` / `.right`; env `ALOHA_LEADER_ENDPOINT_LEFT` / `_RIGHT`) | both live |

| `control.gravity_compensation` | After gripper close / `s` |
| --- | --- |
| `false` (default) | torque-off backdrive only |
| `true` | enable ALOHA gravity compensation |

Challenge-compatible defaults:

- `observation.state` 44-D: left EE pose, right EE pose, 7+7 q, 7+7 qd, 2 gripper ratios
- `action` 14-D: left `xyz + rotvec + gripper` + right `xyz + rotvec + gripper`
  (with `arms=right`, left action slice is reset-time home; right is live)
- images 240×320 RGB: `observation.images.head|left_hand|right_hand`

Retarget / leader / keyboard speeds stay in
[`config/aloha_solo_to_vega_1u.yaml`](../config/aloha_solo_to_vega_1u.yaml)
(`paths.teleop_yaml`). Session timers (`episode_time_s`, `warmup_time_s`,
`num_episodes`) come from the export JSON (overridable by CLI / env).

Actions are the **post-clamp targets sent to IK**, not raw leader readings.
Human recordings are written to `runs/datasets/<repo>_<name>/<run_id>/` whenever
the operator saves (or the episode timer fires), including failed or partial
task attempts. Per-part success is stored in `episodes.jsonl` and `results.json`
for logging only. Quarantine is used only when a session saves **zero** episodes
(interrupt before the first save).

Inspect (metadata / optional matplotlib EE plot):

```bash
conda run -n lerobot python \
  /home/khw/RoCo_TaskBoardAssembly/tools/lerobot_recorder/inspect_dataset.py \
  /home/khw/RoCo_TaskBoardAssembly/runs/datasets/<repo>_<name>/<run_id>
```

Visualize (LeRobot-style Rerun default, or seekable Foxglove). Uses the conda
`lerobot` env with `lerobot[viz]`:

```bash
# Rerun desktop viewer (grouped 44-D state / 14-D action panels + 3 cameras)
conda run -n lerobot python \
  /home/khw/RoCo_TaskBoardAssembly/tools/lerobot_recorder/visualize_dataset.py \
  /home/khw/RoCo_TaskBoardAssembly/runs/datasets/local_roco_aloha_teleop/<run_id> \
  --episode-index 0

# latest run under runs/datasets/local_roco_aloha_teleop/
conda run -n lerobot python \
  /home/khw/RoCo_TaskBoardAssembly/tools/lerobot_recorder/visualize_dataset.py \
  --latest --episode-index 0

# Foxglove: connect the app to ws://127.0.0.1:8765
conda run -n lerobot python \
  /home/khw/RoCo_TaskBoardAssembly/tools/lerobot_recorder/visualize_dataset.py \
  --latest --episode-index 0 --display-mode foxglove
```

## Emergency and recovery

- `e` or a stale leader stream holds the last safe DexMate target.
- Ctrl-C in either terminal restores leader torque on shutdown and discards the
  unsaved episode buffer. Already saved episodes in this session are kept.
- Sessions with no saved episodes land in `runs/quarantine/<run_id>` with `stats.json`.
- Re-run `scripts/preflight.py` after a driver change.

## Tests that do not need hardware or Isaac

```bash
cd /home/khw/RoCo_TaskBoardAssembly
python3 -m pytest task/teleop/tests
```
