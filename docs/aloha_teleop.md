# ALOHA Solo to DexMate Vega 1U teleoperation

This pipeline uses the physical ALOHA Solo **leader** as a Cartesian input device
for the RoCo Industrial Task Board scene in Isaac Sim 5.1. The DexMate
`vega_1u_gripper` left arm is driven through the challenge Lula IK. The physical
follower is never launched. Demonstrations are written as LeRobot v3 episodes
with the challenge 44-D state / 14-D action contract.

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

Close the leader gripper (or press `s`) after the arm reaches the start pose.
Gravity compensation then enables backdriving.

Terminal 2, Isaac + recorder:

```bash
export PATH="$HOME/.local/bin:$PATH"
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py
```

### Operator keys (leader terminal)

| Key | Action |
| --- | --- |
| space | clutch toggle (captures origins on engage) |
| r | recenter / recapture origins |
| p | pause (hold last DexMate target) |
| u | resume |
| n | mark current gravity-settled (`open`) part done |
| x | abort remaining parts |
| e | emergency hold (deadman off) |
| s | start |

Snap parts advance automatically when the harness snap gate fires. `open`
parts (gears, batteries) wait for `n`.

ROS command topic: `/aloha_isaac_teleop/command` (`std_msgs/String`).

## Synthetic leader (no hardware)

```bash
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/synthetic_leader.py
python3 /home/khw/RoCo_TaskBoardAssembly/scripts/collect_aloha_episode.py \
  --synthetic --max-parts 1 --max-sim-seconds 30
```

## Calibration

After empty-space teleop, edit
[`config/aloha_solo_to_vega_1u.yaml`](../config/aloha_solo_to_vega_1u.yaml)
`retarget.axes_perm` and `axes_sign` so that pushing the leader forward/right/up
moves DexMate the same way in Isaac. Keep translation/rotation gains ≤ 1 until
the map feels natural. Workspace bounds and velocity limits are the software
safety layer; stale packets hold the last target at 100 ms and pause tracking
at 500 ms.

## Dataset contract

10 Hz rows, challenge-compatible:

- `observation.state` 44-D: left EE pose, right EE pose, 7+7 q, 7+7 qd, 2 gripper ratios
- `action` 14-D: left `xyz + rotvec + gripper` + right home `xyz + rotvec + gripper`
- images 240×320 RGB: `observation.images.head|left_hand|right_hand`

Actions are the **post-clamp targets sent to IK**, not raw leader readings.
Successful 9/9 episodes are moved to `runs/datasets/`. Failed or interrupted
episodes go to `runs/quarantine/`.

Inspect:

```bash
conda run -n lerobot python \
  /home/khw/RoCo_TaskBoardAssembly/tools/lerobot_recorder/inspect_dataset.py \
  /home/khw/RoCo_TaskBoardAssembly/runs/datasets/<repo>
```

## Emergency and recovery

- `e` or a stale leader stream holds the last safe DexMate target.
- Ctrl-C in either terminal restores leader torque on shutdown.
- Interrupted recordings land in `runs/quarantine/<run_id>` with `stats.json`.
- Re-run `scripts/preflight.py` after a driver change.

## Tests that do not need hardware or Isaac

```bash
cd /home/khw/RoCo_TaskBoardAssembly
python3 -m pytest task/teleop/tests
```
