# RoCo IROS 2026 policy submission

## Evaluator entry point

```text
policies.camera_offset_scripted.CameraOffsetScriptedPolicy
```

The evaluator should instantiate this class with the public `EnvInfo` object
and drive it through `reset(obs, target)`, `act(obs)`, and `is_done(obs)`.

The policy requires camera observations. Enable them before starting Isaac Sim:

```bash
export TASK_ENABLE_CAMERA_OUTPUT=1
```

Example launch using the bundled public evaluator:

```bash
uv run python task/run_pick_place.py \
  --policy policies.camera_offset_scripted.CameraOffsetScriptedPolicy
```

For an existing Isaac Sim installation:

```bash
TASK_ENABLE_CAMERA_OUTPUT=1 \
"${ISAAC_SIM}/python.sh" task/run_pick_place.py \
  --policy policies.camera_offset_scripted.CameraOffsetScriptedPolicy
```

The organizers may replace `task/run_pick_place.py` with their blind-seed
evaluator. The replacement must retain the published `task/policy_api.py`
contract and expose the head RGB camera through `Observation.rgb["head"]`.
Depth and intrinsics should be exposed through the matching `Observation`
dictionaries when available.

## Runtime contents

The archive deliberately contains the policy's complete runtime dependency
chain:

- the policy, camera-offset estimator, and packaged nominal camera reference;
- the scripted motion policy and joint-rate limiter that execute corrected
  private `PartTarget` copies;
- the controller stack and robot configuration used by that scripted policy;
- the task scene, robot, part, and table assets;
- the public policy API, task configuration, snap helper, and reference
  evaluator;
- the pinned Python/Isaac Sim environment metadata.

Training code, datasets, checkpoints, generated rollouts, tests, debug tools,
the local virtual environment, and Git metadata are excluded.

## Runtime dependencies

- Linux x86-64
- Python 3.11
- NVIDIA Isaac Sim 5.1.0
- NumPy 1.26.x (`numpy < 2`)
- OpenCV when available (Isaac Sim's environment provides it); deterministic
  NumPy fallbacks are implemented for the image-registration operations

No model server, network access, dataset, or external checkpoint is required.
All camera reference files used by the policy are inside
`task/policies/camera_reference/`.

## Important evaluation constraints

- Do not enable `--privileged-xy-randomization`.
- Do not set `CAMERA_OFFSET_EXPORT_DIR` during scoring; it is a development-only
  diagnostic output.
- Keep the archive directory structure intact because USD and policy imports
  use repository-relative paths.
- The policy class is constructed once per evaluator process. Independent
  blind-seed trials should preferably use independent evaluator processes.

## Archive integrity

`MANIFEST.sha256` is generated at packaging time and covers every submitted
file except the manifest itself. `SUBMISSION_COMMIT.txt` records the source
commit and branch. Build the archive with:

```bash
scripts/build_submission_zip.sh
```
