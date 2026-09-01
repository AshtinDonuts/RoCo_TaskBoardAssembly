# Fairness randomization and deterministic camera-only policy plan

## Official evaluation randomization

Official simulation leaderboard evaluation applies deterministic, per-trial
XY translation randomization so policies cannot overfit to the released
initial state.

- The allowed offset on each world-frame axis is **[-0.01 m, +0.01 m]**
  (**[-1 cm, +1 cm]**). X and Y are sampled independently from a uniform
  distribution. Z position and orientation are not randomized.
- One XY vector is sampled for the task board. Placement targets, snap
  targets, snap connection anchors, and grading targets move by that same
  board vector.
- Each evaluated movable part receives an independent XY vector, except
  `gear_60teeth`, `rod_16mm`, and `bolt_8mm`. Those support-coupled parts
  receive the board vector so their initial support/contact relationship is
  preserved.
- A deterministic evaluator seed fixes the complete trial. Part names are
  sorted before sampling, so incidental changes to iteration order do not
  change a seed's layout.
- Physics parameters, success criteria, scoring tolerances, and the evaluation
  horizon are unchanged.

The randomized scene is privileged evaluator state. In a competition-faithful
run, the policy receives the released nominal `PartTarget` configuration and
camera observations only. It does not receive the seed, sampled board offset,
sampled part offsets, shifted targets, object transforms, or hole transforms.
The evaluator may write the true offsets to results metadata after a local
development run so estimator error can be measured offline.

`--random-seed N` now uses this camera-only behavior by default. The
`--privileged-xy-randomization` and `--log-randomization-offsets` options are
development-only diagnostics and must not be used for submission evaluation.

## Deterministic policy design

The submission policy should be a camera-aware wrapper around the existing
scripted baseline. It should change only the XY coordinates used to construct
the baseline's paths; grasp heights, orientations, gripper values, phase
timings, snap search, IK, and grading remain unchanged.

### 1. Build nominal reference assets offline

Use the released base scene and `task/param_config.py` to capture a nominal
head RGB-D frame before robot motion. Store only submission-legal, nominal
assets:

- the nominal head RGB/depth image and camera intrinsics;
- a fixed camera-to-world calibration or board-plane pixel-to-world mapping;
- a board registration mask that excludes movable-part regions;
- one part template/mask and nominal search centre per subtask;
- optional target-hole templates for board registration validation.

The head pose, focal length, resolution, lighting, and render settings used to
create these assets must match evaluation. Do not package randomized ground
truth, seed-to-offset tables, scene prim paths, or simulator transform readers
with the policy.

The revised `scripts/generate_randomization_final_frames.sh` captures the
nominal reference by default, then captures randomized camera-only frames. Its
post-run JSON files contain evaluator ground truth for local accuracy reports,
not as policy input.

### 2. Estimate the board offset before any part moves

On the first `Policy.reset`, buffer several head RGB-D frames and calculate one
episode-level board translation:

1. Register the randomized board ROI against the nominal board reference.
   Use deterministic masked ECC/phase correlation or fixed feature detection
   plus RANSAC with fixed parameters. Constrain the model to board-plane
   translation; rotation and scale are not randomized.
2. Reject foreground pixels using the nominal movable-part masks and depth
   disagreement. This prevents a shifted loose part from pulling the board
   estimate.
3. Convert the registered pixel displacement to world XY using the stored
   calibration. A local board-plane homography is sufficient for the ±1 cm
   domain; RGB-D back-projection is an alternative.
4. Take a deterministic median over the buffered frames, clamp only tiny
   numerical overshoot to [-1 cm, +1 cm], and record a confidence score.

Estimate this once at the untouched initial layout. Later subtasks may obscure
holes or change the appearance of the board as parts are installed.

### 3. Estimate every independent part offset from the initial view

For each part that does not share the board offset, search a bounded ROI around
its nominal image location. The ROI must cover the projection of the full
[-1 cm, +1 cm] XY range plus template extent and a small calibration margin.

Use deterministic multi-scale normalized cross-correlation on RGB/edges,
checked against the depth template. Refine the best integer-pixel match with a
fixed sub-pixel quadratic fit, then convert its displacement to world XY using
the stored local Jacobian/homography. Resolve ambiguous matches with a fixed
ordering of scores: template score, depth residual, distance from nominal,
then lexicographic pixel coordinate. Buffering and taking the median across a
few initial frames reduces render noise without introducing randomness.

For `gear_60teeth`, `rod_16mm`, and `bolt_8mm`, do not run an independent part
search: assign the estimated board offset directly, matching the evaluator's
support-coupling rule.

### 4. Adjust a private copy of each nominal target

At the start of subtask `p`, construct an adjusted `PartTarget` without
mutating the harness object:

```text
adjusted_pick_xy   = nominal_pick_xy   + estimated_part_offset[p]
adjusted_place_xy  = nominal_place_xy  + estimated_board_offset
adjusted_grade_xy  = nominal_grade_xy  + estimated_board_offset
adjusted_snap_xy   = nominal_snap_xy   + estimated_board_offset
```

Apply the board offset consistently to both `snap.target_pos` and
`snap.connect_pos` in `target.extra`. Pass this adjusted target to the existing
baseline path builder. This preserves its tested motion and changes only the
two translations that randomization affects.

If an estimate fails validation, use a deterministic fallback: retry once
with the next fixed matcher configuration, then use the median of valid
neighbouring/template estimates or the nominal zero offset. Never read the
evaluator's shifted runtime config.

### 5. Implementation structure

Add a submission policy such as
`task/policies/camera_offset_scripted.py` with three independently testable
pieces:

1. `ReferenceBundle` loads and validates nominal images, masks, calibration,
   expected resolution, and a version/hash.
2. `OffsetEstimator` consumes only `Observation.rgb["head"]`,
   `Observation.depth["head"]`, and `Observation.intrinsics["head"]`, returning
   the board offset, per-part offsets, confidence, and diagnostics.
3. `CameraOffsetScriptedPolicy` adjusts a deep copy of the nominal
   `PartTarget`, then delegates trajectory execution to `BaselinePolicy`.

Set `TASK_ENABLE_CAMERA_OUTPUT=1` for capture and evaluation. The policy should
fail clearly if the head stream, calibration, or expected image shape is
missing rather than silently behaving as though privileged waypoints exist.

### 6. Verification and submission gates

- Unit-test pixel/world conversion, range handling, support coupling, target
  copying, snap-field adjustment, deterministic tie-breaking, and missing-frame
  behavior without Isaac Sim.
- Replay the nominal scene: estimated offsets should be near zero and the
  adjusted path should reproduce the existing baseline.
- Run a held-out set of at least 100 evaluator seeds. Compare estimates with
  the post-run JSON only after each run. Report board and per-part XY MAE,
  95th-percentile error, worst-case error, confidence failures, and task score.
- Include deliberate near-boundary cases on both axes. Random seeds alone are
  not guaranteed to exercise all four corners of the allowed square.
- Target sub-millimetre-to-1 mm XY error for the connector parts because their
  snap tolerances are as tight as 2 mm. Validate the full assembly score, not
  only image registration error.
- Audit the submission policy for imports or calls that expose randomization
  seeds, shifted configs, USD prim transforms, result metadata, or simulator
  object poses. Re-run identical recorded frames twice and require identical
  offsets and action sequences.

## Implementation status

### Done

1. Nominal reference bundle under `task/policies/camera_reference/` plus
   `scripts/build_camera_reference.py`.
2. `ReferenceBundle` / `OffsetEstimator` / `CameraOffsetScriptedPolicy` with
   ECC+ORB board consensus, multi-scale NCC (early-exit), support coupling,
   and failed-estimate → nominal-zero fallback.
3. Camera-only harness default (`--random-seed`); privileged flags are
   development-only.
4. Unit tests (`task/tests/test_camera_offset_policy.py`) and privilege audit
   (`task/tests/test_camera_offset_privilege_audit.py`).
5. Offline gates pass via `scripts/evaluate_camera_offset_gates.py`:
   nominal near-zero, 100 synthetic held-out warps, ±1 cm corners,
   identical-frame determinism. Accuracy reporter:
   `scripts/report_camera_offset_accuracy.py`.

### Remaining (Isaac Sim / leaderboard)

- Capture ≥100 **real** evaluator seeds:
  `scripts/generate_randomization_final_frames.sh --count 100`
- Full assembly score under
  `TASK_ENABLE_CAMERA_OUTPUT=1 ... --policy policies.camera_offset_scripted.CameraOffsetScriptedPolicy --random-seed N`
- Drive connector (usb/hdmi/pin) **real-render** XY error to ≤1 mm (current
  ~10-seed sample: connectors MAE ~1.7 mm; synthetic warp gates already
  pass the 1 mm board/connector MAE target when the template matches).
