Plan
Asset-Centroid Scripted Policy V1
Summary
Implement a submission-only scripted policy for gear_20teeth, inspired by ACT/ALOHA’s reset-time waypoint generation and per-step interpolation, but using quintic easing for smoother motion. The policy will use privileged live USD geometry to localize the asset and will not modify the host baseline, runner, or policy API. **V1 must drive the right gripper** (`env_info.R_controller`, `active_arms=("R",)`); the left arm stays held by the harness. ACT scripted policy reference

Implementation Changes
Add policies.asset_centroid_scripted.AssetCentroidScriptedPolicy plus a pure-math trajectory helper.
Require `env_info.R_controller` and set `active_arms = ("R",)` so the runner applies right-arm DOFs from the policy action and holds the left arm.
On reset():
Skip immediately when release_mode == "snap" or the part is not V1-supported.
Resolve /World/parts/gear_20teeth and its mesh descendants from part_init_poses.json.
Transform all mesh points into the asset-root frame, calculate the local AABB midpoint, then transform that centroid through the asset’s live 6D world pose.
Fail safely with a clear diagnostic and no arm movement if the prim, mesh geometry, centroid, or IK plan cannot be resolved.
Configure gear-20 manually:
Axial direction: asset-local +Y.
Centroid-to-grasp offset: [0, -0.005, 0] metres, matching the existing 5 mm lowered grasp.
Tool-frame grasp-center offset: [0, -0.0144, 0.1972] metres (R fingertip midpoint in `R_ee_link_gripper_link` from gripper meshes + URDF).
Gripper open/close: default ``gripper.mode=compliant`` uses soft GripperCompliance stall while treating the hand-tuned part close aperture as a hard lower bound. Configure its slew with ``gripper.close_speed_rad_s``. Set ``gripper.mode=aperture`` for numeric open/close commands. Do not call the geometric aperture resolver or read part_local_aabb_extents.json for gripper commands.
Keep tool +Z exactly world-down. Generate top-down yaw candidates in 15° increments and choose the candidate with feasible, continuous IK across the pick/place keyframes and the lowest joint-motion cost. Do not fall back to a tilt cone.
Generate the canonical path: current pose → hover pick → descend → compliant close → lift → transfer hover → descend place → settle → compliant open → retract.
Preserve target.place_pos as the scripted release/grasp-center target.
Use 50 mm pick/place hover clearance and 100 mm final retract.
Use quintic interpolation with zero endpoint velocity, a 0.05 m/s peak Cartesian speed cap, 20°/s initial orientation-alignment cap, and 0.5-second minimum move duration.
Close for at least 1.5 seconds, settle before release for 0.5 seconds, and hold open for 0.5 seconds.
Freeze trajectory progress during transient IK failure; abort safely after 100 consecutive failed steps. Never continue into close or release after an unreached critical waypoint.
Interfaces and Part Handling
No changes to Policy, EnvInfo, Observation, PartTarget, the runner, or baseline_scripted.py.
The new dotted policy entrypoint is the only public addition.
Verified release classification:
Pick/drop: gear_20teeth, gear_60teeth, battery_size1, battery_size5.
Snap and skipped: rod_16mm, bolt_8mm, usb_a, hdmi, pin.
V1 supports only gear_20teeth; unsupported targets return a no-op action and report done.
Leave the repository’s current part_order untouched, despite its gear-60/gear-20 order differing from the supplied order. Run V1 with ROCO_PART_ORDER=gear_20teeth.
Test Plan
Unit-test local-AABB centroid calculation and asset-local-to-world transformation with translated and rotated synthetic assets.
Verify every yaw candidate keeps the TCP approach axis exactly world-down.
Test quintic endpoint position/velocity, speed limits, duration calculation, and deterministic sampling at EnvInfo.physics_dt.
Verify the manual 0.12/0.065 rad apertures are used and the geometry-derived aperture resolver is never called.
Test safe skipping of snap and unsupported parts, missing geometry, no feasible yaw, and repeated IK failure.
Assert the policy requires R_controller and advertises active_arms=("R",).
Run an Isaac Sim smoke test with:
ROCO_PART_ORDER=gear_20teeth
the new policy entrypoint
one-part results JSON and optional video
Acceptance requires a completed smooth rollout on the **right** gripper, no snap behavior, successful compliant grasp/release, and a gear-20 grading pass within the existing 10 mm tolerance.

## Tunables
Edit [`config/asset_centroid_policy.json`](config/asset_centroid_policy.json) (or set `ROCO_ASSET_CENTROID_CONFIG` to another JSON path). Speed lives under `motion.max_linear_speed_m_s` / `motion.max_angular_speed_deg_s` / `motion.minimum_move_s`. Gripper: `gripper.mode` is `compliant` (soft stall close) or `aperture` (numeric `parts.*.gripper_*_rad`). Path clearances, dwell times, IK guards, and per-part grasp/TCP values are in the same file.
