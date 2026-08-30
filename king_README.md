Aug 29

Task A.
✅ Integrate lerobot huggingface to train and run ACT policy, integrating with existing endpoints. Reference the diffusion policy and pi05 setup.

Task B.
Find and use a suitable smoother, slower motion planner - use it with the scripted_baseline file. The purpose is a direct substitute for current motion planner in the scripted baseline, which produces extremely jerky motions.

❌ RPMFlow - more of an exotic motion planner -> poor path planning, difficult (for codex) to debug.
✅ Cartesian interpolation between existing waypoints
- Simple accel / decel

Tune params here: [param config](task/param_config.py) - CARTESIAN_MAX_EE_SPEED_M_S


Task C.
- randomize object target offsets based on the Fairness Update rules.
- view fairness update markdown ./Fairness_update.md to comprehend the translation randomizations
- rephrase to me what to offset to align our understanding of the eval randomizations

Task C2.
- modify scripted baseline to randomize object target offsets, while using the same privileged translation informtaion on the scripted baseline.
- The rationale is for rollout data generation for downstream behavior cloning.
- Render one randomized rollout without the modification.
- Render one randomized rollout with the modification.


Task D.
- Train ACT with Chunk-size 5 on a single subtask
- Then train on all subtasks.
