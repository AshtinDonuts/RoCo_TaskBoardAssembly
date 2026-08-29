Aug 29

Task A.
✅ Integrate lerobot huggingface to train and run ACT policy, integrating with existing endpoints. Reference the diffusion policy and pi05 setup.

Task B.
Find and use a suitable smoother, slower motion planner - use it with the scripted_baseline file. The purpose is a direct substitute for current motion planner in the scripted baseline, which produces extremely jerky motions.

Failed:
❌ RPMFlow - exotic motion planner -> inaccurate path planning, difficult (for codex) to debug.
- Simple Slowing down motion
- Simple accel / decel


Task C.
- modify scripted baseline to randomize object target offsets, while performing the tasks.
- view fairness update markdown ./Fairness_update.md to comprehend the translation randomizations
- rephrase to me what to offset to align our understanding of the eval randomizations

Task D.
- Train ACT with Chunk-size 5 on a single subtask
- Then train on all subtasks.
