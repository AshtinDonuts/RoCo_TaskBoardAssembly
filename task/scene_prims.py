# Static USD prim paths for SingleRigidPrim observation wrappers.
#
# These are passed to PickPlaceTask_scene_bimanual at scene-setup time and
# wrapped in SingleRigidPrim / PhysX tensor views for the duration of the
# run.  They MUST point at prims that are never joint-locked by snap_attach:
# pointing at a snappable part (e.g. rod_16mm, bolt_8mm) was the original
# bug — snap_attach adds a FixedJoint whose child is the movable part, which
# invalidates the pre-existing tensor view for that prim mid-snap and causes
# a PhysX crash.
#
# The task board color mesh is a valid rigid body in scene_init.usd (it
# appears as `parent_body_path` in snap configs, meaning it is the FixedJoint
# ANCHOR, never the joint-locked movable body).  Its tensor view stays valid
# across the entire pick-and-place sequence.
#
# The observation outputs (get_params()) are not consumed by the harness; only
# the Robot and Gripper wrappers matter at runtime.  Using a shared prim for
# both arms is intentional — the wrapper names (object_L / object_R) are kept
# unique by find_unique_string_name regardless.

TASK_BOARD_PRIM = "/World/task_board/task_board_color"

# Observation prim for the left-arm SingleRigidPrim wrapper.
L_OBSERVATION_PRIM: str = TASK_BOARD_PRIM

# Observation prim for the right-arm SingleRigidPrim wrapper.
R_OBSERVATION_PRIM: str = TASK_BOARD_PRIM
