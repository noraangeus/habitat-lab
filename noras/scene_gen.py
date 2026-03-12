import habitat_sim
import magnum as mn
import numpy as np
import cv2

# --- 1. Create sim with an empty/minimal stage ---
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "NONE"  # empty scene
backend_cfg.enable_physics = True


# camera_spec = habitat_sim.CameraSensorSpec()
# camera_spec.uuid = "overhead_rgb"
# camera_spec.sensor_type = habitat_sim.SensorType.COLOR
# camera_spec.resolution [720, 1280]
# camera_spec.position = mn.Vector3(0, 8.0, 0) # aka high above
# camera_spec.orientation = mn.Vector3(-90, 0, 0) #look straight down


agent_cfgs = [habitat_sim.agent.AgentConfiguration() for _ in range(2)]  # 2 agents
# agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, agent_cfgs)
sim = habitat_sim.Simulator(cfg)

rigid_mgr = sim.get_rigid_object_manager()
obj_template_mgr = sim.get_object_template_manager()

# --- 2. Add a floor (flat cube, scaled very thin) ---
floor_template = obj_template_mgr.get_template_by_handle(
    obj_template_mgr.get_template_handles("cubeSolid")[0]
)
floor_template.scale = mn.Vector3(10.0, 0.05, 10.0)
obj_template_mgr.register_template(floor_template, "floor")
floor = rigid_mgr.add_object_by_template_handle("floor")
floor.translation = mn.Vector3(0, -0.025, 0)
floor.motion_type = habitat_sim.physics.MotionType.STATIC

# --- 3. Add walls ---
def add_wall(pos, scale):
    t = obj_template_mgr.get_template_by_handle(
        obj_template_mgr.get_template_handles("cubeSolid")[0]
    )
    t.scale = mn.Vector3(*scale)
    obj_template_mgr.register_template(t, f"wall_{pos}")
    w = rigid_mgr.add_object_by_template_handle(f"wall_{pos}")
    w.translation = mn.Vector3(*pos)
    w.motion_type = habitat_sim.physics.MotionType.STATIC

add_wall([0, 1.5, -5],  [10, 3, 0.2])  # back wall
add_wall([0, 1.5,  5],  [10, 3, 0.2])  # front wall
add_wall([-5, 1.5, 0],  [0.2, 3, 10])  # left wall
add_wall([ 5, 1.5, 0],  [0.2, 3, 10])  # right wall
# interior divider wall with a gap (doorway):
add_wall([2, 1.5, 0],   [6, 3, 0.2])   # partial divider

# --- 4. Recompute NavMesh so agents respect the walls ---
navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.agent_height = 1.5
navmesh_settings.agent_radius = 0.2
navmesh_settings.agent_max_climb = 0.2
navmesh_settings.agent_max_slope = 45.0
navmesh_settings.include_static_objects = True

sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

print("Navigable area:", sim.pathfinder.navigable_area)

# --- 5. Place your two agents ---
for i, agent in enumerate(sim.agents):
    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)


# ----- Video writer -----
# fps = 30
# out = cv2.VideoWriter(
#     "scene_gen_attempt.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (1280,720)
# )