import habitat_sim
import magnum as mn
import numpy as np
import cv2
from datetime import datetime as dt

# --- 1. Create sim with an empty/minimal stage ---
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "NONE"  # empty scene
backend_cfg.enable_physics = True


camera_spec = habitat_sim.CameraSensorSpec()
camera_spec.uuid = "overhead_rgb"
camera_spec.sensor_type = habitat_sim.SensorType.COLOR
camera_spec.resolution = [720, 1280]
camera_spec.position = mn.Vector3(0, 8.0, 0) # aka high above
camera_spec.orientation = mn.Vector3(-90, 0, 0) #look straight down

# Attach camera sensors to agents
agent_cfgs = []
for _ in range(2):
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [camera_spec]
    agent_cfgs.append(agent_cfg)

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


# --- 4. Recompute NavMesh so agents respect the walls ---
navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()
navmesh_settings.agent_height = 1.5
navmesh_settings.agent_radius = 0.2
navmesh_settings.agent_max_climb = 0.05
navmesh_settings.agent_max_slope = 45.0
navmesh_settings.include_static_objects = True
navmesh_settings.cell_height = 0.05
navmesh_settings.cell_size = 0.05

succes = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

print("NavMesh recompute success:", succes)
print("Navigable area:", sim.pathfinder.navigable_area)


# Verification between two points
p1 = sim.pathfinder.get_random_navigable_point()
p2 = sim.pathfinder.get_random_navigable_point()
path = habitat_sim.ShortestPath()
path.requested_start = p1
path.requested_end = p2
sim.pathfinder.find_path(path)
print("Geodesic distance:", path.geodesic_distance)
print("Test path points:", len(path.points))

# --- 5. Place your two agents ---
for i, agent in enumerate(sim.agents):
    state = agent.get_state()
    state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(state)


# ----- Video writer -----
fps = 30
now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"scene_gen_attempt_{now}.mp4"
out = cv2.VideoWriter(
    name,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (1280,720)
)

# --- Setup agent goals before the render loop ---
def get_agent_goal(sim):
    return sim.pathfinder.get_random_navigable_point()

agent_goals = [get_agent_goal(sim) for _ in sim.agents]
agent_paths = [habitat_sim.ShortestPath() for _ in sim.agents]

def step_agents(sim, agent_goals, agent_paths, step_size=0.05):
    for i, agent in enumerate(sim.agents):
        state = agent.get_state()
        current_pos = state.position

        # Check if agent reached goal, pick a new one
        dist = np.linalg.norm(
            np.array([current_pos.x, current_pos.y, current_pos.z]) -
            np.array([agent_goals[i].x, agent_goals[i].y, agent_goals[i].z])
        )
        if dist < 0.3:
            agent_goals[i] = get_agent_goal(sim)

        # Compute shortest path to goal
        agent_paths[i].requested_start = current_pos
        agent_paths[i].requested_end = agent_goals[i]
        sim.pathfinder.find_path(agent_paths[i])

        if agent_paths[i].geodesic_distance == float("inf"):
            # No path found, pick new goal
            agent_goals[i] = get_agent_goal(sim)
            continue

        # Move one step along the path
        points = agent_paths[i].points
        if len(points) >= 2:
            next_waypoint = points[1]
            direction = np.array([
                next_waypoint.x - current_pos.x,
                0,  # keep y stable
                next_waypoint.z - current_pos.z
            ])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * min(step_size, norm)

            new_pos = mn.Vector3(
                current_pos.x + direction[0],
                current_pos.y,
                current_pos.z + direction[2]
            )

            snapped = sim.pathfinder.snap_point(new_pos)
            if sim.pathfinder.is_navigable(snapped):
                state.position = snapped
            else:
                state.position = new_pos

            # Face the direction of movement
            angle = np.arctan2(direction[0], direction[2])
            state.rotation = mn.Quaternion.rotation(
                mn.Rad(angle), mn.Vector3(0, 1, 0)
            )
            agent.set_state(state)

sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

# Test agent movement BEFORE the render loop
agent = sim.agents[0]
state = agent.get_state()
print("Initial position:", state.position)

# Manually move the agent
state.position = mn.Vector3(1.0, 0.0, 1.0)
agent.set_state(state)

state2 = agent.get_state()
print("Position after set_state:", state2.position)

# Test pathfinder
p1 = sim.pathfinder.get_random_navigable_point()
p2 = sim.pathfinder.get_random_navigable_point()
print("Random point 1:", p1)
print("Random point 2:", p2)

path = habitat_sim.ShortestPath()
path.requested_start = p1
path.requested_end = p2
sim.pathfinder.find_path(path)
print("Geodesic distance:", path.geodesic_distance)
print("Path points:", path.points)

# Test Geodesic desitance (keeps being inf...)
island1 = sim.pathfinder.get_island(p1)
island2 = sim.pathfinder.get_island(p2)
print("Island 1:", island1)
print("Island 2:", island2)

# --- 6. Render loop ---
num_frames = fps * 10  # 10 seconds of video

for frame in range(num_frames):
    # Step the simulation
    sim.step_physics(1.0 / fps)

    # Get observation from agent 0's camera
    obs = sim.get_sensor_observations(agent_ids=0)
    rgb = obs["overhead_rgb"]

    # habitat_sim returns RGBA, OpenCV needs BGR
    bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)

    out.write(bgr)

out.release()
print("Video saved!!!!!")