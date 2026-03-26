import habitat_sim
import numpy as np
import matplotlib.pyplot as plt


# Point to the scene dataset config bundled with hab3_bench_assets
scene_dataset_config = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"

# Pick your scene ID (only use scene name, not the full path)
scene_id = "103997919_171031233" 

# Build simulator config
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = scene_id
backend_cfg.scene_dataset_config_file = scene_dataset_config
backend_cfg.enable_physics = True

# Add an agent
agent_cfg = habitat_sim.agent.AgentConfiguration()

cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

print("NavMesh bounds:", sim.pathfinder.get_bounds())
#print("Navigable area:", sim.pathfinder.navigable_area)


#========================= Top down map action ==========================
# Get navigable area and bounds
# print("NavMesh area:", sim.pathfinder.navigable_area)
# print("Bounds:", sim.pathfinder.get_bounds())

# Generate a top-down view at floor height
height = sim.pathfinder.get_bounds()[0].y # min y = floor level
topdown_map = sim.pathfinder.get_topdown_view(
    meters_per_pixel=0.05,
    height=height
    )
plt.imshow(topdown_map)
plt.savefig("scene_topdown.png")


#========================= Sample random navigable points ==========================
# Sample random valid points
# point_a = sim.pathfinder.get_random_navigable_point()
# point_b = sim.pathfinder.get_random_navigable_point()

# # Or check if your own coordinates are valid
# my_point = np.array([1.5, 0.1, -2.3]) #[x, y, z]
# is_valid = sim.pathfinder.is_navigable(my_point)

# # Snap an arbitrary point to the nearest navigable location
# snapped = sim.pathfinder.snap_point(my_point)


# #========================= Determine the shortest path ==========================
# path = habitat_sim.ShortestPath()
# path.requested_start = point_a
# path.requested_end = point_b

# found = sim.pathfinder.find_path(path)
# if found:
#     print("Path waypoints:", path.points)
#     print("Geodesic distance:", path.geodesic_distance)