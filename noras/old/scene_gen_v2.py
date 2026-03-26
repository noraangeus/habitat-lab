"""
habitat_scene_video.py
======================
Builds a custom scene entirely from Python primitives (no Blender / .glb files),
navigates two agents between waypoints, and writes an overhead RGB video.

Requirements:
    conda install habitat-sim headless -c conda-forge -c aihabitat
    pip install opencv-python numpy magnum

Usage:
    python habitat_scene_video.py
    → writes  scene_walkthrough.mp4  in the current directory
"""

import numpy as np
import cv2
import magnum as mn
import habitat_sim
import habitat_sim.physics


# ─────────────────────────────────────────────
# 1.  SIMULATOR CONFIGURATION
# ─────────────────────────────────────────────

def make_cfg(scene_filepath="NONE"):
    # Backend / physics
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_filepath   # "NONE" = empty world
    backend_cfg.enable_physics = True
    backend_cfg.use_semantic_textures = False

    # Overhead spectator camera (attached to agent-0, but we'll lock its
    # pose so it never moves)
    camera_spec = habitat_sim.CameraSensorSpec()
    camera_spec.uuid          = "overhead_rgb"
    camera_spec.sensor_type   = habitat_sim.SensorType.COLOR
    camera_spec.resolution    = [720, 1280]   # H × W
    camera_spec.position      = mn.Vector3(0, 0, 0)   # offset from agent root
    camera_spec.orientation   = mn.Vector3(0, 0, 0)

    # Agent-0 carries the camera; agent-1 is camera-free
    agent0_cfg = habitat_sim.agent.AgentConfiguration()
    agent0_cfg.sensor_specifications = [camera_spec]
    agent0_cfg.height  = 1.5
    agent0_cfg.radius  = 0.2

    agent1_cfg = habitat_sim.agent.AgentConfiguration()
    agent1_cfg.sensor_specifications = []
    agent1_cfg.height  = 1.5
    agent1_cfg.radius  = 0.2

    return habitat_sim.Configuration(backend_cfg, [agent0_cfg, agent1_cfg])


# ─────────────────────────────────────────────
# 2.  BUILD THE SCENE FROM PRIMITIVES
# ─────────────────────────────────────────────

def add_box(sim, pos, scale, name):
    """Add a STATIC box primitive (floor, wall, furniture …)."""
    mgr = sim.get_object_template_manager()
    handle = mgr.get_template_handles("cubeSolid")[0]
    tmpl = mgr.get_template_by_handle(handle)
    tmpl.scale = mn.Vector3(*scale)
    mgr.register_template(tmpl, name)

    rigid_mgr = sim.get_rigid_object_manager()
    obj = rigid_mgr.add_object_by_template_handle(name)
    obj.translation  = mn.Vector3(*pos)
    obj.motion_type  = habitat_sim.physics.MotionType.STATIC
    return obj


def build_scene(sim):
    """
    Simple two-room apartment:

        ┌──────────┬──────────┐
        │          │  room B  │
        │  room A  ├────  ────┤  ← doorway gap at Z = -5 to -3
        │          │  room B  │
        └──────────┴──────────┘

    World units: metres.  Y = 0 is the floor.
    X:  -10 … 0
    Z:  -10 … 0
    """
    W = 0.2   # wall thickness

    # Floor  (10 m × 10 m, centred on (-5, 0, -5))
    add_box(sim, pos=(-5, -0.05, -5), scale=(10, 0.1, 10), name="floor")

    # Outer walls
    add_box(sim, pos=(-5, 1.5,   0), scale=(10, 3, W),  name="wall_front")
    add_box(sim, pos=(-5, 1.5, -10), scale=(10, 3, W),  name="wall_back")
    add_box(sim, pos=( 0, 1.5,  -5), scale=(W,  3, 10), name="wall_right")
    add_box(sim, pos=(-10,1.5,  -5), scale=(W,  3, 10), name="wall_left")

    # Interior dividing wall with a doorway gap (Z = -6 … -4)
    add_box(sim, pos=(-5, 1.5, -1.0), scale=(10, 3, W), name="divider_north")  # Z: 0  → -3
    add_box(sim, pos=(-5, 1.5, -8.0), scale=(10, 3, W), name="divider_south")  # Z: -7 → -10
    # The gap between Z=-3 and Z=-7 is the doorway — no wall segment there.

    # Some furniture (static boxes acting as obstacles)
    # add_box(sim, pos=(-2,  0.4, -2),  scale=(1.2, 0.8, 0.6), name="sofa")
    # add_box(sim, pos=(-8,  0.4, -2),  scale=(0.6, 0.8, 0.6), name="armchair")
    # add_box(sim, pos=(-8,  0.4, -8),  scale=(1.8, 0.6, 0.9), name="bed")
    add_box(sim, pos=(-2,  0.4, -8),  scale=(0.8, 0.8, 0.8), name="desk")


def recompute_navmesh(sim):
    settings = habitat_sim.NavMeshSettings()
    settings.agent_height    = 1.5
    settings.agent_radius    = 0.2
    settings.agent_max_climb = 0.2
    settings.agent_max_slope = 45.0
    settings.include_static_objects = True

    sim.recompute_navmesh(sim.pathfinder, settings)

    print(f"NavMesh ready  |  navigable area: {sim.pathfinder.navigable_area:.1f} m²")


# ─────────────────────────────────────────────
# 3.  PATH PLANNING
# ─────────────────────────────────────────────

def plan_path(sim, start, end):
    """Return list of (x, y, z) waypoints, or empty list if unreachable."""
    p = habitat_sim.ShortestPath()
    p.requested_start = np.array(start, dtype=np.float32)
    p.requested_end   = np.array(end,   dtype=np.float32)
    found = sim.pathfinder.find_path(p)
    if not found:
        print(f"  ⚠ No path from {start} to {end}")
        return []
    print(f"  Path found  |  {len(p.points)} waypoints  "
          f"|  geodesic dist: {p.geodesic_distance:.2f} m")
    return [tuple(pt) for pt in p.points]


def interpolate_path(waypoints, steps_per_segment=30):
    """
    Linearly interpolate between waypoints so agents move smoothly
    at a fixed speed rather than teleporting.
    """
    frames = []
    for i in range(len(waypoints) - 1):
        a = np.array(waypoints[i])
        b = np.array(waypoints[i + 1])
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            frames.append(tuple(a + t * (b - a)))
    frames.append(waypoints[-1])
    print(f"Path interpolated!!!!")
    return frames


# ─────────────────────────────────────────────
# 4.  OVERHEAD CAMERA HELPERS
# ─────────────────────────────────────────────

CAMERA_HEIGHT = 12.0   # metres above the floor
CAMERA_CENTRE = (-5.0, CAMERA_HEIGHT, -5.0)   # above scene centre


def set_overhead_camera(sim):
    state = sim.agents[0].get_state()
    state.position = np.array(CAMERA_CENTRE, dtype=np.float32)

    angle = -np.pi / 2
    state.rotation = np.quaternion(
        np.cos(angle / 2),
        np.sin(angle / 2),
        0.0,
        0.0
    )
    sim.agents[0].set_state(state, reset_sensors = True)


def draw_agent_overlay(frame, pos3d_list, colors, sim):
    # Explicitly convert to a standard numpy array first
    frame = np.array(frame, dtype=np.uint8)

    H = frame.shape[0]   # index directly instead of tuple unpacking
    W = frame.shape[1]   # guarantees plain Python int via index
    # H = int(H)
    # W = int(W)

    scene_w = 10.0
    scene_h = 10.0
    x_min, z_min = -10.0, -10.0

    for pos, color in zip(pos3d_list, colors):
        # print(type(pos[0]))
        # print(type(pos[1]))
        print("pos3d_list:", pos3d_list)
        print("colors:", colors)

        pos_flat = np.array(pos).flatten()
        x = float(pos_flat[0])   # index instead of tuple unpacking
        z = float(pos_flat[-1])   # forces scalar regardless of input type

        px = int((x - x_min) / scene_w * W)
        py = int((z - z_min) / scene_h * H)
        py = H - py

        if 0 <= px < W and 0 <= py < H:
            cv2.circle(frame, (px, py), 12, color, -1)
            cv2.circle(frame, (px, py), 12, (255, 255, 255), 2)

    return frame


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    VIDEO_PATH   = "scene_walkthrough.mp4"
    FPS          = 24
    RESOLUTION   = (1280, 720)   # W × H  for OpenCV
    STEPS_PER_SEG = 20           # interpolation smoothness

    # ── Build sim ────────────────────────────────────────────────────────────
    print("Initialising simulator …")
    sim = habitat_sim.Simulator(make_cfg())
    build_scene(sim)

    # MORE DEBUGGING
    # ← Check this value before recompute
    print("Objects in scene:", sim.get_rigid_object_manager().get_num_objects())
    recompute_navmesh(sim)
    print("Navigable area after bake:", sim.pathfinder.navigable_area)

    recompute_navmesh(sim)


    # DEBUGGING

    print("\nRandom navigable points:")
    for i in range(5):
        pt = sim.pathfinder.get_random_navigable_point()
        print(f"  {i}: {pt}")

    print("\nNavigability of start/end points:")
    test_points = [
        ("p0 start", (-2.0, 0.1, -1.0)),
        ("p0 end",   (-2.5, 0.1, -0.5)),
        ("p1 start", (-7.0, 0.1, -8.0)),
        ("p1 end",   (-6.0, 0.1, -7.5)),
    ]
    for label, pt in test_points:
        arr = np.array(pt, dtype=np.float32)
        print(f"  {label}: navigable={sim.pathfinder.is_navigable(arr)}, snapped={sim.pathfinder.snap_point(arr)}")

    print("\nRaw path results:")
    for (start, end) in [
        ((-2.0, 0.1, -1.0), (-2.5, 0.1, -0.5)),
        ((-7.0, 0.1, -8.0), (-6.0, 0.1, -7.5)),
    ]:
        p = habitat_sim.ShortestPath()
        p.requested_start = np.array(start, dtype=np.float32)
        p.requested_end   = np.array(end,   dtype=np.float32)
        found = sim.pathfinder.find_path(p)
        print(f"  {start} → {end}")
        print(f"    found={found}, num points={len(p.points)}, geodesic dist={p.geodesic_distance:.3f}")
        print(f"    points={p.points}")


    # ── Plan paths ───────────────────────────────────────────────────────────
    print("Planning paths …")

    plan_path0 = plan_path(sim,
                  start=(-2.0, 0.1, -1.0),
                  end  =(-2.5, 0.1, -0.5)),
    
    plan_path1 = plan_path(sim,
                  start=(-7.0, 0.1, -8.0),
                  end  =(-6.0, 0.1, -7.5)),
    
    print(plan_path0)
    print(plan_path1)

    # Agent 0: room A → through doorway → room B
    path0 = interpolate_path(
        plan_path0,
        steps_per_segment=STEPS_PER_SEG
    )

    # Agent 1: room B → through doorway → room A  (opposite direction)
    path1 = interpolate_path(
        plan_path1,
        steps_per_segment=STEPS_PER_SEG
    )

    if not path0 or not path1:
        print("path0:", path0)
        print("path1:", path1)
        print("Path planning failed — check that start/end points are navigable.")
        sim.close()
        return

    total_frames = max(len(path0), len(path1))
    print(f"Rendering {total_frames} frames → {VIDEO_PATH}")

    # ── Lock overhead camera once ─────────────────────────────────────────────
    set_overhead_camera(sim)
    print("Managed to set overhead camera")

    # ── Video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, RESOLUTION)

    # Agent colours for the overlay dot: BGR
    AGENT_COLORS = [
        (80,  80,  255),   # red   — agent 0
        (80,  255, 80),    # green — agent 1
    ]

    print("Managed to set video writer")

    # ── Render loop ───────────────────────────────────────────────────────────
    for frame_idx in range(total_frames):
        # Clamp so shorter path holds its final position
        pos0 = path0[min(frame_idx, len(path0) - 1)]
        pos1 = path1[min(frame_idx, len(path1) - 1)]

        print("Managed to clamp shorter paths")

        # TODO: here is where the issues happens

        # Move agent 1 (no camera — just update world position via rigid body
        # or directly through the agent state)
        # state1 = sim.agents[1].get_state()
        # state1.position = np.array(pos1, dtype=np.float32)
        # sim.agents[1].set_state(state1)

        print("Managed to move agent 1")

        # Keep overhead camera fixed (agent 0 must not drift)
        set_overhead_camera(sim)

        print("Managed to set overhead camera 1")

        # Render
        obs   = sim.get_sensor_observations()
        frame = obs["overhead_rgb"][..., :3]   # H×W×3, drop alpha
        frame = np.ascontiguousarray(frame)

        # Overlay agent positions as dots
        frame = draw_agent_overlay(
            frame,
            pos3d_list=[pos0, pos1],
            colors=AGENT_COLORS,
            sim=sim
        )

        # Add frame counter HUD
        cv2.putText(frame, f"Frame {frame_idx+1}/{total_frames}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Agent 1", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, AGENT_COLORS[0], 2)
        cv2.putText(frame, "Agent 2", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, AGENT_COLORS[1], 2)

        # OpenCV expects BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if frame_idx % 50 == 0:
            print(f"  … frame {frame_idx}/{total_frames}")

    writer.release()
    sim.close()
    print(f"\nDone!  Video saved to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()