import magnum as mn
import numpy as np
from typing import Dict, Optional, Tuple

from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from omegaconf import OmegaConf
from habitat.core.env import Env
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0, RearrangeEpisode
from habitat.sims.habitat_simulator.debug_visualizer import DebugVisualizer

from habitat_sim.physics import JointMotorSettings, MotionType

from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)

#SCENE = "HSSD"
SCENE = "ReplicaCAD"

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    if SCENE == "HSSD":
        sim_cfg.scene = "data/hab3_bench_assets/hab3-hssd/scenes/102816009.scene_instance.json"
        sim_cfg.scene_dataset = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"

    elif SCENE == "ReplicaCAD":
        # --> Look into data/replica_cad/scene_instances for more available scenes
        #sim_cfg.scene = "data/replica_cad/configs/scenes/apt_1.scene_instance.json"
        #sim_cfg.scene = "data/replica_cad/configs/scenes/v3_sc3_staging_20.scene_instance.json"

        # -->  Or look into the versioned_data/replica_cad_dataset for more recent scene instances
        sim_cfg.scene = "data/versioned_data/replica_cad_dataset/configs/scenes/apt_0.scene_instance.json"
        sim_cfg.scene = "data/versioned_data/replica_cad_dataset/configs/scenes/v3_sc1_staging_05.scene_instance.json"

        sim_cfg.scene_dataset = "data/replica_cad/replicaCAD.scene_dataset_config.json"

    sim_cfg.additional_object_paths = ['data/objects/ycb/configs/']
    cfg = OmegaConf.create(sim_cfg)
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()

    if SCENE == "HSSD":
        dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")

    elif SCENE == "ReplicaCAD":
        # Keep this tutorial in single-scene mode while still supplying thedataset object required by RearrangeEmptyTask.
        dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/datasets/in_memory_rearrange.json.gz")
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(agent_dict, action_dict):
    hab_cfg = make_hab_cfg(agent_dict, action_dict)
    res_cfg = OmegaConf.create(hab_cfg)

    if SCENE == "HSSD":
        return Env(res_cfg)


    elif SCENE == "ReplicaCAD":
        dataset = RearrangeDatasetV0(config=None)
        dataset.config = res_cfg.dataset
        dataset.episodes = [
            RearrangeEpisode(
                episode_id="0",
                scene_id=res_cfg.simulator.scene,
                scene_dataset_config=res_cfg.simulator.scene_dataset,
                additional_obj_config_paths=list(
                    res_cfg.simulator.additional_object_paths
                ),
                start_position=[0.0, 0.0, 0.0],
                start_rotation=[0.0, 0.0, 0.0, 1.0],
                info={"object_labels": {}},
                ao_states={},
                rigid_objs=[],
                targets={},
                markers=[],
                target_receptacles=[],
                goal_receptacles=[],
                name_to_receptacle={},
            )
        ]
        return Env(res_cfg, dataset=dataset)


def quick_debug_scene_views(
    env,
    views: Optional[Dict[str, Dict[str, mn.Vector3]]] = None,
    resolution: Tuple[int, int] = (500, 500),
    show: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Capture scene snapshots from multiple viewpoints using DebugVisualizer.

    Usage right after env creation:
        env = init_rearrange_env(agent_dict, action_dict)
        views = quick_debug_scene_views(env)

    :param env: Habitat env instance.
    :param views: Optional mapping of view names to {"look_from", "look_at"}.
        If None, three default views are generated from scene bounds.
    :param resolution: Debug camera resolution as (height, width).
    :param show: If True, displays captured views with matplotlib.
    :return: Dict of {view_name: RGB ndarray}.
    """

    sim = env.sim if hasattr(env, "sim") else env._sim
    dbv = DebugVisualizer(sim=sim, resolution=resolution)

    if views is None:
        scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        center = scene_bb.center()
        size = scene_bb.size()
        radius = max(float(size[0]), float(size[2]), 1.0)
        height = max(float(size[1]), 2.0)

        views = {
            "oblique": {
                "look_from": center + mn.Vector3(radius, height * 0.8, radius),
                "look_at": center,
            },
            "top_down": {
                "look_from": center + mn.Vector3(0.0, height * 2.0, 0.0),
                "look_at": center,
            },
            "side": {
                "look_from": center + mn.Vector3(radius * 1.5, height * 0.5, 0.0),
                "look_at": center,
            },
        }

    captured: Dict[str, np.ndarray] = {}
    for name, pose in views.items():
        obs = dbv.get_observation(
            look_at=pose["look_at"],
            look_from=pose["look_from"],
        )
        captured[name] = np.array(obs.obs_data)

    if show and len(captured) > 0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, len(captured), figsize=(5 * len(captured), 4))
        if len(captured) == 1:
            ax = [ax]
        for i, (name, img) in enumerate(captured.items()):
            ax[i].imshow(img)
            ax[i].set_axis_off()
            ax[i].set_title(name)
        plt.tight_layout()
        plt.show()

    dbv.remove_dbv_agent()
    return captured


def generate_trajectory_points(
    start_location: mn.Vector3,
    end_location: mn.Vector3,
    curvature: int,
    num_points: int,
) -> np.ndarray:
    """
    Generate a trajectory between two 3D points.

    If `curvature == 0`, the trajectory is a straight line. Otherwise, a
    quadratic Bezier curve is created by offsetting the midpoint in the XZ plane.

    :param start_location: trajectory start point.
    :param end_location: trajectory end point.
    :param curvature: curve offset amount. The magnitude controls how curved
        the path is, and the sign controls which side the curve bends toward.
    :param num_points: number of sampled points along the trajectory.
    :return: numpy array of shape `(num_points, 3)`.
    """

    if num_points <= 0:
        raise ValueError("num_points must be greater than 0")

    start = np.array(start_location, dtype=np.float32)
    end = np.array(end_location, dtype=np.float32)

    if num_points == 1:
        return np.array([start], dtype=np.float32)

    t_values = np.linspace(0.0, 1.0, num_points, dtype=np.float32)

    if curvature == 0:
        return np.array(
            [(1.0 - t) * start + t * end for t in t_values],
            dtype=np.float32,
        )

    midpoint = 0.5 * (start + end)
    direction = end - start
    flat_direction = np.array([direction[0], 0.0, direction[2]], dtype=np.float32)
    flat_length = np.linalg.norm(flat_direction)

    if flat_length < 1e-6:
        return np.array(
            [(1.0 - t) * start + t * end for t in t_values],
            dtype=np.float32,
        )

    flat_direction /= flat_length
    perpendicular = np.array(
        [-flat_direction[2], 0.0, flat_direction[0]], dtype=np.float32
    )
    control_point = midpoint + (float(curvature) * perpendicular)

    return np.array(
        [
            ((1.0 - t) ** 2) * start
            + (2.0 * (1.0 - t) * t) * control_point
            + (t**2) * end
            for t in t_values
        ],
        dtype=np.float32,
    )


def _build_static_joint_pose(
    pose_controller: HumanoidRearrangeController, pose_name: str
) -> np.ndarray:
    pose_key = pose_name.lower()

    if pose_key in ("neutral", "stop"):
        pose_controller.calculate_stop_pose()
    elif pose_key == "right":
        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(
            mn.Vector3(0.12, 0.78, 0.03), index_hand=0
        )
    elif pose_key == "left":
        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(
            mn.Vector3(-0.12, 0.78, 0.03), index_hand=1
        )
    elif pose_key == "both":
        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(
            mn.Vector3(0.08, 0.80, 0.04), index_hand=0
        )
        pose_controller.calculate_reach_pose(
            mn.Vector3(-0.08, 0.80, 0.04), index_hand=1
        )
    else:
        raise ValueError(
            f"Unsupported static humanoid pose '{pose_name}'. "
            "Use one of: neutral, right, left, both."
        )

    return np.array(pose_controller.joint_pose, dtype=np.float32)

# @param sim: the habitat-sim simulator instance
# @param urdf_paths: list of paths to humanoid URDFs to spawn
# @param placements: list of dicts with "pos" (mn.Vector3), "yaw" (float in radians),
#                    and optional "pose" in {"neutral", "right", "left", "both"}
def spawn_static_humanoids(sim, urdf_paths, placements, static_motion_paths):
    """
    Spawn scene-only humanoids as articulated objects.
    These are not part of Rearrange task agents, so they do not clash
    with task actions/dataset/simulator agent config state.
    """
    aom = sim.get_articulated_object_manager()
    if not hasattr(aom, "add_articulated_object_from_urdf"):
        raise RuntimeError(
            "This habitat-sim build does not expose add_articulated_object_from_urdf."
        )

    # Ensures each humanoid gets a different URDF & motion path
    urdf_index = 0
    motion_path_index = 0
    scene_humanoids = []

    for placement in placements:
        static_pose_controller = HumanoidRearrangeController(static_motion_paths[motion_path_index])
        static_yaw_quat = mn.Quaternion.rotation(
            mn.Rad(placement["yaw"]), mn.Vector3(0.0, 1.0, 0.0)
        )
        static_obj_transform = mn.Matrix4.from_(
            static_yaw_quat.to_matrix(), placement["pos"]
        )
        static_pose_controller.reset(static_obj_transform)
        pose_name = placement.get("pose", "neutral")
        static_joint_pose = _build_static_joint_pose(
            static_pose_controller, pose_name
        )
        
        humanoid_obj = aom.add_articulated_object_from_urdf(
            urdf_paths[urdf_index],
            fixed_base=True,
        )
        
        humanoid_obj.motion_type = MotionType.KINEMATIC
        humanoid_obj.translation = placement["pos"]
        humanoid_obj.rotation = static_yaw_quat
        humanoid_obj.joint_positions = static_joint_pose

        urdf_index = (urdf_index + 1)
        motion_path_index = (motion_path_index + 1)

        scene_humanoids.append(humanoid_obj)

    return scene_humanoids


def build_static_idle_pose_library(static_motion_paths):
    """
    Precompute a few subtle upper-body poses per humanoid model.
    We then blend between these poses during loop 1.
    """
    idle_pose_library = []

    for motion_path in static_motion_paths:
        pose_controller = HumanoidRearrangeController(motion_path)

        pose_controller.calculate_stop_pose()
        neutral_pose = np.array(pose_controller.joint_pose, dtype=np.float32)

        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(mn.Vector3(0.12, 0.78, 0.03), index_hand=0)
        right_pose = np.array(pose_controller.joint_pose, dtype=np.float32)

        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(mn.Vector3(-0.12, 0.78, 0.03), index_hand=1)
        left_pose = np.array(pose_controller.joint_pose, dtype=np.float32)

        pose_controller.calculate_stop_pose()
        pose_controller.calculate_reach_pose(mn.Vector3(0.08, 0.80, 0.04), index_hand=0)
        pose_controller.calculate_reach_pose(mn.Vector3(-0.08, 0.80, 0.04), index_hand=1)
        both_pose = np.array(pose_controller.joint_pose, dtype=np.float32)

        idle_pose_library.append(
            {
                "neutral": neutral_pose,
                "right": right_pose,
                "left": left_pose,
                "both": both_pose,
            }
        )

    return idle_pose_library
