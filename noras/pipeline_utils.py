import magnum as mn
import numpy as np
import gzip
import json
import os
from typing import Optional, Sequence

from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from omegaconf import OmegaConf
from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.core.env import Env

from habitat_sim.physics import JointMotorSettings, MotionType

from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)


DEFAULT_HSSD_HAB_SCENE_DATASET = (
    "data/versioned_data/hssd-hab/hssd-hab.scene_dataset_config.json"
)
DEFAULT_HSSD_HAB_SCENE = (
    "data/versioned_data/hssd-hab/scenes/102343992.scene_instance.json"
)
DEFAULT_HSSD_HAB_EPISODE_DATASET = "noras/data/hssd_hab_empty.json.gz"
DEFAULT_ADDITIONAL_OBJECT_PATHS = ["data/objects/ycb/configs/"]


def get_hssd_hab_scene_path(scene_id: str) -> str:
    return f"data/versioned_data/hssd-hab/scenes/{scene_id}.scene_instance.json"


def _load_json_or_gz(file_path: str):
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _write_json_or_gz(file_path: str, payload) -> None:
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "wt", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj)
        return
    with open(file_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj)


def _build_default_hssd_hab_episode(scene_path: str, scene_dataset_path: str):
    return {
        "episode_id": "0",
        "scene_id": scene_path,
        "scene_dataset_config": scene_dataset_path,
        "additional_obj_config_paths": list(DEFAULT_ADDITIONAL_OBJECT_PATHS),
        "start_position": [0, 0, 0],
        "start_rotation": [0, 0, 0, 1],
        "info": {"object_labels": {}},
        "ao_states": {},
        "rigid_objs": [],
        "targets": {},
        "markers": [],
        "name_to_receptacle": {},
    }


def _ensure_default_hssd_hab_episode_dataset(
    episode_dataset_path: str,
    scene_path: str,
    scene_dataset_path: str,
) -> None:
    default_episode = _build_default_hssd_hab_episode(
        scene_path=scene_path,
        scene_dataset_path=scene_dataset_path,
    )
    default_payload = {"episodes": [default_episode]}

    if not os.path.exists(episode_dataset_path):
        _write_json_or_gz(episode_dataset_path, default_payload)
        return

    try:
        payload = _load_json_or_gz(episode_dataset_path)
    except Exception:
        _write_json_or_gz(episode_dataset_path, default_payload)
        return

    if not isinstance(payload, dict):
        _write_json_or_gz(episode_dataset_path, default_payload)
        return

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) == 0:
        _write_json_or_gz(episode_dataset_path, default_payload)
        return

    first_episode = episodes[0]
    if not isinstance(first_episode, dict):
        episodes[0] = default_episode
        payload["episodes"] = episodes
        _write_json_or_gz(episode_dataset_path, payload)
        return

    mutated = False
    for key, value in default_episode.items():
        if key not in first_episode:
            first_episode[key] = value
            mutated = True

    if not isinstance(first_episode.get("info"), dict):
        first_episode["info"] = {"object_labels": {}}
        mutated = True
    elif "object_labels" not in first_episode["info"]:
        first_episode["info"]["object_labels"] = {}
        mutated = True

    if mutated:
        episodes[0] = first_episode
        payload["episodes"] = episodes
        _write_json_or_gz(episode_dataset_path, payload)


def make_sim_cfg(
    agent_dict,
    scene_path: str = DEFAULT_HSSD_HAB_SCENE,
    scene_dataset_path: str = DEFAULT_HSSD_HAB_SCENE_DATASET,
    additional_object_paths: Optional[Sequence[str]] = None,
):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    sim_cfg.scene = scene_path
    sim_cfg.scene_dataset = scene_dataset_path
    sim_cfg.additional_object_paths = list(
        additional_object_paths or DEFAULT_ADDITIONAL_OBJECT_PATHS
    )

    cfg = OmegaConf.create(sim_cfg)

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(
    agent_dict,
    action_dict,
    scene_path: str = DEFAULT_HSSD_HAB_SCENE,
    scene_dataset_path: str = DEFAULT_HSSD_HAB_SCENE_DATASET,
    episode_dataset_path: str = DEFAULT_HSSD_HAB_EPISODE_DATASET,
):
    sim_cfg = make_sim_cfg(
        agent_dict,
        scene_path=scene_path,
        scene_dataset_path=scene_dataset_path,
    )
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(
        type="RearrangeDataset-v0",
        data_path=episode_dataset_path,
    )
    
    hab_cfg = HabitatConfig()
    hab_cfg.environment = env_cfg
    hab_cfg.task = task_cfg
    hab_cfg.dataset = dataset_cfg
    hab_cfg.simulator = sim_cfg
    hab_cfg.simulator.seed = hab_cfg.seed

    return hab_cfg

def init_rearrange_env(
    agent_dict,
    action_dict,
    scene_path: str = DEFAULT_HSSD_HAB_SCENE,
    scene_dataset_path: str = DEFAULT_HSSD_HAB_SCENE_DATASET,
    episode_dataset_path: str = DEFAULT_HSSD_HAB_EPISODE_DATASET,
):
    _ensure_default_hssd_hab_episode_dataset(
        episode_dataset_path=episode_dataset_path,
        scene_path=scene_path,
        scene_dataset_path=scene_dataset_path,
    )
    hab_cfg = make_hab_cfg(
        agent_dict,
        action_dict,
        scene_path=scene_path,
        scene_dataset_path=scene_dataset_path,
        episode_dataset_path=episode_dataset_path,
    )
    res_cfg = OmegaConf.create(hab_cfg)
    return Env(res_cfg)


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
        
        # Reset the controller right before pose calculation
        static_pose_controller.reset(static_obj_transform)
        static_pose_controller.calculate_stop_pose()

        
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

        urdf_index = (urdf_index + 1) % len(urdf_paths)
        motion_path_index = (motion_path_index + 1) % len(static_motion_paths)

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
