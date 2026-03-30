# The use of the task dataset RearrangeDataset-v0 introduces limitations to number of agents in scene
# You must circumvent it if num of agents in scene > 2
# Note that coordinates vary per agent for some reason
# E.g.the walking agent being on y=0 puts it with feet on the floor,
# but all static agents must be y=1 to be floor height
# Which direction x and z produce also vary, so beware that coordinates do not make sense

import habitat_sim
import magnum as mn
import warnings
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
warnings.filterwarnings('ignore')
from habitat_sim.utils.settings import make_cfg
from matplotlib import pyplot as plt
from habitat_sim.utils import viz_utils as vut
from omegaconf import DictConfig
import numpy as np
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default import get_agent_config
import habitat
from habitat_sim.physics import JointMotorSettings, MotionType
from omegaconf import OmegaConf
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig

from habitat.config.default_structured_configs import TaskConfig, EnvironmentConfig, DatasetConfig, HabitatConfig
from habitat.config.default_structured_configs import ArmActionConfig, BaseVelocityActionConfig, OracleNavActionConfig
from habitat.core.env import Env

def make_sim_cfg(agent_dict):
    # Start the scene config
    sim_cfg = SimulatorConfig(type="RearrangeSim-v0")
    
    # This is for better graphics
    sim_cfg.habitat_sim_v0.enable_hbao = True
    sim_cfg.habitat_sim_v0.enable_physics = True

    # Set up an example scene
    sim_cfg.scene = "data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json"
    sim_cfg.scene_dataset = "data/hab3_bench_assets/hab3-hssd/hab3-hssd.scene_dataset_config.json"
    sim_cfg.additional_object_paths = ['data/objects/ycb/configs/']

    cfg = OmegaConf.structured(sim_cfg)
    cfg.agents = OmegaConf.create(
        {name: OmegaConf.structured(agent) for name, agent in agent_dict.items()}
    )
    cfg.agents_order = list(cfg.agents.keys())
    # cfg.num_agents = len(cfg.agents_order)

    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = {"agent_0_humanoid_joint_action": HumanoidJointActionConfig()}
    env_cfg = EnvironmentConfig()
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", data_path="data/hab3_bench_assets/episode_datasets/small_large.json.gz")
    
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
    return Env(res_cfg)

# @param sim: the habitat-sim simulator instance
# @param urdf_paths: list of paths to humanoid URDFs to spawn
# @param placements: list of dicts with "pos" (mn.Vector3) and "yaw" (float in radians) for each humanoid
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
        static_pose_controller.calculate_stop_pose()
        static_joint_pose = np.array(static_pose_controller.joint_pose)
        
        humanoid_obj = aom.add_articulated_object_from_urdf(
            urdf_paths[urdf_index],
            fixed_base=True,
        )
        
        # TODO: failed attempt at resetting the controller so that humanoids don't look weird
        humanoid_obj.set_joint_transforms(
            static_pose_controller.pose_humanoid_joints,
            static_obj_transform,
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

# --------------------------------------------------------------------------- #
##################### Initializing humanoids AND CAMERA in the scene #####################
# --------------------------------------------------------------------------- #

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path =  "data/humanoids/humanoid_data/male_0/male_0.urdf"
main_agent_motion_path = "data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = main_agent_motion_path


# Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
# We will later talk about why giving the sensors these names
main_agent_config.sim_sensors = {
    "third_rgb_1": ThirdRGBSensorConfig(),
    "head_rgb_1": HeadRGBSensorConfig(),
}

# Keep only the controllable task agent in the Rearrange config.
agent_dict = {"agent_0": main_agent_config}


# Robot coordinates
robot_pos = mn.Vector3(-3.8, 1, -6.0)

# Additional humanoids are scene-only articulated objects (not task agents).
scene_humanoid_placements = [
    {"pos": robot_pos, "yaw": 0.75},
    {"pos": mn.Vector3(-4.0, 1, -5.2), "yaw": 2.7},
    {"pos": mn.Vector3(-3.7, 1, -4.5), "yaw": 2.3},
    {"pos": mn.Vector3(-3.1, 1, -4.3), "yaw": 3.6},
]

# Define the actions
action_dict = {
    "agent_0_humanoid_joint_action": HumanoidJointActionConfig()
}
env = init_rearrange_env(agent_dict, action_dict)

# Define here humanoid controller for the main agent
humanoid_controller = HumanoidRearrangeController(main_agent_motion_path)

# ------------------------ SET INITIAL AGENT AND CAM CONFIGURATION ----------------------------- #
env.reset()
sim = env.sim
#sim.reset()

################ Placing of static camera ################
initial_camera_pos = mn.Vector3(6.9, 2, 0.9)
initial_camera_rot = mn.Vector3(-0.35, 1.4, 0) 
"""
(0, 0, 0) faces directly north, aka the short wall in the living room without windows/up on the map pic
(0, 1.5, 0) faces the three windows to the west
(0, 2,5, 0) faces southwest
negative x-value in rotation tilts the camera downwards
"""

# Add the fixed scene camera for recording
camera_sensor_spec = habitat_sim.CameraSensorSpec()
camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
camera_sensor_spec.uuid = "static_cam"
camera_sensor_spec.resolution = [720, 1280]
camera_sensor_spec.position = initial_camera_pos
camera_sensor_spec.orientation = initial_camera_rot
sim.add_sensor(camera_sensor_spec, 0)

################ Inital values p placing for humanoids ################
# Start position for walking agent!!!!
initial_pos = mn.Vector3(-2.5, 0.25, -7.5)

# Start angle (no rotation, in radians)
initial_rot = 0.0

# Initalize the humanoids
main_art_agent = sim.get_agent_data(0).articulated_agent
main_art_agent.base_pos = initial_pos
main_art_agent.base_rot = initial_rot

# Specify each static agent's URDF here IN ORDER!!!!!!!!!
urdf_paths = ["data/humanoids/humanoid_data/neutral_1/neutral_1.urdf", 
              "data/humanoids/humanoid_data/female_3/female_3.urdf", 
              "data/humanoids/humanoid_data/male_1/male_1.urdf", 
              "data/humanoids/humanoid_data/female_1/female_1.urdf"]
static_motion_paths = [
    "data/humanoids/humanoid_data/neutral_1/neutral_1_motion_data_smplx.pkl",
    "data/humanoids/humanoid_data/female_3/female_3_motion_data_smplx.pkl",
    "data/humanoids/humanoid_data/male_1/male_1_motion_data_smplx.pkl",
    "data/humanoids/humanoid_data/female_1/female_1_motion_data_smplx.pkl",
]

 # Generate the static humanaoids by calling spawner
scene_humanoids = spawn_static_humanoids(
    sim,
    urdf_paths,
    scene_humanoid_placements,
    static_motion_paths
)
static_idle_pose_library = build_static_idle_pose_library(static_motion_paths)

# ------------------------ GENERATE MOTIONS ------------------------------------ #

# We reset the controller
humanoid_controller.reset(main_art_agent.base_transformation)
# this controls the walking speed so it can be decreased
# lin_speed = 1 is normal, less than is slower and higher faster
humanoid_controller.set_framerate_for_linspeed(
    lin_speed=0.8,
    ang_speed=2.0,
    ctrl_freq=30.0,
)
observations = []
# obs = ["camera_agent"]["static_cam"]

# ---------- LOOP 1: walk some ----------
target_pos = mn.Vector3(3, 0.25, 25)
# mn.Vector3(x, y, z)
#            ↑  ↑  ↑
#          left up forward
# Generate the fixed target position to walk towards
# target_position = main_art_agent.base_pos + target_pos
target_position = target_pos

num_iter = 60
for step_i in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_walk_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }

    # Upper-body-only idle motion by blending precomputed arm poses.
    for humanoid_i, humanoid_obj in enumerate(scene_humanoids):
        pose_set = static_idle_pose_library[humanoid_i]
        phase = (0.12 * step_i) + (0.9 * humanoid_i)

        right_w = max(0.0, np.sin(phase)) * 0.08
        left_w = max(0.0, np.sin(phase + 1.7)) * 0.06
        both_w = max(0.0, np.sin(phase + 3.1)) * 0.04

        blended_pose = pose_set["neutral"].copy()
        blended_pose += right_w * (pose_set["right"] - pose_set["neutral"])
        blended_pose += left_w * (pose_set["left"] - pose_set["neutral"])
        blended_pose += both_w * (pose_set["both"] - pose_set["neutral"])

        humanoid_obj.joint_positions = blended_pose.astype(np.float32)


    _ = env.step(action_dict)
    sensor_obs = sim.get_sensor_observations()
    observations.append({"static_cam": sensor_obs["static_cam"]})

# ---------- LOOP 2: change direction and walk  ----------
# Set new start pos
main_art_agent.base_pos = target_pos

# Set new target position
target_pos = mn.Vector3(-5, 0, 30) 
target_position = target_pos

# Set number of iterations to dictate walking distance
num_iter = 70

for _ in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_walk_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    _ = env.step(action_dict)
    sensor_obs = sim.get_sensor_observations()
    observations.append({"static_cam": sensor_obs["static_cam"]})

# ---------- LOOP 3: turn towards robot ----------
# Set new start pos
main_art_agent.base_pos = target_pos

# Set number of iterations to dictate walking distance
num_iter = 50

# Set new target position
target_pos = mn.Vector3(-8, 0, -10)  # Adjust the distance as needed
target_position = target_pos

for _ in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_turn_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "agent_0_humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    _ = env.step(action_dict)
    sensor_obs = sim.get_sensor_observations()
    observations.append({"static_cam": sensor_obs["static_cam"]})




# ------------------------ GENERATE OUTPUT VIDEO ------------------------------------ #

vut.make_video(
    observations,
    "static_cam",
    "color",
    "robot_tutorial_video_test",
    open_vid=True,
)

print("Video done!")