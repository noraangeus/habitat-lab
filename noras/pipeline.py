
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

    cfg = OmegaConf.create(sim_cfg)
    cfg.agents = agent_dict

    # More attempts at fixing agent_dict
    # cfg = OmegaConf.structured(sim_cfg)
    # cfg.agents = OmegaConf.create()
    # for name, agent in agent_dict.items():
    #     cfg.agents[name] = OmegaConf.structured(agent)

    cfg.agents_order = list(cfg.agents.keys())
    # cfg.num_agents = len(cfg.agents_order)
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = {"humanoid_joint_action": HumanoidJointActionConfig()}
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

# --------------------------------------------------------------------------- #
##################### Initializing humanoids AND CAMERA in the scene #####################
# --------------------------------------------------------------------------- #

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path = "data/hab3_bench_assets/humanoids/female_0/female_0.urdf"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"


# Define sensors that will be attached to this agent, here a third_rgb sensor and a head_rgb.
# We will later talk about why giving the sensors these names
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}

# We create a dictionary with names of agents and their corresponding agent configuration
agent_dict = {"main_agent": main_agent_config}
              #"camera_agent": camera_agent_config}

# Define the actions
action_dict = {
    "humanoid_joint_action": HumanoidJointActionConfig()
}
env = init_rearrange_env(agent_dict, action_dict)

# As before, we first define the controller, here we use a special motion file we provide for each agent.
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl" 
# We define here humanoid controller
humanoid_controller = HumanoidRearrangeController(motion_path)

# ------------------------ SET INITIAL AGENT AND CAM CONFIGURATION ----------------------------- #
env.reset()
sim = env.sim
#sim.reset()

# Start position!!!!
initial_pos = mn.Vector3(-2, 0, -5)

# Start angle (no rotation, in radians)
initial_rot = 0.0

initial_camera_pos = mn.Vector3(6.5, 2, -1.2)
initial_camera_rot = mn.Vector3(-0.35, 1.7, 0) 
# (0, 0, 0) faces directly north, aka the short wall in the living room without windows/up on the map pic
# (0, 1.5, 0) faces the three windows to the west
# (0, 2,5, 0) faces southwest
# negative x-value in rotation tilts the camera downwards

# Add a fixed scene camera for recording
camera_sensor_spec = habitat_sim.CameraSensorSpec()
camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
camera_sensor_spec.uuid = "static_cam"
camera_sensor_spec.resolution = [720, 1280]
camera_sensor_spec.position = initial_camera_pos
camera_sensor_spec.orientation = initial_camera_rot
sim.add_sensor(camera_sensor_spec, 0)

# Initalize the humanoid
art_agent = sim.articulated_agent
art_agent.base_pos = initial_pos
art_agent.base_rot = initial_rot

# ------------------------ GENERATE MOTIONS ------------------------------------ #

# We reset the controller
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []
# obs = ["camera_agent"]["static_cam"]

# ---------- LOOP 1: walk forward ----------
target_pos = mn.Vector3(-0.001, 0, -0.005)
# mn.Vector3(x, y, z)
#            ↑  ↑  ↑
#          left up forward
# Generate the fixed target position to walk towards
target_position = env.sim.articulated_agent.base_pos + target_pos

num_iter = 100
for _ in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_walk_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    _ = env.step(action_dict)
    sensor_obs = sim.get_sensor_observations()
    observations.append({"static_cam": sensor_obs["static_cam"]})

# ---------- LOOP 2: turn and walk forward in new direction ----------
# Set new start pos
art_agent.base_pos = target_pos

# Set number of iterations to dictate walking distance
num_iter=150

# Set new target position
target_pos = mn.Vector3(-7, 0, -2)  # Adjust the distance as needed
target_position = env.sim.articulated_agent.base_pos + target_pos

for _ in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_walk_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    _ = env.step(action_dict)
    sensor_obs = sim.get_sensor_observations()
    observations.append({"static_cam": sensor_obs["static_cam"]})

# ---------- LOOP 3: turn again  ----------
# Set new start pos
art_agent.base_pos = target_pos

# Set new target position
target_pos = mn.Vector3(-7, 0, 2) 
target_position = env.sim.articulated_agent.base_pos + target_pos

# Set number of iterations to dictate walking distance
num_iter=150

for _ in range(num_iter):
    # This computes a pose that moves the agent to the fixed target position
    humanoid_controller.calculate_walk_pose(target_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
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