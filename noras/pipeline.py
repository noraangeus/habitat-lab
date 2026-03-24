
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

    # Set the scene agents
    cfg.agents = agent_dict
    cfg.agents_order = list(cfg.agents.keys())
    return cfg

def make_hab_cfg(agent_dict, action_dict):
    sim_cfg = make_sim_cfg(agent_dict)
    task_cfg = TaskConfig(type="RearrangeEmptyTask-v0")
    task_cfg.actions = action_dict
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
##################### Initializing humanoids in the scene #####################
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

# Define the actions
action_dict = {
    "humanoid_joint_action": HumanoidJointActionConfig()
}
env = init_rearrange_env(agent_dict, action_dict)

# As before, we first define the controller, here we use a special motion file we provide for each agent.
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl" 
# We define here humanoid controller
humanoid_controller = HumanoidRearrangeController(motion_path)

# ------------------------ SET INITIAL AGENT CONFIGURATION ----------------------------- #
env.reset()
sim = env.sim
#sim.reset()
art_agent = sim.articulated_agent

# Start position!!!!
initial_pos = mn.Vector3(-2, 0, -5)
art_agent.base_pos = initial_pos

# Start angle (no rotation, in radians)
initial_rot = 0.0
art_agent.base_rot = initial_rot

# ------------------------ CAMERA CONFIGURATION ----------------------------- #

# Create a new scene node for the third-person camera
scene_graph = sim.get_active_scene_graph()
root_node = scene_graph.get_root_node()
third_cam_node = root_node.create_child()


# Position the camera wherever you want
third_cam_node.translation = initial_pos  # same as inital_pos
# third_cam_node.rotation = habitat_sim.utils.quat_from_angle_axis(
#     0.0, [0, 1, 0]
# )

# Create the sensor spec
cam_spec = habitat_sim.CameraSensorSpec()
cam_spec.uuid = "third_person_camera"
cam_spec.sensor_type = habitat_sim.SensorType.COLOR
cam_spec.resolution = [720, 1280]
cam_spec.position = initial_pos
cam_spec.orientation = mn.Vector3(0, 0, 0)

# Attach the sensor to the node
third_person_cam = habitat_sim.CameraSensor(third_cam_node, cam_spec)


# ------------------------ MAP COORDINATES ------------------------------------ #
# Biggest scene:
# bottom wall has x=0, and x > 0 is outside the flat

# Smallest scene:
# 

# ------------------------ GENERATE MOTIONS ------------------------------------ #

# We reset the controller
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []

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
    observations.append(env.step(action_dict))

# ---------- LOOP 2: turn and walk forward in new direction ----------
# Set new start pos
art_agent.base_pos = target_pos
num_iter=150

# Reset the controller with the new transformation
# TODO: how to make the camera stay smooth?
#humanoid_controller.reset(env.sim.articulated_agent.base_transformation)

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
    observations.append(env.step(action_dict))

# ---------- LOOP 3: turn again  ----------
# Set new start pos
art_agent.base_pos = target_pos

# Reset the controller with the new transformation
# TODO: how to make the camera stay smooth?
# humanoid_controller.reset(env.sim.articulated_agent.base_transformation)

# Sety new target position
target_pos = mn.Vector3(-7, 0, 2) 
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
    observations.append(env.step(action_dict))



# ------------------------ GENERATE OUTPUT VIDEO ------------------------------------ #

vut.make_video(
    observations,
    "overhead_rgb",
    "color",
    "robot_tutorial_video_test",
    open_vid=True,
)

print("Video done!")