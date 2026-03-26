import habitat_sim
import magnum as mn
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

# Habitat imports
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.utils import viz_utils as vut
from habitat.articulated_agents.robots import FetchRobot
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
    HeadRGBSensorConfig,
    HeadPanopticSensorConfig,
    SimulatorConfig,
    HabitatSimV0Config,
    AgentConfig,
    TaskConfig,
    EnvironmentConfig,
    DatasetConfig,
    HabitatConfig,
    ArmActionConfig,
    BaseVelocityActionConfig,
    OracleNavActionConfig,
)
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    HumanoidPickActionConfig,
)
from habitat.core.env import Env
import habitat

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

# ==================== HELPER FUNCTIONS ====================

def random_rotation():
    """Generate a random rotation quaternion."""
    random_dir = mn.Vector3(np.random.rand(3)).normalized()
    random_angle = np.random.random() * np.pi
    random_quat = mn.Quaternion.rotation(mn.Rad(random_angle), random_dir)
    return random_quat


def custom_sample_humanoid():
    """Sample a random humanoid pose configuration."""
    base_transform = mn.Matrix4()
    random_rot = random_rotation()
    offset_transform = mn.Matrix4.from_(random_rot.to_matrix(), mn.Vector3())
    joints = []
    num_joints = 54
    for _ in range(num_joints):
        Q = random_rotation()
        joints = joints + list(Q.vector) + [float(Q.scalar)]
    offset_trans = list(np.asarray(offset_transform.transposed()).flatten())
    base_trans = list(np.asarray(base_transform.transposed()).flatten())
    random_vec = joints + offset_trans + base_trans
    return {"human_joints_trans": random_vec}


# ==================== ENVIRONMENT SETUP ====================

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path = "data/hab3_bench_assets/humanoids/female_0/female_0.urdf"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"

# Define sensors that will be attached to this agent
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
    "head_rgb": HeadRGBSensorConfig(),
}

# Create agent dictionary
agent_dict = {"main_agent": main_agent_config}

# Define the actions
action_dict = {
    "humanoid_joint_action": HumanoidJointActionConfig()
}

# Initialize environment
env = init_rearrange_env(agent_dict, action_dict)

# ==================== EXAMPLE 1: BASIC SIMULATION ====================
print("\n=== EXAMPLE 1: Basic Humanoid Simulation ===")
obs = env.reset()

# Display observations
fig, ax = plt.subplots(1, len(obs.keys()), figsize=(15, 5))
for ind, name in enumerate(obs.keys()):
    ax[ind].imshow(obs[name])
    ax[ind].set_axis_off()
    ax[ind].set_title(name)
plt.tight_layout()
plt.savefig("noras-habitat-lab/videos/00_basic_observations.png", dpi=100, bbox_inches='tight')
print("✓ Saved initial observations to noras-habitat-lab/videos/00_basic_observations.png")

# Basic movement with base position and rotation
sim = env.sim
observations = []
num_iter = 100
pos_delta = mn.Vector3(0.02, 0, 0)
rot_delta = np.pi / (8 * num_iter)
art_agent = sim.articulated_agent
sim.reset()

for _ in range(num_iter):
    art_agent.base_pos = art_agent.base_pos + pos_delta
    art_agent.base_rot = art_agent.base_rot + rot_delta
    sim.step({})
    observations.append(sim.get_sensor_observations())

now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_basic_movement_video"
vut.make_video(observations, "third_rgb", "color", name, open_vid=False)
# vut.save_video(name, "noras-habitat-lab/videos/")
print(f"✓ Saved basic movement video to noras-habitat-lab/videos/{name}.mp4")

# ==================== EXAMPLE 2: RANDOM POSE SAMPLING ====================
print("\n=== EXAMPLE 2: Random Pose Sampling ===")
observations = []
num_iter = 40
env.reset()
for i in range(num_iter):
    params = custom_sample_humanoid()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": params
    }
    obs_result = env.step(action_dict)
    observations.append(obs_result)
    if (i + 1) % 10 == 0:
        print(f"  Completed {i + 1}/{num_iter} steps")

now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_random_poses_video"
vut.make_video(observations, "third_rgb", "color", name, open_vid=False)
# vut.save_video(name, "noras-habitat-lab/videos/")
print(f"✓ Saved random poses video to noras-habitat-lab/videos/{name}.mp4")

# ==================== EXAMPLE 3: MOTION PLAYBACK FROM FILE ====================
print("\n=== EXAMPLE 3: Motion Playback (Walking) ===")
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"
humanoid_controller = HumanoidRearrangeController(motion_path)

env.reset()
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []

print(f"  Initial agent position: {env.sim.articulated_agent.base_pos}")

for step in range(100):
    # Calculate walking pose towards a target position
    relative_position = env.sim.articulated_agent.base_pos + mn.Vector3(0, 0, 2)
    humanoid_controller.calculate_walk_pose(relative_position)
    
    # Get the computed pose and execute action
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    obs_result = env.step(action_dict)
    observations.append(obs_result)
    
    if (step + 1) % 25 == 0:
        print(f"  Completed {step + 1}/100 walking steps")

now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_walking_video"
vut.make_video(observations, "third_rgb", "color", name, open_vid=False)
# vut.save_video(name, "noras-habitat-lab/videos/")
print(f"✓ Saved walking video to noras-habitat-lab/videos/{name}.mp4")

# ==================== EXAMPLE 4: REACHING ACTION ====================
print("\n=== EXAMPLE 4: Humanoid Reaching ===")
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl"
humanoid_controller = HumanoidRearrangeController(motion_path)

env.reset()
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []

# Get hand pose and add offset
offset = env.sim.articulated_agent.base_transformation.transform_vector(mn.Vector3(0, 0.3, 0))
hand_pose = env.sim.articulated_agent.ee_transform(0).translation + offset
print(f"  Initial hand pose: {hand_pose}")

for step in range(100):
    # Modify hand pose with random small movements
    hand_pose = hand_pose + mn.Vector3((np.random.rand(3) - 0.5) * 0.1)
    humanoid_controller.calculate_reach_pose(hand_pose, index_hand=0)
    
    # Get the computed pose and execute action
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    obs_result = env.step(action_dict)
    observations.append(obs_result)
    
    if (step + 1) % 25 == 0:
        print(f"  Completed {step + 1}/100 reaching steps")

now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_reaching_video"
vut.make_video(observations, "third_rgb", "color", name, open_vid=False)
# vut.save_video(name, "noras-habitat-lab/videos/")
print(f"✓ Saved reaching video to noras-habitat-lab/videos/{name}.mp4")

print("\n=== ✓ All humanoid simulations completed successfully! ===")
print("Check 'noras-habitat-lab/videos/' for output videos and images.")


obs = env.reset()
_, ax = plt.subplots(1,len(obs.keys()))

for ind, name in enumerate(obs.keys()):
    ax[ind].imshow(obs[name])
    ax[ind].set_axis_off()
    ax[ind].set_title(name)

sim = env.sim
observations = []
num_iter = 100
pos_delta = mn.Vector3(0.02,0,0)
rot_delta = np.pi / (8 * num_iter)
art_agent = sim.articulated_agent
sim.reset()
# set_fixed_camera(sim)
for _ in range(num_iter):
    # TODO: this actually seems to give issues...
    art_agent.base_pos = art_agent.base_pos + pos_delta
    art_agent.base_rot = art_agent.base_rot + rot_delta
    sim.step({})
    observations.append(sim.get_sensor_observations())

# Name the video with the current timestamp to avoid overwiting old attempts
now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_human_tutorial_video"

vut.make_video(
    observations,
    "third_rgb",
    "color",
    name,
    open_vid=True,
)

# TODO: maybe we can make joint_action a subclass of dict, and have a custom function for it
import random
def random_rotation():
    random_dir = mn.Vector3(np.random.rand(3)).normalized()
    random_angle = random.random() * np.pi
    random_rat = mn.Quaternion.rotation(mn.Rad(random_angle), random_dir)
    return random_rat
def custom_sample_humanoid():
    base_transform = mn.Matrix4() 
    random_rot = random_rotation()
    offset_transform = mn.Matrix4.from_(random_rot.to_matrix(), mn.Vector3())
    joints = []
    num_joints = 54
    for _ in range(num_joints):
        Q = random_rotation()
        joints = joints + list(Q.vector) + [float(Q.scalar)]
    offset_trans = list(np.asarray(offset_transform.transposed()).flatten())
    base_trans = list(np.asarray(base_transform.transposed()).flatten())
    random_vec = joints + offset_trans + base_trans
    return {
        "human_joints_trans": random_vec
    }

# We can now call the defined actions
observations = []
num_iter = 40
env.reset()
for _ in range(num_iter):
    params = custom_sample_humanoid()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": params
    }
    observations.append(env.step(action_dict))

# vut.make_video(
#     observations,
#     "third_rgb",
#     "color",
#     "robot_tutorial_video",
#     open_vid=True,
# )

from habitat.utils.humanoid_utils import MotionConverterSMPLX
PATH_TO_URDF = "data/humanoids/humanoid_data/female_3/female_3.urdf"
PATH_TO_MOTION_NPZ = "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.npz"
convert_helper = MotionConverterSMPLX(urdf_path=PATH_TO_URDF)
convert_helper.convert_motion_file(
    motion_path=PATH_TO_MOTION_NPZ,
    output_path=PATH_TO_MOTION_NPZ.replace(".npz", ""),
)

env.reset()
motion_path = "data/humanoids/humanoid_data/walk_motion/CMU_10_04_stageii.pkl" 
# We define here humanoid controller
humanoid_controller = HumanoidSeqPoseController(motion_path)


# Because we want the humanoid controller to generate a motion relative to the current agent, we need to set
# the reference pose
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
# humanoid_controller.apply_base_transformation(env.sim.articulated_agent.base_transformation)

observations = []
for _ in range(humanoid_controller.humanoid_motion.num_poses):
    # These computes the current pose and calculates the next pose
    humanoid_controller.calculate_pose()
    humanoid_controller.next_pose()
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
    observations.append(env.step(action_dict))
    
vut.make_video(
    observations,
    "third_rgb",
    "color",
    "robot_tutorial_video",
    open_vid=True,
)

# Name the video with the current timestamp to avoid overwiting old attempts
now = dt.now().strftime("%Y-%m-%dT%H:%M:%S")
name = f"{now}_motion_converter_video"

vut.make_video(
    observations,
    "third_rgb",
    "color",
    name,
    open_vid=True,
)

#  # Save to root folder
# vut.save_video(
#     name
# )

# As before, we first define the controller, here we use a special motion file we provide for each agent.
motion_path = "data/hab3_bench_assets/humanoids/female_0/female_0_motion_data_smplx.pkl" 
# We define here humanoid controller
humanoid_controller = HumanoidRearrangeController(motion_path)

# We reset the controller
env.reset()
humanoid_controller.reset(env.sim.articulated_agent.base_transformation)
observations = []
print(env.sim.articulated_agent.base_pos)
for _ in range(100):
    # This computes a pose that moves the agent to relative_position
    relative_position = env.sim.articulated_agent.base_pos + mn.Vector3(0,0,1)
    humanoid_controller.calculate_walk_pose(relative_position)
    
    # The get_pose function gives as a humanoid pose in the same format as HumanoidJointAction
    new_pose = humanoid_controller.get_pose()
    action_dict = {
        "action": "humanoid_joint_action",
        "action_args": {"human_joints_trans": new_pose}
    }
print(f"✓ Saved reaching video to noras-habitat-lab/videos/{name}.mp4")

print("\n=== ✓ All humanoid simulations completed successfully! ===")
print("Check 'noras-habitat-lab/videos/' for output videos and images.")