# ==================== IMPORTS ====================

import habitat_sim
from omegaconf import OmegaConf
from habitat.core.env import Env
from habitat_sim.utils import viz_utils as vut
from matplotlib import pyplot as plt
from datetime import datetime as dt
import magnum as mn
import numpy as np
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat_sim.physics import JointMotorSettings, MotionType

from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import (
    HumanoidJointActionConfig,
    HumanoidPickActionConfig,
)
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

# ==================== CONFIGS SETUP ====================

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
    dataset_cfg = DatasetConfig(type="RearrangeDataset-v0", split="train", data_path="output/dataset.json.gz")

# The above line achieves the same as:
# habitat:
#   dataset:
#     type: RearrangeDataset-v0
#     split: train
#     data_path: output/dataset.json.gz
    
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

# TODO: check if I can get this to work
def init_rearrange_sim(agent_dict):
    # Start the scene config
    sim_cfg = make_sim_cfg(agent_dict)    
    cfg = OmegaConf.create(sim_cfg)
    
    # Create the scene
    sim = RearrangeSim(cfg)

    # This is needed to initialize the agents
    sim.agents_mgr.on_new_scene()

    # For this tutorial, we will also add an extra camera that will be used for third person recording.
    camera_sensor_spec = habitat_sim.CameraSensorSpec()
    camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    camera_sensor_spec.uuid = "scene_camera_rgb"

    # TODO: this is a bit dirty but I think its nice as it shows how to modify a camera sensor...
    sim.add_sensor(camera_sensor_spec, 0)

    return sim


# ==================== ENVIRONMENT SETUP ====================

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path = "data/hab3_bench_assets/humanoids/female_2/female_2.urdf"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = "data/hab3_bench_assets/humanoids/female_2/female_2_motion_data_smplx.pkl"

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

# TODO: attempt at initializing sim
sim = init_rearrange_sim(agent_dict)


# ==================== ATTEMPT 1: MOTION PLAYBACK FROM FILE ====================
print("\n=== ATTEMPT 1: Motion Playback (Walking) ===")
motion_path = "noras/data/female_0_motion_data_smplx.pkl"
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
name = f"{now}_walking_attempt_video"

# Make video both makes the video and save to the root folder
# TODO: figure out how to add oputput path
vut.make_video(
    observations,
    "third_rgb", "color",
    name,
    open_vid=True
    )

# vut.save_video(name, "noras-habitat-lab/videos/")

print(f"✓ Saved walking video to noras-habitat-lab/videos/{name}.mp4")


# ==================== ATTEMPT 2: REACHING ACTION ====================
print("\n=== ATTEMPT 2: Humanoid Reaching ===")
motion_path = "noras/data/female_0_motion_data_smplx.pkl"
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
name = f"{now}_reaching_attempt_video"

vut.make_video(observations,
               "third_rgb",
               "color",
               name,
               open_vid=True)

print(f"✓ Saved reaching video to noras-habitat-lab/videos/{name}.mp4")


# ==================== ATTEMPT 3: ARTICULATION ATTEMPT ====================
print("\n=== ATTEMPT 3: Humanoid Articulation ===")
motion_path = "noras/data/female_0_motion_data_smplx.pkl"
humanoid_controller = HumanoidRearrangeController(motion_path)

init_pos = mn.Vector3(-5.5,0,-1.5)
art_agent = sim.articulated_agent
# We will see later about this
art_agent.sim_obj.motion_type = MotionType.KINEMATIC
print("Current agent position:", art_agent.base_pos)
art_agent.base_pos = init_pos 
print("New agent position:", art_agent.base_pos)
# We take a step to update agent position
_ = sim.step({})

observations = sim.get_sensor_observations()
print(observations.keys())

# TODO: somewhere after here it fails

_, ax = plt.subplots(1,len(observations.keys()))

for ind, name in enumerate(observations.keys()):
    ax[ind].imshow(observations[name])
    ax[ind].set_axis_off()
    ax[ind].set_title(name)

art_agent.params.cameras.keys()

observations = []
num_iter = 100
pos_delta = mn.Vector3(0.02,0,0)
rot_delta = np.pi / (8 * num_iter)
art_agent.base_pos = init_pos

# sim.reset()
# # set_fixed_camera(sim)
# for _ in range(num_iter):
#     # TODO: this actually seems to give issues...
#     art_agent.base_pos = art_agent.base_pos + pos_delta
#     art_agent.base_rot = art_agent.base_rot + rot_delta
#     sim.step({})
#     observations.append(sim.get_sensor_observations())

# vut.make_video(
#     observations,
#     "scene_camera_rgb",
#     "color",
#     "object_interaction_1_tutorial_video",
#     open_vid=True,
# )
# vut.make_video(
#     observations,
#     "third_rgb",
#     "color",
#     "object_interaction_2_tutorial_video",
#     open_vid=True,
# )

# sim.reset()

observations = []
# We start by setting the arm to the minimum value
lower_limit = art_agent.arm_joint_limits[0].copy()
lower_limit[lower_limit == -np.inf] = 0
upper_limit = art_agent.arm_joint_limits[1].copy()
upper_limit[upper_limit == np.inf] = 0
for i in range(num_iter):
    alpha = i/num_iter
    current_joints = upper_limit * alpha + lower_limit * (1 - alpha)
    art_agent.arm_joint_pos = current_joints
    sim.step({})
    observations.append(sim.get_sensor_observations())
    if i in [0, num_iter-1]:
        print(f"Step {i}:")
        print("Arm joint positions:", art_agent.arm_joint_pos)
        print("Arm end effector translation:", art_agent.ee_transform().translation)
        print(art_agent.sim_obj.joint_positions)

vut.make_video(
    observations,
    "third_rgb",
    "color",
    "object_interaction_3_tutorial_video",
    open_vid=True,
)