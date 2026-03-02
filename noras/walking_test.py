# ==================== IMPORTS ====================

from omegaconf import OmegaConf
from habitat.core.env import Env
from habitat_sim.utils import viz_utils as vut
import magnum as mn
from datetime import datetime as dt

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


# ==================== ATTEMPT: MOTION PLAYBACK FROM FILE ====================
print("\n=== ATTEMPT: Motion Playback (Walking) ===")
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
name = f"{now}_walking_attempt_video"

vut.make_video(
    observations,
    "third_rgb", "color",
    name,
    open_vid=True
    )

# TODO: saving videos doesn't work
# vut.save_video(name, "noras-habitat-lab/videos/")

print(f"✓ Saved walking video to noras-habitat-lab/videos/{name}.mp4")