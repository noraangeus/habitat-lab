import habitat_sim
import magnum as mn
import warnings
warnings.filterwarnings('ignore')
from habitat_sim.utils import viz_utils as vut
import numpy as np
from habitat.config.default_structured_configs import SimulatorConfig, HabitatSimV0Config, AgentConfig
from habitat.config.default_structured_configs import ThirdRGBSensorConfig, HeadRGBSensorConfig, HeadPanopticSensorConfig
from habitat.articulated_agent_controllers import (
    HumanoidRearrangeController,
    HumanoidSeqPoseController,
)
from habitat.config.default_structured_configs import HumanoidJointActionConfig, HumanoidPickActionConfig

from habitat.tasks.rearrange.utils import get_angle_to_pos

from matplotlib import pyplot as plt

from pipeline_utils import init_rearrange_env, generate_trajectory_points, spawn_static_humanoids, build_static_idle_pose_library

# --------------------------------------------------------------------------- #
##################### Initializing humanoids in the scene #####################
# --------------------------------------------------------------------------- #

# Define the agent configuration
main_agent_config = AgentConfig()
urdf_path =  "data/humanoids/humanoid_data/male_0/male_0.urdf"
main_agent_motion_path = "data/humanoids/humanoid_data/male_0/male_0_motion_data_smplx.pkl"
main_agent_config.articulated_agent_urdf = urdf_path
main_agent_config.articulated_agent_type = "KinematicHumanoid"
main_agent_config.motion_data_path = main_agent_motion_path

# Define sensors 
main_agent_config.sim_sensors = {
    "third_rgb": ThirdRGBSensorConfig(),
   "head_rgb": HeadRGBSensorConfig(),
}

# We create a dictionary with names of agents and their corresponding agent configuration
agent_dict = {"main_agent": main_agent_config,
              }

# Define the actions
action_dict = {"humanoid_joint_action": HumanoidJointActionConfig()}
env = init_rearrange_env(agent_dict, action_dict)

# Define the controller
human_main_controller = HumanoidRearrangeController(main_agent_motion_path)

# ------------------------ SET INITIAL CONFIGURATION ----------------------------- #
#env.reset()
obs = env.reset()
sim = env.sim

camera_location = {
    "origin": {
        "position": mn.Vector3(0, 2, 0),
        "orientation": mn.Vector3(-0.4, -1.57, 0) 
        },
    "dinner_table": {
        "position": mn.Vector3(3, 2, -2),
        "orientation": mn.Vector3(0, np.pi, 0) 
    },
    "bed": {
        "position": mn.Vector3(2, 2, 6),
        "orientation": mn.Vector3(0, 0, 0) 
    },
    "fridge": {
        "position": mn.Vector3(7, 2, 0),
        "orientation": mn.Vector3(0, 1.57, 0) 
    }      
}

agent_location = {
    "door": {
        "position": mn.Vector3(-0.5, 0, 1.5),
            },
    "bed": {
        "position": mn.Vector3(-4, 0, 1.5),
            },
    "fridge": {
        "position": mn.Vector3(-1, 0, -4),
            },
    "dinner_table": {
        "position": mn.Vector3(-4, 0, -7),
            },
    "desk": {
        "position": mn.Vector3(-6, 0, -3),
            },
    "room_center": {
        "position": mn.Vector3(-2, 0, -5),
            },
}

################ Placing of static camera ################
cam_location = "dinner_table"
camera_sensor_spec = habitat_sim.CameraSensorSpec()
camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
camera_sensor_spec.uuid = "static_cam"
camera_sensor_spec.resolution = [720, 1280]
camera_sensor_spec.position = camera_location[cam_location]["position"]
camera_sensor_spec.orientation = camera_location[cam_location]["orientation"]
sim.add_sensor(camera_sensor_spec, 0)


################ Generate Trajectories #################
urdf_paths = ["data/humanoids/humanoid_data/female_3/female_3.urdf", 
              "data/humanoids/humanoid_data/male_1/male_1.urdf", 
              "data/humanoids/humanoid_data/female_1/female_1.urdf"]
static_motion_paths = [
    "data/humanoids/humanoid_data/female_3/female_3_motion_data_smplx.pkl",
    "data/humanoids/humanoid_data/male_1/male_1_motion_data_smplx.pkl",
    "data/humanoids/humanoid_data/female_1/female_1_motion_data_smplx.pkl",
]
scene_humanoid_placements = [
    {"pos": mn.Vector3(-4.0, 1, -5.2), "yaw": 2.7, "pose": "neutral"},
    {"pos": mn.Vector3(-3.7, 1, -4.5), "yaw": 2.3, "pose": "neutral"},
    {"pos": mn.Vector3(-3.1, 1, -4.3), "yaw": 3.6, "pose": "neutral"},
]

scene_humanoids = spawn_static_humanoids(
    sim,
    urdf_paths,
    scene_humanoid_placements,
    static_motion_paths
)
static_idle_pose_library = build_static_idle_pose_library(static_motion_paths)

################ Generate Trajectories #################

epsilon = 1e-4

human_main_start_loc = "bed"
human_main_target_loc = ["dinner_table"]

human_main_agent = sim.get_agent_data(0).articulated_agent
human_main_agent.base_pos = agent_location[human_main_start_loc]["position"]
pose_diff = agent_location[human_main_target_loc[0]]["position"] - agent_location[human_main_start_loc]["position"]
human_main_agent.base_rot = get_angle_to_pos(pose_diff)
human_main_controller.reset(human_main_agent.base_transformation)
human_main_controller.set_framerate_for_linspeed(lin_speed=1, ang_speed=2.0, ctrl_freq=30.0,)

observations = []

trajectory_curvature = 2
trajectory_num_points = 4

for next_goal in human_main_target_loc:
    target_location = agent_location[next_goal]["position"]
    next_subgoals = generate_trajectory_points(
        human_main_agent.base_pos,
        target_location,
        trajectory_curvature,
        trajectory_num_points,
    )

    for subgoal in next_subgoals[1:]:
        next_subgoal = mn.Vector3(float(subgoal[0]), float(subgoal[1]), float(subgoal[2]))

        while True:
            pose_diff = next_subgoal - human_main_agent.base_pos
            human_main_controller.calculate_walk_pose(pose_diff)
            new_pose = human_main_controller.get_pose()

            action_dict = {
                "action": "humanoid_joint_action",
                "action_args": {"human_joints_trans": new_pose}
            }
            env.step(action_dict)

            sensor_obs = sim.get_sensor_observations()
            observations.append({"static_cam": sensor_obs["static_cam"]})

            if pose_diff.length() < epsilon:
                break

    print(f"Location reached: {next_goal}")

# Compute the stop pose at the end of the trajectory
human_main_controller.calculate_stop_pose()
new_pose = human_main_controller.get_pose()
action_dict = {
    "action": "humanoid_joint_action",
    "action_args": {"human_joints_trans": new_pose}
}
env.step(action_dict)
sensor_obs = sim.get_sensor_observations()
observations.append({"static_cam": sensor_obs["static_cam"]})


vut.make_video(
    observations,
    "static_cam",
    "color",
    "robot_tutorial_video_test",
    open_vid=True,
)
