import pickle
import json
import numpy as np

# Source - https://stackoverflow.com/a/63345888
# Posted by Sarabjeet Sandhu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-09, License - CC BY-SA 4.0

# with open('file.txt', 'r') as data:
#     contents = data.read()

# Open the json
with open("noras/noradata/new_female_0_motion_data.json", "r") as data:
    new_female_0_motion_data_obj = json.load(data)

walk_info = new_female_0_motion_data_obj["walk_motion"]
stop_info = new_female_0_motion_data_obj["stop_pose"]
left_hand_info = new_female_0_motion_data_obj["left_hand"]


# Convert lists to numpy arrays (apparently necessary?)

# the walking info
walk_info["joints_array"] = np.array(walk_info["joints_array"])
walk_info["transform_array"] = np.array(walk_info["transform_array"], dtype=np.float32)
walk_info["transform_array2"] = np.array(walk_info["transform_array2"],dtype=np.float32)
walk_info["displacement"] = np.array(walk_info["displacement"])

# the stop info
stop_info["joints"] = np.array(stop_info["joints"])
stop_info["transform"] = np.array(stop_info["transform"], dtype=np.float32)

# left hand info
pose_motion = left_hand_info["pose_motion"]
pose_motion["joints_array"] = np.array(pose_motion["joints_array"])
pose_motion["transform_array"] = np.array(pose_motion["transform_array"], dtype=np.float32)
left_hand_info["coord"] = np.array(left_hand_info["coord"])
left_hand_info["coord_info"] = np.array(left_hand_info["coord_info"], dtype=object)
#coord_info = left_hand_info["coord_info"]
#coord_info["num_bins"] = np.array(coord_info["num_bins"])
left_hand_info["min"] = np.array(left_hand_info["min"])
left_hand_info["max"] = np.array(left_hand_info["max"])

# Open the target file to write to
new_female_0_motion_data_file = open("noras/noradata/new_new_female_0_motion_data.pkl", "wb")
pkl_name = "new_new_female_0_motion_data_file.pkl"

new_female_0_motion_data = pickle.dump(new_female_0_motion_data_obj, new_female_0_motion_data_file)
new_female_0_motion_data_file.close()
print (f"✓ Succesfully pickled {pkl_name} ")


# Base structure for JSON based on the unpickled version of female_0_motion_data_smplx.pkl:
# {'walk_motion':
#   {'joints_array': array([]),
#   'transform_array': array([]),
#   'transform_array2': array([]),
#   'displacement': array([]),
#   'fps': 120.0
#   },
# 'stop_pose':
#   {'joints': array([]),
#   'transform': array([], dtype=float32)
#   },
#   'left_hand':
#     {'pose_motion':
#       {'joints_array': array([]),
#       'transform_array': array([], dtype=float32),
#       'displacement': None,
#       'fps': 1
#       },
#     'coord': array([[]),
#     'coord_info': array({'num_bins': array([]),
#     'min': array([]),
#     'max': array([])},
#         dtype=object)
#     }
# }


