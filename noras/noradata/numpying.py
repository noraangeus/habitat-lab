import numpy as np
import pickle

with open("noras/noradata/new_female_0_motion_data.pkl", "rb") as f:
    walk_data = pickle.load(f)

walk_info = walk_data["walk_motion"]

# Convert lists back to numpy arrays
walk_info["joints_array"] = np.array(walk_info["joints_array"])
walk_info["transform_array"] = np.array(walk_info["transform_array"])
walk_info["transform_array2"] = np.array(walk_info["transform_array2"])
walk_info["displacement"] = np.array(walk_info["displacement"])