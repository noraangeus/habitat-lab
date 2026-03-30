import pickle

# this one is OK!
# with open("data/humanoids/humanoid_data/female_3/female_3_motion_data_smplx.pkl", "rb") as f:
#     data = pickle.load(f)

# this one too!
# with open("data/humanoids/humanoid_data/male_1/male_1_motion_data_smplx.pkl", "rb") as f:
#     data = pickle.load(f)

with open("data/humanoids/humanoid_data/female_1/female_1_motion_data_smplx.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())
print(data["stop_pose"].keys())