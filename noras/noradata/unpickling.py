import pickle

female_0_motion_data_file = open("noras/data/female_0_motion_data_smplx.pkl", "rb")
female_0_motion_data = pickle.load(female_0_motion_data_file)
female_0_motion_data_file.close()

with open("female_0_motion_data.py", "x") as f:
    f.write(str(female_0_motion_data))
    # except Exception as e:
#    print(f"Error writing to file: {e}")

