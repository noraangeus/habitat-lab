import pickle
import json

# Source - https://stackoverflow.com/a/63345888
# Posted by Sarabjeet Sandhu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-09, License - CC BY-SA 4.0

# with open('file.txt', 'r') as data:
#     contents = data.read()


with open("noras/noradata/new_female_0_motion_data.json", "r") as data:
    new_female_0_motion_data_obj = json.load(data)
new_female_0_motion_data_file = open("noras/noradata/new_female_0_motion_data.pkl", "wb")
pkl_name = "new_female_0_motion_data_file.pkl"
new_female_0_motion_data = pickle.dump(new_female_0_motion_data_obj, new_female_0_motion_data_file)
new_female_0_motion_data_file.close()
print (f"✓ Succesfully pickled {pkl_name} ")


