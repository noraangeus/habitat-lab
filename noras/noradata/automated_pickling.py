import pickle
import json
import numpy as np

def to_numpy(value, dtype=None):
    """Convert lists/dicts to numpy arrays where appropriate."""
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    elif isinstance(value, list):
        try:
            # Try float32 first (matches your explicit dtype choices for transform arrays)
            arr = np.array(value, dtype=np.float32)
            # Fall back to default dtype if it contains ints that look like flags (0/1)
            # or mixed shapes
            return arr
        except (ValueError, TypeError):
            # Jagged or mixed — use object dtype
            return np.array(value, dtype=object)
    elif value is None or isinstance(value, (int, float, str, bool)):
        return value
    return value

# Load JSON (change input file HERE)
with open("noras/noradata/new_female_0_motion_data.json", "r") as f:
    motion_data = json.load(f)

# Recursively convert all lists to numpy arrays
motion_data = to_numpy(motion_data)

# Save as pickle (change output name HERE)
pkl_path = "noras/noradata/automated_female_0_motion_data.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(motion_data, f)

print(f"✓ Successfully pickled {pkl_path}")

# Additional allow list
FLOAT32_KEYS = {"transform_array", "transform_array2", "transform", "transform_array"}

def to_numpy(value, key=None):
   if isinstance(value, dict):
       return {k: to_numpy(v, key=k) for k, v in value.items()}
   elif isinstance(value, list):
       dtype = np.float32 if key in FLOAT32_KEYS else None
       try:
           return np.array(value, dtype=dtype)
       except (ValueError, TypeError):
           return np.array(value, dtype=object)
   return value
