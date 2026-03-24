import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load your JSON scene file
with open("data/hab3_bench_assets/hab3-hssd/scenes/103997919_171031233.scene_instance.json", "r") as f:
    scene = json.load(f)

objects = scene["object_instances"]

# Group by template_name for coloring
groups = defaultdict(list)
for obj in objects:
    x, y, z = obj["translation"]
    groups[obj["template_name"]].append((x, z))

plt.figure(figsize=(8, 8))

for template, coords in groups.items():
    xs = [c[0] for c in coords]
    zs = [c[1] for c in coords]
    plt.scatter(xs, zs, label=template[:6])  # short label

plt.xlabel("X")
plt.ylabel("Z")
plt.title("Top-down map of scene 103997919_171031233")
plt.gca().invert_yaxis()  # optional: make negative Z appear “up”
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(True, linestyle="--", alpha=0.4)
plt.axis("equal")
plt.tight_layout()
plt.show()

#plt.savefig("scene_102816009_map.png", dpi=300)
