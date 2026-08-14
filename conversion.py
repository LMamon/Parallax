import json
import numpy as np
from pathlib import Path

src = Path("config/camera/calibration/results/rectification/")

dst = src

# load existing calibration
P1 = np.load(src / "P1.npy")
P2 = np.load(src / "P2.npy")
Q  = np.load(src / "Q.npy")
R1 = np.load(src / "R1.npy")
R2 = np.load(src / "R2.npy")

left_map_x = np.load(src / "left_map_x.npy")
left_map_y = np.load(src / "left_map_y.npy")
right_map_x = np.load(src / "right_map_x.npy")
right_map_y = np.load(src / "right_map_y.npy")

# load existing metadata
with open(src / "metadata.json") as f:
    metadata = json.load(f)

# put the small matrices into the JSON
metadata["P1"] = P1.tolist()
metadata["P2"] = P2.tolist()
metadata["Q"]  = Q.tolist()
metadata["R1"] = R1.tolist()
metadata["R2"] = R2.tolist()

metadata["map_format"] = {"dtype": "float32",
                        "byte_order": "little",
                        "layout": "row_major"}

with open(dst / "calibration.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Raw contiguous float32 maps.
left_map_x.astype("<f4", copy=False).tofile(dst / "left_map_x.bin")
left_map_y.astype("<f4", copy=False).tofile(dst / "left_map_y.bin")
right_map_x.astype("<f4", copy=False).tofile(dst / "right_map_x.bin")
right_map_y.astype("<f4", copy=False).tofile(dst / "right_map_y.bin")