import timeit
start_time = timeit.default_timer()

import numpy as np
import rasterio
import os
from pathlib import Path

# --- SETTINGS ---
filename_ending = "with_2025"
district = "Musanze"
eyear = 2025
season = "B"
num_classes = 4

root = '/cluster/archiving/GIZ/data/'
output_dir = f"{root}outputs/{district}_tiles_{filename_ending}/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Paths
final_label_path = f"{root}outputs/{district}_{season}{eyear}_merged_labels_{filename_ending}.tif"
final_prob_path = f"{root}outputs/{district}_{season}{eyear}_merged_probs_{filename_ending}.tif"
#masked_label_output_path = f"/cluster/archiving/GIZ/{district}_{season}{eyear}_masked_labels_{filename_ending}.tif"
masked_label_output_path = f"{district}_{season}{eyear}_masked_labels_{filename_ending}.tif"
# Parameters
ignore_value = 255
confidence_threshold = 0.5

# === Load merged probabilities ===
with rasterio.open(final_prob_path) as prob_src:
    probs = prob_src.read()  # shape: (num_classes, height, width)
    prob_profile = prob_src.profile
    transform = prob_src.transform

# Compute confidence and mask
confidence_map = np.max(probs, axis=0)
mask = confidence_map >= confidence_threshold  # True where confidence is high

# === Load label raster ===
with rasterio.open(final_label_path) as label_src:
    label_data = label_src.read(1)  # shape: (height, width)
    label_profile = label_src.profile

# Apply mask to label data
masked_labels = np.where(mask, label_data, ignore_value).astype("uint8")

# === Save masked labels ===
label_profile.update({
    "dtype": "uint8",
    "nodata": ignore_value,
    "compress": "deflate",
    "tiled": True,
    "count": 1
})

with rasterio.open(masked_label_output_path, "w", **label_profile) as dst:
    dst.write(masked_labels, 1)

print(f"[INFO] Masked label raster saved to: {masked_label_output_path}")
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)


'''
import os
import subprocess
import glob
import numpy as np
import rasterio

# === Config ===
input_dir = "/cluster/archiving/GIZ/data/outputs/Musanze_tiles_all_with_2025"
output_label_path = "Musanze_2025B_merged_labels.tif"
output_conf_path = "Musanze_2025B_confidence_map.tif"
merged_probs_path = "merged_probs.tif"  # intermediate merged probabilities
VALID_LABELS = [0, 1, 2, 3]
IGNORE_VALUE = 255         # << Use 255 as mask value
CONF_THRESHOLD = 0.8

# === Collect all tiles ===
tile_paths = sorted(glob.glob(os.path.join(input_dir, "*_probs.tif")))

# === Merge tiles using gdalwarp ===
gdalwarp_command = [
    "gdalwarp",
    "-r", "average", "-multi",
    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
    "-overwrite",
    *tile_paths,
    merged_probs_path
]

print("Running gdalwarp to merge tiles with averaging...")
subprocess.run(gdalwarp_command, check=True)

# === Load merged probabilities ===
with rasterio.open(merged_probs_path) as src:
    merged_probs = src.read()  # (n_classes, H, W)
    out_transform = src.transform
    out_meta = src.meta.copy()

# === Generate confidence and label maps ===
confidence_map = np.max(merged_probs, axis=0).astype("float32")
label_map = np.argmax(merged_probs, axis=0).astype("uint8")

# === Create mask for low confidence or zero probability pixels ===
prob_sum = np.sum(merged_probs, axis=0)
low_conf_mask = confidence_map < CONF_THRESHOLD
zero_prob_mask = prob_sum == 0
final_mask = np.logical_or(low_conf_mask, zero_prob_mask)

# === Apply mask: set IGNORE_VALUE (255) in label map, NaN in confidence map
label_map[final_mask] = IGNORE_VALUE
confidence_map[final_mask] = np.nan

# === Write label map (as uint8)
label_meta = out_meta.copy()
label_meta.update({
    "count": 1,
    "dtype": "uint8",
    "transform": out_transform,
    "nodata": IGNORE_VALUE,
    "compress": "deflate",
    "tiled": True,
})
with rasterio.open(output_label_path, "w", **label_meta) as dst:
    dst.write(label_map, 1)

# === Write confidence map (as float32)
conf_meta = out_meta.copy()
conf_meta.update({
    "count": 1,
    "dtype": "float32",
    "transform": out_transform,
    "nodata": np.nan,
    "compress": "deflate",
    "tiled": True,
})
with rasterio.open(output_conf_path, "w", **conf_meta) as dst:
    dst.write(confidence_map, 1)
    
'''
