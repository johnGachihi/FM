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
IGNORE_VALUE = 255
CONF_THRESHOLD = 0.5

# === Collect all tiles ===
tile_paths = sorted(glob.glob(os.path.join(input_dir, "*_probs.tif")))

# === Use gdalwarp to mosaic with averaging for overlapping regions ===
gdalwarp_command = [
    "gdalwarp",
    "-r", "average",                  # average overlapping pixels
    "-multi",                         # use multiple cores
    "-co", "COMPRESS=DEFLATE",        # apply compression
    "-co", "TILED=YES",               # enable tiling
    "-overwrite",                     # overwrite if file exists
    *tile_paths,
    merged_probs_path
]

print("Running gdalwarp to merge tiles with averaging...")
subprocess.run(gdalwarp_command, check=True)

# === Load merged probabilities ===
with rasterio.open(merged_probs_path) as src:
    merged_probs = src.read()  # shape: (n_classes, height, width)
    out_transform = src.transform
    out_meta = src.meta.copy()
# === Generate label and confidence maps ===
label_map = np.argmax(merged_probs, axis=0).astype("uint8")
confidence_map = np.max(merged_probs, axis=0).astype("float32")

# === Mask background pixels where all class probabilities are 0 ===
prob_sum = np.sum(merged_probs, axis=0)  # Shape: (height, width)
bg_mask = prob_sum == 0

# Assign IGNORE_VALUE to background pixels
label_map[bg_mask] = IGNORE_VALUE
confidence_map[bg_mask] = np.nan
# === Write label map ===
out_meta.update({
    "count": 1,
    "dtype": "uint8",
    "transform": out_transform,
    "nodata": IGNORE_VALUE,
    "compress": "deflate",
    "tiled": True,
})
with rasterio.open(output_label_path, "w", **out_meta) as dst:
    dst.write(label_map, 1)

# === Write confidence map ===
conf_meta = out_meta.copy()
conf_meta.update({
    "dtype": "float32",
    "nodata": np.nan,
})
with rasterio.open(output_conf_path, "w", **conf_meta) as dst:
    dst.write(confidence_map, 1)

print("Done. Merged label and confidence maps written.")
