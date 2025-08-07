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

# === Collect all tiles ===
tile_paths = sorted(glob.glob(os.path.join(input_dir, "*_probs.tif")))

# === Use gdalwarp to mosaic with averaging for overlapping regions ===
# This will average pixel values where overlaps exist
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
    merged_probs = src.read()
    out_transform = src.transform
    out_meta = src.meta.copy()

# === Generate label and confidence maps ===
label_map = np.argmax(merged_probs, axis=0).astype("uint8")
confidence_map = np.max(merged_probs, axis=0).astype("float32")

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

'''
import os
import numpy as np
import rasterio
from rasterio.merge import merge
from osgeo import gdal

# ------------------------
# Configurations
# ------------------------
district = 'Musanze'
input_dir = "/cluster/archiving/GIZ/data/outputs/Musanze_tiles_all_with_2025/"  # directory with rasters
out_root = '/cluster/archiving/GIZ/results/'
filename_map = f"{out_root}{district}_crop_map.tif"
filename_conf = f"{out_root}{district}_confidence_map.tif"
nodata_value = 255
prob_threshold = 0.7

# ------------------------
# Step 1: Load and merge all raster tiles
# ------------------------
input_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith("_probs.tif")]

tmp_vrt_path = f"{out_root}temp_merged.vrt"
final_prob_output_path = f"{out_root}merged_probabilities.tif"

# Build VRT
vrt = gdal.BuildVRT(
    tmp_vrt_path,
    input_paths,
    options=gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=False, separate=False)
)
vrt.FlushCache()

# Warp to average overlaps
gdal.Warp(
    destNameOrDestDS=final_prob_output_path,
    srcDSOrSrcDSTab=tmp_vrt_path,
    format="GTiff",
    options=gdal.WarpOptions(
        resampleAlg="average",
        outputType=gdal.GDT_Float32,
        dstNodata=float('nan'),
        multithread=True,
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "BIGTIFF=YES"
        ]
    )
)
os.remove(tmp_vrt_path)

print(f"[✓] Saved merged probability raster to: {final_prob_output_path}")

# ------------------------
# Step 2: Load merged data and compute predicted class + confidence
# ------------------------
with rasterio.open(final_prob_output_path) as src:
    merged_data = src.read()  # (bands=classes, height, width)
    merged_profile = src.profile
    merged_transform = src.transform
    merged_crs = src.crs

class_indices = np.argmax(merged_data, axis=0).astype(np.uint8)   # Most probable class
confidences = np.max(merged_data, axis=0).astype(np.float32)      # Associated confidence

# Apply confidence threshold
predicted_map = np.where(confidences >= prob_threshold, class_indices, nodata_value).astype(np.uint8)
print("Classes found:", np.unique(class_indices))
print("Pixels retained after threshold:", np.sum(confidences >= prob_threshold))

# ------------------------
# Step 3: Save predicted class map
# ------------------------
pred_profile = merged_profile.copy()
pred_profile.update({
    'count': 1,
    'dtype': 'uint8',
    'transform': merged_transform,
    'crs': merged_crs,
    'nodata': nodata_value,
    'compress': 'deflate',
    'tiled': True
})

with rasterio.open(filename_map, 'w', **pred_profile) as dst:
    dst.write(predicted_map, 1)

print(f"[✓] Saved predicted class map to: {filename_map}")

# ------------------------
# Step 4: Save confidence map
# ------------------------
conf_profile = merged_profile.copy()
conf_profile.update({
    'count': 1,
    'dtype': 'float32',
    'transform': merged_transform,
    'crs': merged_crs,
    'nodata': None,
    'compress': 'deflate',
    'tiled': True
})

with rasterio.open(filename_conf, 'w', **conf_profile) as dst:
    dst.write(confidences, 1)

print(f"[✓] Saved confidence map to: {filename_conf}")
'''
