import os
import glob
import numpy as np
import rasterio
from osgeo import gdal

# --------------------
# Config
# --------------------
input_dir = "/cluster/archiving/GIZ/data/outputs/Musanze_tiles_all_with_2025"
output_label_path = "Musanze_B2025_merged_labels.tif"
output_conf_path = "Musanze_B2025_confidence_map.tif"
merged_probs_path = "Musanze_B2025_merged_probs.tif"  # intermediate merged probabilities
 
CONF_THRESHOLD = 0.5
IGNORE_VALUE = 255
 
# IMPORTANT: set this to match your per-tile _probs.tif
# If tiles use 0.0 to mean "no data", set to 0.0. If they use NaN, set to np.nan.
SRC_NODATA = np.nan   # or 0.0
 
DST_NODATA = np.nan   # keep NaN for the merged probs/confidence
 
 
tile_paths = sorted(glob.glob(os.path.join(input_dir, "*_probs.tif")))
if not tile_paths:
    raise RuntimeError(f"No tiles found in {input_dir} matching '*_probs.tif'")
print(f"Found {len(tile_paths)} tiles.")
 
 
vrt = gdal.BuildVRT("/vsimem/probs.vrt", tile_paths)
if vrt is None:
    raise RuntimeError("Failed to build VRT from tiles.")
 
warp_opts = gdal.WarpOptions(
    resampleAlg="average",
    multithread=True,
    srcNodata=float(SRC_NODATA) if np.isfinite(SRC_NODATA) else SRC_NODATA,
    dstNodata=float(DST_NODATA) if np.isfinite(DST_NODATA) else DST_NODATA,
    outputType=gdal.GDT_Float32,
    creationOptions=["COMPRESS=DEFLATE", "TILED=YES"],
)
warped = gdal.Warp(merged_probs_path, vrt, options=warp_opts)
if warped is None:
    raise RuntimeError("GDAL Warp failed.")
warped = None
vrt = None
print(f"Merged probs written: {merged_probs_path}")
 
 
with rasterio.open(merged_probs_path) as src:
    merged_probs = src.read().astype("float32")  # (C,H,W)
    out_transform = src.transform
    out_crs = src.crs
    out_meta = src.meta.copy()
 
# Optional safety clamp to 0..1
np.clip(merged_probs, 0.0, 1.0, out=merged_probs)
 
 
# confidence = max prob per pixel, ignoring NaNs
confidence_map = np.nanmax(merged_probs, axis=0).astype("float32")  # (H,W)
 
# argmax that ignores NaNs: fill NaN with -inf so they can't win
probs_filled = np.where(np.isfinite(merged_probs), merged_probs, -np.inf)
top_class = np.argmax(probs_filled, axis=0).astype("uint8")
 
# Initialize labels as IGNORE, then fill where confident
label_map = np.full(confidence_map.shape, np.nan, dtype="float32")
valid_mask = np.isfinite(confidence_map) & (confidence_map >= CONF_THRESHOLD)
label_map[valid_mask] = top_class[valid_mask]
 
# Confidence: set non-confident (or NaN) to NaN
confidence_map[~valid_mask] = np.nan
 
 
label_meta = out_meta.copy()
label_meta.update({
    "count": 1,
    "dtype": "float32",
    "transform": out_transform,
    "crs": out_crs,
    "nodata": IGNORE_VALUE,
    "compress": "deflate",
    "tiled": True,
})
with rasterio.open(output_label_path, "w", **label_meta) as dst:
    dst.write(label_map, 1)
 
conf_meta = out_meta.copy()
conf_meta.update({
    "count": 1,
    "dtype": "float32",
    "transform": out_transform,
    "crs": out_crs,
    "nodata": np.nan,
    "compress": "deflate",
    "tiled": True,
})
with rasterio.open(output_conf_path, "w", **conf_meta) as dst:
    dst.write(confidence_map, 1)
 
print("Done. Wrote:")
print(f"  Labels:     {output_label_path} (uint8, nodata={IGNORE_VALUE})")
print(f"  Confidence: {output_conf_path} (float32, nodata=NaN)")