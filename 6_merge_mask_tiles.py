import timeit
start_time = timeit.default_timer()
import os
import glob
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from osgeo import gdal
from rasterio.warp import reproject, Resampling
# --------------------
# Config
# --------------------
filename_ending = "with_2025"
CONF_THRESHOLD = 0.4
district = "Musanze"
input_dir = f"/cluster/archiving/GIZ/data/outputs/{district}_tiles_{filename_ending}"
out_dir = "/cluster/archiving/GIZ/maps/"
output_label_path = f"{out_dir}{district}_labels_{filename_ending}_threshold_{CONF_THRESHOLD}.tif"
output_conf_path =  f"{out_dir}{district}_confidence_map_{filename_ending}_threshold_{CONF_THRESHOLD}.tif" 
merged_probs_path = f"{out_dir}{district}_merged_probs_{filename_ending}_threshold_{CONF_THRESHOLD}.tif" 
IGNORE_VALUE = 255

# Paths for masking steps
mask_raster_path = "/home/bkenduiywo/data/masks/ESA_DW_cropland_mask_2025.tif"  # Binary raster: 1=keep, 0=exclude
protected_areas_path = "/home/bkenduiywo/data/masks/Protected_Areas.shp"  # Shapefile

# IMPORTANT: set this to match your per-tile _probs.tif
SRC_NODATA = -9999  
DST_NODATA = np.nan    

# --------------------
# Find tiles
# --------------------
tile_paths = sorted(glob.glob(os.path.join(input_dir, "*_probs.tif")))
if not tile_paths:
    raise RuntimeError(f"No tiles found in {input_dir} matching '*_probs.tif'")
print(f"Found {len(tile_paths)} tiles.")

# --------------------
# Merge all tiles
# --------------------
warp_opts = gdal.WarpOptions(
    resampleAlg="average",
    multithread=True,
    srcNodata=SRC_NODATA,
    dstNodata=DST_NODATA,
    outputType=gdal.GDT_Float32,
    creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"]
)

merged_ds = gdal.Warp(merged_probs_path, tile_paths, options=warp_opts)
if merged_ds is None:
    raise RuntimeError("GDAL Warp failed.")
merged_ds = None
print(f"Merged probs written: {merged_probs_path}")

# --------------------
# Load merged and compute label + confidence
# --------------------
with rasterio.open(merged_probs_path) as src:
    merged_probs = src.read().astype("float32")  # shape (C, H, W)
    out_transform = src.transform
    out_crs = src.crs
    out_meta = src.meta.copy()

np.clip(merged_probs, 0.0, 1.0, out=merged_probs)

valid_pixels = np.any(np.isfinite(merged_probs), axis=0)
confidence_map = np.full(valid_pixels.shape, np.nan, dtype="float32")
confidence_map[valid_pixels] = np.nanmax(merged_probs[:, valid_pixels], axis=0)

probs_filled = np.where(np.isfinite(merged_probs), merged_probs, -np.inf)
top_class = np.argmax(probs_filled, axis=0).astype("uint8")

label_map = np.full(confidence_map.shape, IGNORE_VALUE, dtype="uint8")
confident_mask = np.isfinite(confidence_map) & (confidence_map >= CONF_THRESHOLD)
label_map[confident_mask] = top_class[confident_mask]

print("Unique labels in output:", np.unique(label_map))

# --------------------
# Step 1: Apply binary raster mask
# --------------------
with rasterio.open(mask_raster_path) as mask_src:
    mask_data = mask_src.read(1)
    mask_transform = mask_src.transform
    mask_crs = mask_src.crs

# Reproject if CRS/grid differs
if (mask_crs != out_crs) or (mask_transform != out_transform) or (mask_data.shape != label_map.shape):
    print("[INFO] Mask CRS/grid differs from labels – reprojecting mask…")
    mask_reproj = np.empty(label_map.shape, dtype=mask_data.dtype)
    reproject(
        source=mask_data,
        destination=mask_reproj,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=out_transform,
        dst_crs=out_crs,
        resampling=Resampling.nearest
    )
    mask_data = mask_reproj
else:
    print("[INFO] Mask already aligned with labels – no reprojection needed.")

# Keep only mask=1
label_map[mask_data != 1] = IGNORE_VALUE

# --------------------
# Step 2: Remove pixels in protected areas
# --------------------
gdf = gpd.read_file(protected_areas_path).to_crs(out_crs)
protected_mask = features.rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=label_map.shape,
    transform=out_transform,
    fill=0,
    dtype="uint8"
)
label_map[protected_mask == 1] = IGNORE_VALUE

# --------------------
# Save results
# --------------------
color_map = {
    0: (85, 255, 0),
    1: (115, 38, 0),
    2: (255, 212, 0),
    3: (0, 169, 230),
    255: (0, 0, 0)
}

label_meta = out_meta.copy()
label_meta.update({
    "count": 1,
    "dtype": "uint8",
    "transform": out_transform,
    "crs": out_crs,
    "nodata": IGNORE_VALUE,
    "compress": "DEFLATE",
    "tiled": True
})

with rasterio.open(output_label_path, "w", **label_meta) as dst:
    dst.write(label_map, 1)
    dst.write_colormap(1, color_map)

conf_meta = out_meta.copy()
conf_meta.update({
    "count": 1,
    "dtype": "float32",
    "transform": out_transform,
    "crs": out_crs,
    "nodata": np.nan,
    "compress": "DEFLATE",
    "tiled": True
})
with rasterio.open(output_conf_path, "w", **conf_meta) as dst:
    dst.write(confidence_map, 1)

print("Done. Wrote:")
print(f"  Labels:     {output_label_path}")
print(f"  Confidence: {output_conf_path}")
# Remove merged_probs file to save space
try:
    os.remove(merged_probs_path)
    print(f"Deleted intermediate file: {merged_probs_path}")
except OSError as e:
    print(f"Could not delete {merged_probs_path}: {e}")
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)

