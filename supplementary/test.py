import timeit
import numpy as np
import rasterio
from rasterio.mask import mask
from osgeo import gdal, osr
from pathlib import Path
import geopandas as gpd

start_time = timeit.default_timer()

# --- SETTINGS ---
filename_ending = "with_2025"
district = "Musanze"
eyear = 2025
season = "B"
num_classes = 4

root = '/cluster/archiving/GIZ/data/'
output_dir = Path(f"{root}outputs/{district}_tiles_{filename_ending}/")
output_dir.mkdir(parents=True, exist_ok=True)

# Paths
prob_image_path = f"{root}outputs/{district}_{season}{eyear}_merged_probs_{filename_ending}.tif"
mask_image_path = f"{root}masks/ESA_crop_lands_only_mask.tif"
aoi_shapefile_path = f"{root}shapefiles/rwa_adm2_selected_districts.shp"
output_label_path = f"{district}_{season}{eyear}_masked_labels_{filename_ending}.tif"

# Parameters
ignore_value = 255
confidence_threshold = 0.5

def clip_raster(input_path, aoi_path, district):
    # Validate inputs
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")
    if not Path(aoi_path).exists():
        raise FileNotFoundError(f"AOI shapefile not found: {aoi_path}")

    # Filter AOI
    aoi_shp = gpd.read_file(aoi_path)
    aoi = aoi_shp[aoi_shp["ADM2_EN"] == district]
    if aoi.empty:
        raise ValueError(f"District '{district}' not found in shapefile")

    geometries = aoi.geometry.values

    with rasterio.open(input_path) as src:
        clipped_image, clipped_transform = mask(src, geometries, crop=True, nodata=ignore_value)
        profile = src.profile.copy()

    profile.update({
        'height': clipped_image.shape[1],
        'width': clipped_image.shape[2],
        'transform': clipped_transform,
        'nodata': ignore_value
    })

    return clipped_image, clipped_transform, profile

def process_probability_image(prob_image_path, mask_image_path, aoi_shapefile_path, output_label_path):
    # Clip probability and mask images
    prob_image, prob_transform, prob_profile = clip_raster(prob_image_path, aoi_shapefile_path, district)
    mask_image, _, _ = clip_raster(mask_image_path, aoi_shapefile_path, district)

    prob_image = np.transpose(prob_image, (1, 2, 0)).astype(np.float32)
    mask_image = mask_image[0]  # (H, W)

    if prob_image.shape[-1] != num_classes:
        raise ValueError(f"Expected {num_classes}-channel probability image, got {prob_image.shape}")

    height, width, _ = prob_image.shape
    if mask_image.shape != (height, width):
        raise ValueError("Probability and mask image shapes do not match.")

    if not np.all(np.isin(mask_image, [0, 1, ignore_value])):
        raise ValueError("Mask image must contain only 0, 1 or ignore_value.")

    # Process
    max_prob = np.max(prob_image, axis=2)
    max_prob_idx = np.argmax(prob_image, axis=2)
    nan_mask = np.any(np.isnan(prob_image), axis=2)

    valid_mask = (~nan_mask) & (max_prob >= confidence_threshold) & (mask_image == 1)

    label_image = np.full((height, width), ignore_value, dtype=np.uint8)
    label_image[valid_mask] = max_prob_idx[valid_mask]

    # Output GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(str(output_label_path), width, height, 1, gdal.GDT_Byte)

    if out_dataset is None:
        raise RuntimeError(f"Failed to create output file: {output_label_path}")

    out_dataset.SetGeoTransform(prob_transform.to_gdal())

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32736)  # WGS 84 / UTM zone 36S
    out_dataset.SetProjection(srs.ExportToWkt())

    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(label_image)
    out_band.SetNoDataValue(ignore_value)
    out_band.FlushCache()
    out_dataset = None

    print(f"[INFO] Class label image saved to {output_label_path} (EPSG:32736, 10m resolution)")
    return label_image

# === Run processing ===
if __name__ == "__main__":
    try:
        label_image = process_probability_image(prob_image_path, mask_image_path, aoi_shapefile_path, output_label_path)
        elapsed = (timeit.default_timer() - start_time) / 3600.0
        print(f"✅ Done! Elapsed time (hours): {elapsed:.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise




