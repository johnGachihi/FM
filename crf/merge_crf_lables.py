import timeit
start_time = timeit.default_timer()
import numpy as np
import rasterio
from rasterio.merge import merge
import glob
import os

# Global variables
district = "Musanze"
root = '/cluster/archiving/GIZ/data/outputs/'
output_dir = f'{root}/CRF/{district}/'
output_path = f'/cluster/archiving/GIZ/maps/{district}_crf_label_merged.tif'

def merge_crf_labels():
    # Collect all CRF label tiles
    label_tiles = sorted(glob.glob(f'{output_dir}/{district}_crf_label_tile_*.tif'))
    if not label_tiles:
        raise FileNotFoundError(f"No CRF label tiles found in {output_dir}")

    print(f"Found {len(label_tiles)} tiles to merge", flush=True)

    # Open all tiles and validate
    src_files = []
    for tile_path in label_tiles:
        src = None
        try:
            src = rasterio.open(tile_path)
            if src.crs != rasterio.crs.CRS.from_epsg(4326):
                raise ValueError(f"Tile {tile_path} has CRS {src.crs}, expected EPSG:4326")
            if src.profile.get('nodata') != 255:
                print(f"Warning: Tile {tile_path} has nodata value {src.profile.get('nodata')}, expected 255", flush=True)
            if src.count != 1 or src.dtypes[0] != 'uint8':
                raise ValueError(f"Tile {tile_path} must be single-band uint8, got count={src.count}, dtype={src.dtypes[0]}")
            src_files.append(src)
        except Exception as e:
            print(f"Error opening tile {tile_path}: {e}", flush=True)
            if src is not None:
                src.close()
            continue

    if not src_files:
        raise ValueError("No valid tiles to merge")

    print(f"Merging {len(src_files)} valid tiles...", flush=True)

    # Merge tiles
    merged_data, merged_transform = merge(src_files, nodata=255)

    # Close all source files
    for src in src_files:
        src.close()

    # Prepare output profile
    profile = src_files[0].profile.copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        transform=merged_transform,
        width=merged_data.shape[2],
        height=merged_data.shape[1],
        compress='lzw'
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write merged output
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(merged_data[0], 1)

    print(f"Merged GeoTIFF saved to {output_path} with CRS {profile['crs']}", flush=True)
    print(f"Elapsed time (hours): {(timeit.default_timer() - start_time) / 3600.0}", flush=True)

if __name__ == "__main__":
    os.environ["PROJ_LIB"] = os.environ.get("CONDA_PREFIX", "") + "/share/proj"  # Use Conda PROJ
    merge_crf_labels()