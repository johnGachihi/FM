import os
import glob
import rasterio
from rasterio.merge import merge
from contextlib import ExitStack

district = "Ruhango"
root = '/cluster/archiving/GIZ/data/outputs/'
tiles_dir = f'{root}/CRF/{district}'
output_path = f'/cluster/archiving/GIZ/maps/{district}_crf_label_merged.tif'

def merge_crf_labels():
    # Collect all CRF label tiles
    label_tiles = sorted(glob.glob(os.path.join(tiles_dir, f'{district}_crf_label_tile_*.tif')))
    if not label_tiles:
        raise FileNotFoundError(f"No CRF label tiles found in {tiles_dir}")

    print(f"Found {len(label_tiles)} tiles to merge", flush=True)

    with ExitStack() as stack:
        # Open all tiles and validate against the first tile
        srcs = []
        first = stack.enter_context(rasterio.open(label_tiles[0]))
        if first.count != 1 or first.dtypes[0] != 'uint8':
            raise ValueError(f"First tile must be single-band uint8, got count={first.count}, dtype={first.dtypes[0]}")
        target_crs = first.crs
        if target_crs is None:
            raise ValueError(f"First tile {label_tiles[0]} has no CRS")

        # Copy profile NOW, while dataset is open
        out_profile = first.profile.copy()

        # Add first and the rest
        srcs.append(first)
        for tile_path in label_tiles[1:]:
            src = stack.enter_context(rasterio.open(tile_path))
            if src.crs != target_crs:
                raise ValueError(f"CRS mismatch: {tile_path} has {src.crs}, expected {target_crs}")
            if src.count != 1 or src.dtypes[0] != 'uint8':
                raise ValueError(f"Tile {tile_path} must be single-band uint8, got count={src.count}, dtype={src.dtypes[0]}")
            nod = src.profile.get('nodata')
            if nod is not None and nod != 255:
                print(f"Warning: {tile_path} nodata={nod}, expected 255", flush=True)
            srcs.append(src)

        print(f"Merging {len(srcs)} valid tiles...", flush=True)

        # Merge while files are still open
        merged, merged_transform = merge(srcs, nodata=255)  # merged shape: (1, H, W)

    # Prepare output profile (after the context, files are closed, but we already copied profile)
    out_profile.update(
        driver='GTiff',
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        transform=merged_transform,
        width=merged.shape[2],
        height=merged.shape[1],
        compress='LZW',
        tiled=True
    )
    # Optional tile sizes for better I/O:
    # out_profile.update(blockxsize=512, blockysize=512)

    # Ensure output directory exists (directory of output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write merged raster
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(merged[0], 1)

    print(f"Merged GeoTIFF saved to {output_path} with CRS {out_profile.get('crs')}", flush=True)
if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()

    # (optional) help GDAL/PROJ find projections when using Conda
    os.environ["PROJ_LIB"] = os.environ.get("CONDA_PREFIX", "") + "/share/proj"

    merge_crf_labels()

    print(f"Elapsed time (hours): {(timeit.default_timer() - start)/3600.0:.3f}", flush=True)

'''
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
'''