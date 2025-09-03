import timeit
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from tqdm import tqdm
import random
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import shutil

print("Libraries initialized successfully")

# --------------------- CONFIG ---------------------
root = '/cluster/archiving/GIZ/'
TILE_ROOTS = [ 
    ("PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONA", f"{root}data/PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONA_tiles"),
    ("PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONB", f"{root}data/PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONB_tiles"),
    ("PAPER_RWA_B2025_Merge_v2_TrainSet", f"{root}data/Ruhango_B2025_tiles"),
    ("PAPER_RWA_B2025_Merge_v2_TrainSet", f"{root}data/Musanze_B2025_tiles"),
    ("PAPER_RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyabihu_B2025_tiles"),
    ("PAPER_RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyagatare_B2025_tiles"),
    #("2020_RWA_WAPOR_POLY_111_MERGED_SEASONA_Nyagatare", f"{root}data/Nyagatare_A2021_tiles"), #Collected between 2020-11-30 to 1st December 2020
    #("2021_RWA_WAPOR_POLY_111_MERGED_SEASONB_Nyagatare", f"{root}data/Nyagatare_B2021_tiles"),
    #("Nyagatare_A2019", f"{root}data/Nyagatare_A2019_tiles"),
    #("Nyagatare_A2020", f"{root}data/Nyagatare_A2020_tiles"),
    #("Nyagatare_A2021", f"{root}data/Nyagatare_A2021_tiles"),
    #("Nyagatare_B2019", f"{root}data/Nyagatare_B2019_tiles"),
    #("Nyagatare_B2020", f"{root}data/Nyagatare_B2020_tiles"),
    #("Nyagatare_B2021", f"{root}data/Nyagatare_B2021_tiles"),
]

SHAPEFILE_DIR = f"{root}data/shapefiles"
LABEL_COLUMN = "code"
VALID_LABELS = {0, 1, 2, 3, 4, 5, 6}  # 7 classes: banana, bean, irish potato, maize, rice, sorghum, soybean
OUTPUT_DIR = f"{root}data/WC/"
PATCH_SIZE = 16#8
STRIDE = 8#4
IGNORE_VALUE = 255
NODATA_VALUE = -9999
FALLBACK_MEDIAN = -9999
SKIP_SINGLE_CLASS_PATCHES = False
NUM_WORKERS = multiprocessing.cpu_count() - 10
SPLIT_RATIOS = {"train": 0.7, "val": 0.3}
SEED = 42
PRATIO = 0.04 # Pixel percentage with valid data or labels
random.seed(SEED)

# --------------------- TILE PROCESSING FUNCTION ---------------------
def process_tile(tile_path, vector_path):
    result = defaultdict(list)
    local_distribution = Counter()
    multi_label_count = 0

    try:
        with rasterio.open(tile_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs

            # Read full image to compute band-wise medians
            full_image = src.read()  # [C, H, W]
            band_medians = np.array([
                np.nanmedian(full_image[i][full_image[i] != NODATA_VALUE])
                if np.any(full_image[i] != NODATA_VALUE) else FALLBACK_MEDIAN
                for i in range(full_image.shape[0])
            ])

            # Check for NaN in band_medians and log problematic tiles
            if np.any(np.isnan(band_medians)):
                print(f"  Warning: Tile {os.path.basename(tile_path)} has bands with all NODATA values. Skipping.")
                return result, local_distribution, multi_label_count

            gdf = gpd.read_file(vector_path).to_crs(crs)
            gdf = gdf[gdf.geometry.is_valid & gdf[LABEL_COLUMN].isin(VALID_LABELS)]
            if gdf.empty:
                print(f"  Warning: No valid geometries or labels in {vector_path} for {os.path.basename(tile_path)}")
                return result, local_distribution, multi_label_count

            label_shapes = (
                (geom, int(attr)) for geom, attr in zip(gdf.geometry, gdf[LABEL_COLUMN])
            )

            label_raster = rasterize(
                label_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=IGNORE_VALUE,
                dtype=np.uint8,
            )

            for row in range(0, height, STRIDE):
                for col in range(0, width, STRIDE):
                    end_row = min(row + PATCH_SIZE, height)
                    end_col = min(col + PATCH_SIZE, width)

                    image_patch = src.read(
                        window=Window(col, row, end_col - col, end_row - row)
                    )
                    label_patch = label_raster[row:end_row, col:end_col]

                    # Pad if near edges
                    pad_h = PATCH_SIZE - image_patch.shape[1]
                    pad_w = PATCH_SIZE - image_patch.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        image_patch = np.pad(
                            image_patch,
                            ((0, 0), (0, pad_h), (0, pad_w)),
                            constant_values=NODATA_VALUE,
                        )
                        label_patch = np.pad(
                            label_patch,
                            ((0, pad_h), (0, pad_w)),
                            constant_values=IGNORE_VALUE,
                        )

                    # Compute valid pixel ratios
                    image_valid_ratio = np.sum(
                        np.any(image_patch != NODATA_VALUE, axis=0)
                    ) / (PATCH_SIZE * PATCH_SIZE)
                    label_valid_ratio = np.sum(
                        label_patch != IGNORE_VALUE
                    ) / (PATCH_SIZE * PATCH_SIZE)

                    if label_valid_ratio < PRATIO:
                        continue

                    # Replace NODATA_VALUE with band-wise medians
                    for i in range(image_patch.shape[0]):
                        image_patch[i][image_patch[i] == NODATA_VALUE] = band_medians[i]

                    valid_pixels = label_patch[label_patch != IGNORE_VALUE]
                    if len(valid_pixels) == 0:
                        continue

                    unique_classes = set(valid_pixels)
                    if SKIP_SINGLE_CLASS_PATCHES and len(unique_classes) == 1:
                        continue

                    dominant_class = int(np.bincount(valid_pixels).argmax())
                    result[dominant_class].append((image_patch, label_patch))

                    for c in unique_classes:
                        local_distribution[c] += 1
                    if len(unique_classes) > 1:
                        multi_label_count += 1

    except Exception as e:
        print(f"  Skipped {os.path.basename(tile_path)}: {e}")

    return result, local_distribution, multi_label_count

# MAIN PROCESS
def main():
    start_time = timeit.default_timer()

    # Prepare output dirs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    print("Collecting patches from all seasons...")
    class_distribution = Counter()
    multi_label_patch_count = 0
    class_to_patches = defaultdict(list)

    for season_id, tile_dir in TILE_ROOTS:
        print(f"\nProcessing {season_id}")
        vector_path = os.path.join(SHAPEFILE_DIR, f"{season_id}.shp")

        if not os.path.exists(vector_path):
            print(f"[WARNING] Missing shapefile: {vector_path}")
            continue

        tile_files = [f for f in os.listdir(tile_dir) if f.endswith(".tif")]

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_tile, os.path.join(tile_dir, tile_file), vector_path
                )
                for tile_file in tile_files
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Processing {season_id}"
            ):
                tile_result, tile_distribution, tile_multi_label = future.result()
                for k, v in tile_result.items():
                    class_to_patches[k].extend(v)
                class_distribution.update(tile_distribution)
                multi_label_patch_count += tile_multi_label

    # SPLIT BY CLASS
    print(
        f"\nTotal patches collected: {sum(len(p) for p in class_to_patches.values())}"
    )
    print(f"Multi-class patches: {multi_label_patch_count}")
    print("Class pixel distribution across patches (not pixel count):")
    for cls, count in sorted(class_distribution.items()):
        print(f"  Class {cls}: {count} patches contain this label")

    balanced_splits = {"train": [], "val": []}

    for cls, patches in class_to_patches.items():
        train, val = train_test_split(
            patches,
            train_size=SPLIT_RATIOS["train"],
            test_size=SPLIT_RATIOS["val"],
            random_state=SEED,
        )
        balanced_splits["train"].extend(train)
        balanced_splits["val"].extend(val)

    # SAVE PATCHES
    for split, items in balanced_splits.items():
        for i, (image, label) in enumerate(items):
            # Check for NaN in image patch before saving
            if np.any(np.isnan(image)):
                print(f"  Warning: Patch {split}_{i:05d} contains NaN values. Skipping.")
                continue
            out_path = os.path.join(
                OUTPUT_DIR, split, f"patch_{split}_{i:05d}.npz"
            )
            np.savez_compressed(
                out_path,
                image=image.astype(np.float32),
                mask=label.astype(np.uint8),
            )

    print("\nBalanced patches saved to:", OUTPUT_DIR)
    print(
        "Done! Elapsed time (hours):",
        (timeit.default_timer() - start_time) / 3600.0,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
