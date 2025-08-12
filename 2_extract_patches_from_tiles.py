"""
Original patch extraction logic enhanced to pad small polygons with IGNORE_VALUE,
ensuring no valuable labeled data is lost. In addition, filters out patches
with less than `pratio` pixels of available data or labels.
Keeps the pixel-wise classification output format using .npz files.
"""
# ------------------------------------------------------------
# Author: Joseph Chemut and Benson Kenduiywo
# Accuracy assessment using pre-rasterized reference labels
# ------------------------------------------------------------
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
    ("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Ruhango_B2025_v2_tiles"),
    ("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Musanze_B2025_v2_tiles"),
    ("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyabihu_B2025_v2_tiles"),
    ("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyagatare_B2025_v2_tiles"),
    ("2021_RWA_WAPOR_POLY_111_MERGED_SEASONA_Nyagatare", f"{root}data/Nyagatare_A2021_v2_tiles"),
    ("2021_RWA_WAPOR_POLY_111_MERGED_SEASONB_Nyagatare", f"{root}data/Nyagatare_B2021_v2_tiles"),
    ("Nyagatare_A2019", f"{root}data/Nyagatare_A2019_v2_tiles"),
    ("Nyagatare_A2020", f"{root}data/Nyagatare_A2020_v2_tiles"),
    ("Nyagatare_A2021", f"{root}data/Nyagatare_A2021_v2_tiles"), 
    ("Nyagatare_B2019", f"{root}data/Nyagatare_B2019_v2_tiles"),
    ("Nyagatare_B2020", f"{root}data/Nyagatare_B2020_v2_tiles"),
    ("Nyagatare_B2021", f"{root}data/Nyagatare_B2021_v2_tiles"),
]

SHAPEFILE_DIR = f"{root}data/shapefiles"
LABEL_COLUMN = "code"
VALID_LABELS = {0, 1, 2, 3}
OUTPUT_DIR = f"{root}data/patches_25/"
PATCH_SIZE = 8
STRIDE = 4
IGNORE_VALUE = 255
NODATA_VALUE = -9999
SKIP_SINGLE_CLASS_PATCHES = False
NUM_WORKERS  = multiprocessing.cpu_count() - 10
SPLIT_RATIOS = {"train": 0.7, "val": 0.3}
SEED = 42
pratio = 0.05 # Pixel percentage with eiter valid data or labels in a patch to retain
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

            gdf = gpd.read_file(vector_path).to_crs(crs)
            gdf = gdf[gdf.geometry.is_valid & gdf[LABEL_COLUMN].isin(VALID_LABELS)]
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

                    if image_valid_ratio < pratio or label_valid_ratio < pratio:
                        continue

                    # Reset NODATA_VALUE to 0 for retained patches
                    image_patch[image_patch == NODATA_VALUE] = 0

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



## BEFORE MODIFYING CODE TO ACCEPT POLYGONS LESS THAN 8by8 Pixels XXXXXXX
'''
import timeit
start_time = timeit.default_timer()
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

print('Libraries initialized successfully') #district = 'Musanze' #Name of the district being classified
# --------------------- CONFIG --------------------- 
#root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/'
root = '/cluster/archiving/GIZ/'
#Add name of shapefile without extension and corresponding image tiles folder name to extract training data
TILE_ROOTS = [ 
    #("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Ruhango_B2025_v2_tiles"),
    #("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Musanze_B2025_v2_tiles"),
    #("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyabihu_B2025_v2_tiles"),
    ("RWA_B2025_Merge_v2_TrainSet", f"{root}data/Nyagatare_B2025_v2_tiles"),
    #("2021_RWA_WAPOR_POLY_111_MERGED_SEASONA_Nyagatare", f"{root}data/Nyagatare_A2021_v2_tiles"),
    ("2021_RWA_WAPOR_POLY_111_MERGED_SEASONB_Nyagatare", f"{root}data/Nyagatare_B2021_v2_tiles"),
    #("Nyagatare_A2019", f"{root}data/Nyagatare_A2019_v2_tiles"),
    #("Nyagatare_A2020", f"{root}data/Nyagatare_A2020_v2_tiles"),
    #("Nyagatare_A2021", f"{root}data/Nyagatare_A2021_v2_tiles"), 
    ("Nyagatare_B2019", f"{root}data/Nyagatare_B2019_v2_tiles"),
    ("Nyagatare_B2020", f"{root}data/Nyagatare_B2020_v2_tiles"),
    ("Nyagatare_B2021", f"{root}data/Nyagatare_B2021_v2_tiles"),
]

SHAPEFILE_DIR = f"{root}data/shapefiles"
LABEL_COLUMN = "code"
VALID_LABELS = {0, 1, 2, 3}
OUTPUT_DIR = f"{root}data/patches_SB/"
PATCH_SIZE = 8
STRIDE = 4
IGNORE_VALUE = 255
SKIP_SINGLE_CLASS_PATCHES = False
NUM_WORKERS  = multiprocessing.cpu_count() - 20

SPLIT_RATIOS = {"train": 0.7, "val": 0.3}  # test dropped
SEED = 42
random.seed(SEED)

# Clean OUTPUT_DIR
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Create split subdirectories
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

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

            gdf = gpd.read_file(vector_path).to_crs(crs)
            gdf = gdf[gdf.geometry.is_valid & gdf[LABEL_COLUMN].isin(VALID_LABELS)]
            label_shapes = ((geom, int(attr)) for geom, attr in zip(gdf.geometry, gdf[LABEL_COLUMN]))

            label_raster = rasterize(
                label_shapes,
                out_shape=(height, width),
                transform=transform,
                fill=IGNORE_VALUE,
                dtype=np.uint8
            )

            for row in range(0, height - PATCH_SIZE + 1, STRIDE):
                for col in range(0, width - PATCH_SIZE + 1, STRIDE):
                    window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                    image_patch = src.read(window=window)
                    label_patch = label_raster[row:row + PATCH_SIZE, col:col + PATCH_SIZE]

                    if np.all(label_patch == IGNORE_VALUE):
                        continue

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

# --------------------- MAIN LOOP ---------------------
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
        futures = [executor.submit(process_tile, os.path.join(tile_dir, tile_file), vector_path)
                   for tile_file in tile_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {season_id}"):
            tile_result, tile_distribution, tile_multi_label = future.result()
            for k, v in tile_result.items():
                class_to_patches[k].extend(v)
            class_distribution.update(tile_distribution)
            multi_label_patch_count += tile_multi_label

# --------------------- SPLIT BY CLASS ---------------------
print(f"\nTotal patches collected: {sum(len(p) for p in class_to_patches.values())}")
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
        random_state=SEED
    )
    balanced_splits["train"].extend(train)
    balanced_splits["val"].extend(val)

# --------------------- SAVE PATCHES ---------------------
for split, items in balanced_splits.items():
    for i, (image, label) in enumerate(items):
        out_path = os.path.join(OUTPUT_DIR, split, f"patch_{split}_{i:05d}.npz")
        np.savez_compressed(out_path, image=image.astype(np.float32), mask=label.astype(np.uint8))

print("\nBalanced patches saved to:", OUTPUT_DIR)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
'''
