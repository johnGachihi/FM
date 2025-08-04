import os
import json
import timeit
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import ee
from google.oauth2 import service_account
import multiprocessing
# Start timing
start_time = timeit.default_timer()

# ----------------- CONFIGURATION -------------------
root = '/cluster/archiving/GIZ/data/'
SAVE_DIR = os.path.join(root, "test")
SPLIT_RATIOS = {"train": 0.7, "val": 0.3}
PATCH_SIZE = 8
IGNORE_VALUE = 255
VALID_LABELS = [0, 1, 2, 3]
PATCH_STEP_METERS = PATCH_SIZE * 10  # Assuming 10m resolution
Class = 'code'
gee_projectid = "cropmapping-365811"
asset_pairs = [
    ('projects/cropmapping-365811/assets/rwanda/Nyagatare_B2025_v2', f'{root}shapefiles/Nyagatare_B2019.shp'),
]
NUM_WORKERS  = multiprocessing.cpu_count() - 10
# ----------------- EARTH ENGINE SETUP -------------------
def get_ee_credentials():
    service_account_path = Path("private_key.json")
    if service_account_path.exists():
        with open(service_account_path) as f:
            key_json = json.load(f)
        service_account_email = key_json["client_email"]
        print(f"Logging into Earth Engine using service account: {service_account_email}")
        return ee.ServiceAccountCredentials(
            service_account_email,
            key_file=str(service_account_path)
        )
    else:
        raise FileNotFoundError(f"Service account file not found at: {service_account_path}")

ee.Initialize(credentials=get_ee_credentials(), project=gee_projectid)

# ----------------- UTILITY FUNCTIONS -------------------
def meters_to_deg(m):
    return m / 111320.0

def get_bbox(geom):
    coords = geom.bounds().coordinates().get(0).getInfo()
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)

def load_shapefile_as_fc(path):
    gdf = gpd.read_file(path)
    return ee.FeatureCollection(json.loads(gdf.to_json()))

def extract_from_polygon(poly, image, step_deg, patch_size, ignore_value, bands):
    # Re-initialize EE inside thread if not already done
    if not ee.data._initialized:
        ee.Initialize(credentials=get_ee_credentials(), project=gee_projectid)

    region = ee.Geometry(poly['geometry'])
    label = poly['label']
    min_lon, min_lat, max_lon, max_lat = get_bbox(region)

    patches = []
    lat_steps = int(np.ceil((max_lat - min_lat) / step_deg))
    lon_steps = int(np.ceil((max_lon - min_lon) / step_deg))

    for i in range(lat_steps):
        for j in range(lon_steps):
            patch_geom = ee.Geometry.Rectangle([
                min_lon + j * step_deg,
                min_lat + i * step_deg,
                min_lon + (j + 1) * step_deg,
                min_lat + (i + 1) * step_deg
            ])
            try:
                patch = image.sampleRectangle(region=patch_geom, defaultValue=ignore_value).getInfo()
                if patch:
                    bands_data = []
                    for b in bands:
                        arr = np.array(patch.get(b, [[ignore_value]*patch_size]*patch_size))
                        if arr.shape != (patch_size, patch_size):
                            arr = np.full((patch_size, patch_size), ignore_value)
                        bands_data.append(arr)
                    stacked = np.stack(bands_data, axis=-1)
                    patches.append({'data': stacked, 'label': label})
            except Exception as e:
                print(f"❌ Patch error: {e}")
                continue

    return patches

# ----------------- MAIN EXTRACTION FUNCTION -------------------
def extract_patches(image_asset_id, shapefile_path, patch_size=PATCH_SIZE):
    print(f"\n📦 Processing: {image_asset_id} with labels from {shapefile_path}")
    
    image = ee.Image(image_asset_id)  # DO NOT SELECT BANDS
    bands = image.bandNames().getInfo()  # Get all available bands
    
    label_fc = load_shapefile_as_fc(shapefile_path)
    features = label_fc.toList(label_fc.size())
    
    polygon_info = []
    for i in range(label_fc.size().getInfo()):
        feat = ee.Feature(features.get(i))
        code = feat.get('code').getInfo()
        if code in VALID_LABELS:
            polygon_info.append({'geometry': feat.geometry(), 'label': code})

    step_deg = meters_to_deg(PATCH_STEP_METERS)
    patches_by_class = defaultdict(list)
    polygons_with_patches = 0
    total_polygons = len(polygon_info)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(
                extract_from_polygon,
                poly,
                image,
                step_deg,
                patch_size,
                IGNORE_VALUE,
                bands  # <-- now dynamic
            )
            for poly in polygon_info
        ]

        for future in tqdm(as_completed(futures), total=total_polygons, desc="Extracting patches"):
            try:
                patch_list = future.result()
                if patch_list:
                    polygons_with_patches += 1
                    for p in patch_list:
                        patches_by_class[p['label']].append(p['data'])
            except Exception as e:
                print(f"❌ Error during patch extraction: {e}")

    total_patches = sum(len(v) for v in patches_by_class.values())
    print(f"\n✅ Total polygons processed: {total_polygons}")
    print(f"✅ Polygons with at least one patch: {polygons_with_patches}")
    print(f"✅ Total patches collected: {total_patches}")
    print("✅ Patch distribution per class:")
    for cls in VALID_LABELS:
        print(f"  Class {cls}: {len(patches_by_class[cls])} patches")

    return patches_by_class

# ----------------- MAIN EXECUTION -------------------
all_results = {}

for image_asset, shapefile_path in asset_pairs:
    result = extract_patches(image_asset, shapefile_path)
    key = os.path.basename(image_asset) if '/' not in image_asset else image_asset.split('/')[-1]
    all_results[key] = result

# ----------------- SPLIT & SAVE -------------------
for split in SPLIT_RATIOS:
    os.makedirs(os.path.join(SAVE_DIR, split), exist_ok=True)

for asset_name, class_dict in all_results.items():
    for cls, patch_list in class_dict.items():
        if len(patch_list) < 2:
            continue

        train_patches, val_patches = train_test_split(
            patch_list,
            train_size=SPLIT_RATIOS["train"],
            test_size=SPLIT_RATIOS["val"],
            random_state=42
        )

        for split, split_patches in zip(["train", "val"], [train_patches, val_patches]):
            for idx, patch_data in enumerate(split_patches):
                patch_filename = f"{asset_name}_class{cls}_patch{idx:04d}.npz"
                patch_path = os.path.join(SAVE_DIR, split, patch_filename)
                np.savez_compressed(patch_path, image=patch_data, label=cls)

# ----------------- DONE -------------------
print("\n✅ Balanced patches saved to:", SAVE_DIR)
print("✅ Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)


'''
import os
import ee
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ee.Initialize()

# PARAMETERS
PATCH_SIZE = 8
IGNORE_VALUE = 255
VALID_LABELS = [0, 1, 2, 3]
PATCH_STEP_METERS = PATCH_SIZE * 10  # Assuming 10m pixels

# ASSET LIST — Update these with your actual assets
asset_pairs = [
    ('users/your_username/S1_mosaic_B2025', 'users/your_username/Nyagatare_B2019'),
    ('users/your_username/S1_mosaic_A2025', 'users/your_username/Nyagatare_A2019'),
    # Add more pairs as needed
]

# Convert meters to approximate degrees for 10m resolution (near equator)
def meters_to_deg(m):
    return m / 111320.0

def get_bbox(geom):
    coords = geom.bounds().coordinates().get(0).getInfo()
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)

def extract_patches(image_asset, shapefile_asset, patch_size=PATCH_SIZE):
    print(f"\n📦 Processing: {image_asset} with labels from {shapefile_asset}")

    image = ee.Image(image_asset).select(['vv', 'vh', 'ndpi', 'nli', 'si'])
    label_fc = ee.FeatureCollection(shapefile_asset)
    features = label_fc.toList(label_fc.size())
    
    polygon_info = []
    for i in range(label_fc.size().getInfo()):
        feat = ee.Feature(features.get(i))
        code = feat.get('code').getInfo()
        if code in VALID_LABELS:
            polygon_info.append({'geometry': feat.geometry(), 'label': code})

    step_deg = meters_to_deg(PATCH_STEP_METERS)
    patches_by_class = defaultdict(list)
    polygons_with_patches = 0
    total_polygons = len(polygon_info)

    def extract_from_polygon(poly):
        region = ee.Geometry(poly['geometry'])
        label = poly['label']
        min_lon, min_lat, max_lon, max_lat = get_bbox(region)

        patches = []
        lat_steps = int(np.ceil((max_lat - min_lat) / step_deg))
        lon_steps = int(np.ceil((max_lon - min_lon) / step_deg))

        for i in range(lat_steps):
            for j in range(lon_steps):
                patch_geom = ee.Geometry.Rectangle([
                    min_lon + j * step_deg,
                    min_lat + i * step_deg,
                    min_lon + (j + 1) * step_deg,
                    min_lat + (i + 1) * step_deg
                ])
                patch = image.sampleRectangle(region=patch_geom, defaultValue=IGNORE_VALUE).getInfo()
                if patch:
                    bands = []
                    for b in ['vv', 'vh', 'ndpi', 'nli', 'si']:
                        arr = np.array(patch.get(b, [[IGNORE_VALUE]*patch_size]*patch_size))
                        if arr.shape != (patch_size, patch_size):
                            arr = np.full((patch_size, patch_size), IGNORE_VALUE)
                        bands.append(arr)
                    stacked = np.stack(bands, axis=-1)
                    patches.append({'data': stacked, 'label': label})
        return patches

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(extract_from_polygon, poly) for poly in polygon_info]
        for future in tqdm(as_completed(futures), total=total_polygons, desc="Extracting patches"):
            try:
                result = future.result()
                if result:
                    polygons_with_patches += 1
                    for p in result:
                        patches_by_class[p['label']].append(p['data'])
            except Exception as e:
                print(f"Error: {e}")

    # Summary
    total_patches = sum(len(v) for v in patches_by_class.values())
    print(f"\n✅ Total polygons processed: {total_polygons}")
    print(f"✅ Polygons with at least one patch: {polygons_with_patches}")
    print(f"✅ Total patches collected: {total_patches}")
    print("✅ Patch distribution per class:")
    for cls in VALID_LABELS:
        print(f"  Class {cls}: {len(patches_by_class[cls])} patches")

    return patches_by_class


# ========== MAIN LOOP ==========
all_results = {}

for image_asset, shapefile_asset in asset_pairs:
    result = extract_patches(image_asset, shapefile_asset)
    key = os.path.basename(image_asset)
    all_results[key] = result

# Example: Accessing patches for one of the assets
# e.g., all_results['S1_mosaic_B2025'][0] → list of class 0 patches

# Optional save per asset
# for asset_name, class_dict in all_results.items():
#     np.savez_compressed(f"{asset_name}_patches_{PATCH_SIZE}x{PATCH_SIZE}.npz", **class_dict)
'''