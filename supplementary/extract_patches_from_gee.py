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
