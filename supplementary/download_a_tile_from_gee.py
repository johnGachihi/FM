# run_export_rwanda.py
import timeit
start_time = timeit.default_timer()
# -------------------- PARAMETERS --------------------
tile_to_download = 3  # Specify the tile ID you want
season = 'B'
version = 'v2'
eyear = 2025 
district = 'Nyabihu'
filename = f'{district}_{season}{eyear}_{version}'
gee_projectpath = "projects/cropmapping-365811"
gee_projectid = "cropmapping-365811"
asset_folder = "rwanda"
image_asset_id = f"{gee_projectpath}/assets/{asset_folder}/{filename}"
#root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/' #
root = '/cluster/archiving/GIZ/data/'


# -------------------- EE AUTH --------------------
import os
import json
import ee
import geemap
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
from pathlib import Path

from google.oauth2 import service_account

def get_ee_credentials():
    service_account_path = Path("private_key.json")  # or wherever your key file is
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

# Initialize EE with service account credentials
ee.Initialize(
    credentials=get_ee_credentials(),
    project=gee_projectid #"cropmapping-365811"
)


# -------------------- CONFIG --------------------
GEE_IMAGE_ID = image_asset_id
OUTPUT_DIR = f'{root}{filename}_tiles'#f"[root]data/rwanda_2022_seasonA_tiles"
TILE_SIZE_PIXELS = 256  # Fixed tile size
RESOLUTION = 10  # Meters per pixel
CRS = "EPSG:4326"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- LOAD IMAGE --------------------
image = ee.Image(GEE_IMAGE_ID)
bounds = image.geometry().bounds().getInfo()["coordinates"][0]

# Get bounding box (in lon/lat)
min_lon = min([p[0] for p in bounds])
max_lon = max([p[0] for p in bounds])
min_lat = min([p[1] for p in bounds])
max_lat = max([p[1] for p in bounds])

# Convert TILE_SIZE_PIXELS to degrees (roughly for EPSG:4326)
tile_deg = (TILE_SIZE_PIXELS * RESOLUTION) / 111320

# -------------------- GENERATE GRID --------------------
print("Generating grid...")
tiles = []
y = min_lat
tile_id = 0
while y < max_lat:
    x = min_lon
    while x < max_lon:
        tile_geom = box(x, y, x + tile_deg, y + tile_deg)
        tiles.append((tile_id, tile_geom))
        tile_id += 1
        x += tile_deg
    y += tile_deg

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(tiles, columns=["tile_id", "geometry"], crs=CRS)

# -------------------- DOWNLOAD TILES --------------------
print(f"Downloading tile {tile_to_download}...")
for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
    if row.tile_id != tile_to_download:
        continue  # Skip all other tiles

    tile_geom = row.geometry
    coords = tile_geom.exterior.coords[:]
    ee_geom = ee.Geometry.Polygon(coords)

    clipped = image.clip(ee_geom)

    try:
        url = clipped.getDownloadURL({
            "region": ee_geom,
            "scale": RESOLUTION,
            "format": "GeoTIFF"
        })

        out_path = os.path.join(OUTPUT_DIR, f"tile_{row.tile_id:04d}.tif")
        geemap.download_file(url, out_path, overwrite=True)
        print(f"[{row.tile_id}] Downloaded successfully: {out_path}")
    except Exception as e:
        print(f"[{row.tile_id}] Skipped due to error: {e}")

print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
