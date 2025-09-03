# ------------------------------------------------------------
# Author:  Benson Kenduiywo
# Check to ensure that all tiles are downloaded sucessfully
# ------------------------------------------------------------
# ------------------------------------------------------------
import timeit
start_time = timeit.default_timer()
# -------------------- USER PARAMS --------------------
season = 'A'
eyear = 2021
district = 'Nyagatare'
filename = f'{district}_{season}{eyear}'
gee_projectpath = "projects/cropmapping-365811"
gee_projectid = "cropmapping-365811"
asset_folder = "rwanda"
image_asset_id = f"{gee_projectpath}/assets/{asset_folder}/{filename}"
root = '/cluster/archiving/GIZ/data/'

OUTPUT_DIR = f'{root}{filename}_tiles'
TILE_SIZE_PIXELS = 256     # px
RESOLUTION = 10            # m per px
CRS = "EPSG:4326"

MAX_WORKERS = 6            # parallel threads (start with 4–8)
MAX_RETRIES = 4            # per tile
BASE_SLEEP = 3.0           # seconds; backoff = BASE_SLEEP * (2**(attempt-1))
SKIP_EXISTING_VALID = True
NODATA_VALUE = -9999
# -------------------- IMPORTS --------------------
import os
import json
import time
import math
import ee
import geemap
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError
from shapely.geometry import box
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# -------------------- EE AUTH --------------------
from google.oauth2 import service_account

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

# -------------------- UTILS --------------------
def is_tiff_magic(path, nbytes=4):
    try:
        with open(path, 'rb') as f:
            sig = f.read(nbytes)
        return sig in (b'II*\x00', b'MM\x00*')
    except Exception:
        return False

def validate_geotiff(path):
    """Exists, non-trivial size, TIFF magic, rasterio can open/read, has CRS & bands."""
    if not os.path.exists(path):
        return False, "missing"
    if os.path.getsize(path) < 512:
        return False, "too_small"
    if not is_tiff_magic(path):
        return False, "bad_magic"
    try:
        with rasterio.open(path) as src:
            if src.count < 1:
                return False, "no_bands"
            if src.width <= 0 or src.height <= 0:
                return False, "bad_dims"
            if src.crs is None:
                return False, "no_crs"
            # Try reading a pixel window
            _ = src.read(1, window=((0, min(1, src.height)), (0, min(1, src.width))))
    except RasterioIOError as e:
        return False, f"rio_io:{e}"
    except Exception as e:
        return False, f"rio:{e}"
    return True, "ok"

def exponential_backoff_sleep(attempt):
    # attempt = 1..MAX_RETRIES
    time.sleep(BASE_SLEEP * (2 ** (attempt - 1)))

# Single global image object (safe for URL generation; keep workers modest)
image = ee.Image(image_asset_id)

# Thread-safe print (tqdm is mostly fine, but keep logs tidy)
print_lock = Lock()

def log(msg):
    with print_lock:
        print(msg, flush=True)

def download_with_retries(tile_id, ee_image, ee_geom, out_path, scale, crs):
    last_err = None
    clipped  = ee_image.unmask(NODATA_VALUE).clip(ee_geom)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Fresh URL each attempt
            url = clipped.getDownloadURL({
                "region": ee_geom,
                "scale": scale,
                "format": "GeoTIFF",
                "crs": crs,
            })
            geemap.download_file(url, out_path, overwrite=True)

            ok, reason = validate_geotiff(out_path)
            if ok:
                return True, "ok"
            else:
                log(f"[{tile_id:04d}] Validation failed ({reason}) attempt {attempt}/{MAX_RETRIES}")
                try:
                    os.remove(out_path)
                except Exception:
                    pass
        except Exception as e:
            last_err = str(e)
            log(f"[{tile_id:04d}] Download error attempt {attempt}/{MAX_RETRIES}: {e}")
        if attempt < MAX_RETRIES:
            exponential_backoff_sleep(attempt)
    return False, last_err or "unknown_error"

def process_tile(tile_row):
    """Worker for a single tile; returns (tile_id, success, message)."""
    tile_id, geom = tile_row
    out_path = os.path.join(OUTPUT_DIR, f"tile_{tile_id:04d}.tif")

    # Skip valid existing
    if SKIP_EXISTING_VALID and os.path.exists(out_path):
        ok, reason = validate_geotiff(out_path)
        if ok:
            return tile_id, True, "exists_valid"
        else:
            log(f"[{tile_id:04d}] Found invalid existing file ({reason}); re-downloading.")

    # Build EE geometry (non-geodesic polygon)
    try:
        coords = list(geom.exterior.coords)
        ee_geom = ee.Geometry.Polygon(coords, None, False)
    except Exception as e:
        return tile_id, False, f"invalid_geom:{e}"

    clipped = image.clip(ee_geom)
    ok, msg = download_with_retries(tile_id, clipped, ee_geom, out_path, RESOLUTION, CRS)
    return tile_id, ok, msg

# -------------------- PREP GRID --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bounds (lon/lat) from image footprint
bounds = image.geometry().bounds().getInfo()["coordinates"][0]
min_lon = min(p[0] for p in bounds)
max_lon = max(p[0] for p in bounds)
min_lat = min(p[1] for p in bounds)
max_lat = max(p[1] for p in bounds)

# Approx tile size in degrees (EPSG:4326)
tile_deg = (TILE_SIZE_PIXELS * RESOLUTION) / 111320.0

# Generate grid indices + shapely boxes
print("Generating grid...")
tiles = []
y = min_lat
tile_id = 0
while y < max_lat:
    x = min_lon
    while x < max_lon:
        tiles.append((tile_id, box(x, y, x + tile_deg, y + tile_deg)))
        tile_id += 1
        x += tile_deg
    y += tile_deg

print(f"Total tiles: {len(tiles)}", flush=True)

# -------------------- PARALLEL DOWNLOAD --------------------
success = 0
failed = 0
skipped = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(process_tile, t): t[0] for t in tiles}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
        tid = futures[fut]
        try:
            tile_id, ok, msg = fut.result()
            if ok:
                if msg == "exists_valid":
                    skipped += 1
                else:
                    success += 1
            else:
                failed += 1
                log(f"[{tile_id:04d}] FAILED: {msg}")
        except Exception as e:
            failed += 1
            log(f"[{tid:04d}] EXCEPTION: {e}")

print(f"Summary → success: {success}, skipped(valid): {skipped}, failed: {failed}", flush=True)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0, flush=True)
