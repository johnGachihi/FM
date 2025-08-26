import timeit
start_time = timeit.default_timer()

import time
import json
import os
from pathlib import Path
from datetime import date, datetime, timedelta

import geopandas as gpd
import ee

# Optional custom exporter (falls back to ee.batch.Export if missing)
try:
    from src.data.earthengine.eo import EarthEngineExporter
    _HAS_EXPORTER = True
except Exception:
    _HAS_EXPORTER = False

print('Necessary libraries imported', flush=True)

# ------------------------------------------------------------------------------
# Parameters (edit as needed)
# ------------------------------------------------------------------------------
root = '/cluster/archiving/GIZ/data/'
filename = 'PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONB'

district_level = False          # True: export single district AOI; False: per-polygon from shapefile
district = 'Ruhango'            # used only if district_level=True

syear = 2021
eyear = 2021
season = 'B'

# Keep EXACT dates as provided (Python date objects)
start_py = date(syear, 2, 1)          # inclusive
end_py   = date(eyear, 6, 30)         # inclusive (kept exactly)

IGNORE_VALUE = 255                    # polygons with code == 255 are ignored

gee_projectid = "cropmapping-365811"
# Asset folder **without** trailing slash for EE API:
asset_folder = f'projects/{gee_projectid}/assets/{filename}'
task_list_file = f'{root}{filename}_task_ids.json'

# Ensure output directory for task list exists
os.makedirs(os.path.dirname(task_list_file), exist_ok=True)

# ------------------------------------------------------------------------------
# Auth: service account (expects private_key.json in current working dir)
# ------------------------------------------------------------------------------
def get_ee_credentials():
    service_account_path = Path("private_key.json")
    if not service_account_path.exists():
        raise FileNotFoundError(f"Service account file not found at: {service_account_path.resolve()}")
    with service_account_path.open() as f:
        key_json = json.load(f)
    service_account_email = key_json["client_email"]
    print(f"Logging into Earth Engine using service account: {service_account_email}", flush=True)
    return ee.ServiceAccountCredentials(
        service_account_email,
        key_file=str(service_account_path)
    )

ee.Initialize(credentials=get_ee_credentials(), project=gee_projectid)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def ensure_asset_folder(folder_path: str):
    """
    Ensure an EE asset folder exists at folder_path
    (e.g., 'projects/cropmapping-365811/assets/PAPER_2021_RWA_WAPOR_POLY_111_MERGED_SEASONB')
    """
    path = folder_path.rstrip('/')  # no trailing slash

    # If it already exists, we're done.
    try:
        ee.data.getAsset(path)
        print(f"Asset folder exists: {path}", flush=True)
        return
    except ee.ee_exception.EEException:
        # Not found or no access; try to create.
        pass

    # Try new-style payload (Cloud EE: 'name')
    try:
        ee.data.createAsset({'name': path, 'type': 'Folder'})
        print(f"Created asset folder (name): {path}", flush=True)
        return
    except ee.ee_exception.EEException as e1:
        # Fall back to legacy signature (properties, path)
        try:
            ee.data.createAsset({'type': 'Folder'}, path)
            print(f"Created asset folder (path): {path}", flush=True)
            return
        except ee.ee_exception.EEException as e2:
            msg = (str(e1) + " " + str(e2)).lower()
            if 'exist' in msg or 'already exists' in msg:
                print(f"Asset folder already exists: {path}", flush=True)
                return
            raise

def ee_date_range_inclusive(start_d, end_d):
    """
    Return (ee_start, ee_end_exclusive) so that end_d is INCLUDED in filterDate.
    Keeps your exact end date; only the EE end is advanced by 1 day.
    """
    ee_start = ee.Date(start_d.isoformat())
    ee_end_excl = ee.Date(end_d.isoformat()).advance(1, 'day')
    return ee_start, ee_end_excl

def monitor_all_tasks(task_ids, poll_interval=30, max_duration=7200):
    """Monitor a set of EE task IDs until completion or timeout."""
    print("Monitoring all Earth Engine export tasks...", flush=True)
    t0 = time.time()
    remaining = set(task_ids)
    while remaining:
        tasks = ee.data.getTaskList()
        statuses = {}
        for t in tasks:
            tid = t.get('id')
            if tid in remaining:
                state = t.get('state', 'UNKNOWN')
                statuses[state] = statuses.get(state, 0) + 1
                desc = t.get('description', '')
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Task {tid} ({desc}): {state}", flush=True)
                if state in {'COMPLETED', 'FAILED', 'CANCELLED'}:
                    if state == 'FAILED':
                        print(f"  ↳ error: {t.get('error_message', 'Unknown error')}", flush=True)
                    remaining.discard(tid)

        summary = " | ".join(f"{k}:{v}" for k, v in sorted(statuses.items()))
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {summary or 'No matching tasks found'}", flush=True)

        if not any(s in statuses for s in ('PENDING', 'READY', 'RUNNING')):
            print("All monitored tasks finished (or none found).", flush=True)
            break
        if max_duration and (time.time() - t0) > max_duration:
            print("Monitoring stopped: max duration reached.", flush=True)
            break
        time.sleep(poll_interval)
    return statuses

def exporter_export_geojson(exporter, *, geo_json, start_date, end_date, identifier, asset_id=None, asset_folder=None):
    """
    Call your EarthEngineExporter if available.
    Falls back to ee.batch.Export if the exporter is missing or has a different signature.
    Returns a task ID string (if possible).
    """
    # Preferred path: your custom exporter
    if _HAS_EXPORTER:
        try:
            # Pass Python dates so the exporter can do date arithmetic (monthly windows, etc.)
            return exporter.export_for_geo_json(
                geo_json=geo_json,
                start_date=start_date,   # Python date (inclusive)
                end_date=end_date,       # Python date (inclusive)
                identifier=identifier,
                asset_id=asset_id,
                asset_folder=asset_folder,
            )
        except TypeError:
            # Try without asset args if exporter doesn't accept them
            return exporter.export_for_geo_json(
                geo_json=geo_json,
                start_date=start_date,
                end_date=end_date,
                identifier=identifier
            )

    # Fallback path: minimal ee.batch.Export (example using WaPOR AETI mean)
    region = ee.Geometry(geo_json, geodesic=False)
    ee_start, ee_end_excl = ee_date_range_inclusive(start_date, end_date)
    imgcol = (ee.ImageCollection('FAO/WAPOR/2/L1_AETI_D')
              .filterDate(ee_start, ee_end_excl)
              .filterBounds(region))
    image = imgcol.mean().set({'identifier': identifier})
    if asset_id is None:
        asset_id = f"{asset_folder}/{identifier}"
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=identifier,
        assetId=asset_id,
        region=region,
        scale=30,            # set your preferred scale
        maxPixels=1e13
    )
    task.start()
    try:
        return task.id
    except Exception:
        return None

# ------------------------------------------------------------------------------
# Main export logic
# ------------------------------------------------------------------------------
def main():
    ensure_asset_folder(asset_folder)

    if district_level:
        # Export one AOI for a district polygon
        shp_path = f"{root}shapefiles/rwa_adm2_selected_districts.shp"
        gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
        match = gdf[gdf['ADM2_EN'] == district]
        if match.empty:
            raise ValueError(f"{district} not found in ADM2_EN of {shp_path}")
        aoi = match.iloc[0].geometry
        if aoi is None or aoi.is_empty:
            raise ValueError(f"{district} geometry is empty.")
        region_geojson = aoi.__geo_interface__
        identifier = f"{district}_{season}{eyear}"

        exporter = EarthEngineExporter(mode="batch") if _HAS_EXPORTER else None
        # Availability check (EE inclusive end date without changing your end_py)
        region = ee.Geometry(region_geojson, geodesic=False)
        ee_start, ee_end_excl = ee_date_range_inclusive(start_py, end_py)
        ic = (ee.ImageCollection('FAO/WAPOR/2/L1_AETI_D')
              .filterDate(ee_start, ee_end_excl)
              .filterBounds(region))
        if ic.size().getInfo() == 0:
            print(f"No WaPOR data for {identifier}; skipping.", flush=True)
            return

        task_id = exporter_export_geojson(
            exporter=exporter,
            geo_json=region_geojson,
            start_date=start_py,   # keep as Python dates
            end_date=end_py,
            identifier=identifier,
            asset_id=f"{asset_folder}/{identifier}",
            asset_folder=asset_folder,
        )
        if task_id:
            monitor_all_tasks([task_id])
        else:
            print("No task ID returned; you can check GEE Tasks UI for progress.", flush=True)
        return

    # Otherwise: export per polygon from the provided shapefile
    shp_path = f"{root}shapefiles/{filename}.shp"
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)  # Ensure WGS84 CRS
    gdf = gdf[gdf['code'] != IGNORE_VALUE]
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]

    if gdf.empty:
        raise ValueError("No valid geometries remain after filtering.")

    exporter = EarthEngineExporter(mode="batch") if _HAS_EXPORTER else None
    task_ids = []
    failed_polygons = []

    for idx, row in gdf.iterrows():
        polygon_id = f"{filename}_polygon_{row['code']}_{idx}"
        asset_id = f"{asset_folder}/{polygon_id}"

        geom = row.geometry
        if geom.is_empty or (geom.area < 1e-12):
            print(f"{polygon_id}: empty or too small geometry; skipping.", flush=True)
            failed_polygons.append(polygon_id)
            continue

        region = ee.Geometry(geom.__geo_interface__, geodesic=False)

        # Availability check with inclusive end date (without mutating your end_py)
        try:
            ee_start, ee_end_excl = ee_date_range_inclusive(start_py, end_py)
            ic = (ee.ImageCollection('FAO/WAPOR/2/L1_AETI_D')
                  .filterDate(ee_start, ee_end_excl)
                  .filterBounds(region))
            sz = ic.size().getInfo()
            if sz == 0:
                print(f"No WaPOR data for {polygon_id}; skipping.", flush=True)
                failed_polygons.append(polygon_id)
                continue
        except Exception as e:
            print(f"Availability check failed for {polygon_id}: {e}", flush=True)
            failed_polygons.append(polygon_id)
            continue

        print(f"Submitting export for {polygon_id} → {asset_id}", flush=True)
        try:
            tid = exporter_export_geojson(
                exporter=exporter,
                geo_json=geom.__geo_interface__,
                start_date=start_py,   # Python dates for exporter
                end_date=end_py,
                identifier=polygon_id,
                asset_id=asset_id,
                asset_folder=asset_folder,
            )
            if tid:
                task_ids.append(tid)
            else:
                # Fallback: wait and try to find the task by description
                time.sleep(10)
                tasks = ee.data.getTaskList()
                found = next(
                    (t.get('id') for t in tasks
                     if t.get('description') == polygon_id and t.get('state') in ('PENDING', 'READY', 'RUNNING')),
                    None
                )
                if found:
                    task_ids.append(found)
                else:
                    print(f"Could not find task for {polygon_id}.", flush=True)
                    failed_polygons.append(polygon_id)
        except Exception as e:
            print(f"Failed to start export for {polygon_id}: {e}", flush=True)
            failed_polygons.append(polygon_id)

    # Save task IDs to file (even if empty, for traceability)
    with open(task_list_file, 'w') as f:
        json.dump(task_ids, f)
    print(f"Saved {len(task_ids)} task IDs to {task_list_file}", flush=True)

    # Monitor tasks
    if task_ids:
        monitor_all_tasks(task_ids)
    if failed_polygons:
        print(f"Failed polygons ({len(failed_polygons)}): {failed_polygons}", flush=True)

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # (Optional) help PROJ lookups in some Conda envs:
    os.environ["PROJ_LIB"] = os.environ.get("CONDA_PREFIX", "") + "/share/proj"
    main()
    print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0, flush=True)
