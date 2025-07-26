# run_export_rwanda.py
import timeit
start_time = timeit.default_timer()
import time
from datetime import date
import geopandas as gpd
import ee
from src.data.earthengine.eo import EarthEngineExporter

print('Necessary libraries imported')

# Parameters
root = '/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/'
district = 'Ruhango'
syear = 2025
eyear = 2025
season = 'B'
start = date(syear, 2, 1)
end = date(eyear, 6, 30)

# Load the shapefile
gdf = gpd.read_file(f"{root}shapefiles/rwa_adm2_selected_districts.shp")
nyagatare_gdf = gdf[gdf['ADM2_EN'] == district]

if nyagatare_gdf.empty:
    raise ValueError("Nyagatare not found in ADM2_EN column.")

# Extract geometry in GeoJSON format
aoi_geojson_geometry = nyagatare_gdf.iloc[0].geometry.__geo_interface__

# Initialize Earth Engine exporter
exporter = EarthEngineExporter(mode="batch")

def monitor_ee_tasks(poll_interval=30, max_duration=None):
    """
    Continuously monitor Earth Engine export tasks.
    
    Args:
        poll_interval (int): Time in seconds between each poll.
        max_duration (int or None): Maximum monitoring time in seconds (optional).
    """
    print("⏳ Monitoring Earth Engine export tasks...")
    start_time = time.time()

    while True:
        tasks = ee.data.getTaskList()
        statuses = {"RUNNING": 0, "READY": 0, "COMPLETED": 0, "FAILED": 0, "CANCELLED": 0}

        for task in tasks:
            state = task["state"]
            if state in statuses:
                statuses[state] += 1

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"RUNNING: {statuses['RUNNING']} | "
              f"READY: {statuses['READY']} | "
              f"COMPLETED: {statuses['COMPLETED']} | "
              f"FAILED: {statuses['FAILED']} | "
              f"CANCELLED: {statuses['CANCELLED']}", flush=True)

        if statuses["RUNNING"] + statuses["READY"] == 0:
            print("✅ All Earth Engine export tasks finished.", flush=True)
            break

        if max_duration is not None and (time.time() - start_time) > max_duration:
            print("⚠️ Monitoring stopped: max duration reached.", flush=True)
            break

        time.sleep(poll_interval)

# Export data for Nyagatare
exporter.export_for_geo_json(
    geo_json=aoi_geojson_geometry,
    start_date=start,
    end_date=end,
    identifier=f"{district}_{season}{eyear}"
)
# Monitor task progress
monitor_ee_tasks(poll_interval=120)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)

'''
from datetime import date
import json
from src.data.earthengine.eo import EarthEngineExporter


with open("data/nyangatare.geojson") as f:
    aoi_geojson = json.load(f)

exporter = EarthEngineExporter(
    mode="batch", 
)

exporter.export_for_geo_json(
    geo_json=aoi_geojson["features"][0]["geometry"],
    start_date=date(2021, 9, 1),
    end_date=date(2022, 2, 28),
    identifier="rwanda_2022_seasonA"
)
'''