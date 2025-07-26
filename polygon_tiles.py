import os
import ee
import geopandas as gpd
import geemap
import json
from shapely.geometry import mapping

# --------------------- EE AUTH ---------------------
def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Using service account: {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Using persistent Earth Engine login")
        return "persistent"

ee.Initialize(**{
    "credentials": get_ee_credentials(),
    "project": "gee-project-368207",
})

# --------------------- SETTINGS ---------------------
GEE_IMAGE_ID = "users/chemuttjose/rwanda_2021_seasonA"
VECTOR_PATH = "data/shapefiles/Nyagatare_A2021.shp"
OUTPUT_DIR = "data/polygon_tiles"
RESOLUTION = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------- LOAD DATA ---------------------
print("Loading image and shapefile...")
image = ee.Image(GEE_IMAGE_ID)
gdf = gpd.read_file(VECTOR_PATH).to_crs("EPSG:4326")

# Filter only Polygon and MultiPolygon
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

# --------------------- DOWNLOAD PER POLYGON ---------------------
print("Processing polygons...")
for i, row in gdf.iterrows():
    geom = row.geometry

    try:
        # Ensure geometry is valid
        if not geom.is_valid:
            geom = geom.buffer(0)

        ee_geom = ee.Geometry(mapping(geom))  # Safe GeoJSON-compatible geometry

        clipped = image.clip(ee_geom)

        url = clipped.getDownloadURL({
            "region": ee_geom,
            "scale": RESOLUTION,
            "format": "GeoTIFF"
        })

        out_path = os.path.join(OUTPUT_DIR, f"tile_{i:04d}.tif")
        print(f"[{i}] Downloading tile to {out_path}...")
        geemap.download_file(url, out_path)

    except Exception as e:
        print(f"[{i}] Skipped due to error: {e}")
