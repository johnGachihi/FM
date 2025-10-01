from datetime import date

import ee

image_collection = "USGS/SRTMGL1_003"
SRTM_BANDS = ["elevation", "slope"]
# Visually gauged 90th percentile from exploratory data analysis
SRTM_SHIFT_VALUES = [0.0, 0.0]
SRTM_DIV_VALUES = [2000.0, 50.0]
NODATA_VALUE = -9999  # Define NODATA value to match patch extraction script

def get_single_srtm_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    elevation = ee.Image(image_collection).clip(region).select(SRTM_BANDS[0])
    slope = ee.Terrain.slope(elevation)  # type: ignore
    together = ee.Image.cat([elevation, slope]).toDouble()  # type: ignore
    
    # Ensure masked pixels are set to NODATA_VALUE (-9999)
    together = together.unmask(NODATA_VALUE)

    return together
