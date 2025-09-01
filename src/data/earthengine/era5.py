from datetime import date

import ee

from .utils import date_to_string, get_monthly_data

image_collection = "ECMWF/ERA5_LAND/MONTHLY_AGGR"
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
# For temperature, shift to Celsius and divide by 35 (ranges from -22 to 37°C)
# For rainfall, based on exploratory data analysis
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]
NODATA_VALUE = -9999  # Define NODATA value to match patch extraction script

def get_single_era5_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    # Pass NODATA_VALUE to get_monthly_data for unmasking
    return get_monthly_data(
        image_collection,
        ERA5_BANDS,
        region,
        start_date,
        unmask=True,
        unmask_value=NODATA_VALUE
    ).toDouble()

# Fallback implementation of get_monthly_data (if not defined in utils)
def get_monthly_data(
    collection_id: str,
    bands: list,
    region: ee.Geometry,
    start_date: date,
    unmask: bool = False,
    unmask_value: float = 0
) -> ee.Image:
    """
    Fallback implementation to fetch a single image from a collection for a given date and region.
    """
    start_date_str = date_to_string(start_date)
    end_date = date(start_date.year, start_date.month + 1, 1) if start_date.month < 12 else date(start_date.year + 1, 1, 1)
    end_date_str = date_to_string(end_date)

    img_col = (
        ee.ImageCollection(collection_id)
        .filterDate(start_date_str, end_date_str)
        .filterBounds(region)
        .select(bands)
    )
    
    # Take the first image (assuming monthly data)
    img = img_col.first().clip(region)
    
    # Apply unmasking if requested
    if unmask:
        img = img.unmask(unmask_value)
    
    return img