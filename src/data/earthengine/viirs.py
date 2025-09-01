import warnings
from datetime import date

import ee

from .utils import date_to_string, get_monthly_data

VIIRS_URL = (
    "https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG"
)
image_collection = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
VIIRS_BANDS = ["avg_rad"]
VIIRS_SHIFT_VALUES = [0.0]
VIIRS_DIV_VALUES = [100]
NODATA_VALUE = -9999  # Define NODATA value to match patch extraction script
LATEST_START_DATE = date(2024, 5, 4)

def get_single_viirs_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    if (start_date.year == 2023) & (start_date.month == 10):
        # For some reason, VIIRS data for October 2023 is missing
        # so we replace it with November 2023 data
        start_date = date(start_date.year, 11, 1)
    elif start_date > LATEST_START_DATE:
        warnings.warn(
            f"No data for {start_date} (check {VIIRS_URL} to see if this can be updated). "
            f"Defaulting to latest date of {LATEST_START_DATE} checked on October 28 2024"
        )
        start_date = LATEST_START_DATE

    # Pass NODATA_VALUE to get_monthly_data for unmasking
    return get_monthly_data(
        image_collection,
        VIIRS_BANDS,
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