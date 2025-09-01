from datetime import date

import ee

from .utils import date_to_string, get_closest_dates

image_collection = "IDAHO_EPSCOR/TERRACLIMATE"
TC_BANDS = ["def", "soil", "aet"]
TC_SHIFT_VALUES = [0.0, 0.0, 0.0]
TC_DIV_VALUES = [4548, 8882, 2000]
NODATA_VALUE = -9999  # Define NODATA value to match patch extraction script

def get_terraclim_image_collection(
    region: ee.Geometry, start_date: date, end_date: date
) -> ee.ImageCollection:
    dates = ee.DateRange(date_to_string(start_date), date_to_string(end_date))

    startDate = ee.DateRange(dates).start()  # type: ignore
    endDate = ee.DateRange(dates).end()  # type: ignore

    return ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)


def get_single_terraclimate_image(
    region: ee.Geometry,
    start_date: date,
    end_date: date,
    tc_imcol: ee.ImageCollection,
) -> ee.Image:
    mid_date = start_date + ((end_date - start_date) / 2)

    # Select the image closest to the mid_date, clip to region, and select bands
    kept_tc = get_closest_dates(mid_date, tc_imcol).first().clip(region).select(TC_BANDS)
    
    # Replace masked pixels with NODATA_VALUE (-9999)
    kept_tc = kept_tc.unmask(NODATA_VALUE)
    
    return kept_tc.toDouble()
