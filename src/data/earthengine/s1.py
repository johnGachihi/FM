from datetime import date
from typing import Tuple

import ee

from .utils import date_to_string, get_closest_dates

image_collection = "COPERNICUS/S1_GRD"
S1_BANDS = ["VV", "VH"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]
NODATA_VALUE = -9999  # Define NODATA value to match patch extraction script

def get_s1_image_collection(
    region: ee.Geometry, start_date: date, end_date: date
) -> Tuple[ee.ImageCollection, ee.ImageCollection]:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()  # type: ignore
    endDate = ee.DateRange(dates).end()  # type: ignore

    s1 = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    # Different areas have either ascending, descending coverage or both.
    # We want consistent coverage, so select based on the orbit of the first image.
    orbit = s1.filter(
        ee.Filter.eq("orbitProperties_pass", s1.first().get("orbitProperties_pass"))
    ).filter(ee.Filter.eq("instrumentMode", "IW"))

    return (
        orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")),
        orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")),
    )


def get_single_s1_image(
    region: ee.Geometry,
    start_date: date,
    end_date: date,
    vv_imcol: ee.ImageCollection,
    vh_imcol: ee.ImageCollection,
) -> ee.Image:
    mid_date = start_date + ((end_date - start_date) / 2)

    kept_vv = get_closest_dates(mid_date, vv_imcol)
    kept_vh = get_closest_dates(mid_date, vh_imcol)

    composite = ee.Image.cat(
        [
            kept_vv.select("VV").median(),
            kept_vh.select("VH").median(),
        ]
    ).clip(region)

    # Rename to the bands and ensure masked pixels are set to NODATA_VALUE (-9999)
    final_composite = composite.select(S1_BANDS).unmask(NODATA_VALUE)

    return final_composite