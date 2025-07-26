'''
Author: Benson Kenduiywo
The script separate polygons into training and validation sets using stratified random sampling, while ensuring a minimum XX m separation between training and validation polygons of the same class.
'''
import timeit, os
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split
from pathlib import Path

start_time = timeit.default_timer()

# ───────────── USER CONFIGURATION ─────────────
out_dir   = Path("/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/shapefiles")
shp_path  = "/cluster01/Projects/USA_IDA_AICCRA/1.Data/FINAL/Galileo/data/shapefiles/RWA_B2025_Merge_v2.shp"

train_frac   = 0.70
buffer_dist  = 0.001# approx 111 metres
random_state = 123
classColumn  = "code"
NoData       = 255
#target_crs   = "EPSG:32736"   # UTM‑36S (Rwanda)
# ──────────────────────────────────────────────

# Ensure output folder exists
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load and prepare
# ------------------------------------------------------------------
gdf = gpd.read_file(shp_path)
gdf = gdf[gdf[classColumn] != NoData]           # drop ND polygons
#gdf = gdf.to_crs(target_crs)                    # metres
gdf["geometry"] = gdf.geometry.buffer(0)        # fix invalid rings

# Containers
train_gdfs, val_gdfs = [], []

# ------------------------------------------------------------------
# 2. Stratified split + 100 m separation
# ------------------------------------------------------------------
for cls, group in gdf.groupby(classColumn):
    # shuffle & stratified split
    tr_idx, va_idx = train_test_split(
        group.index,
        train_size=train_frac,
        random_state=random_state,
        shuffle=True,
    )
    train_polys = group.loc[tr_idx].copy()
    val_polys   = group.loc[va_idx].copy()

    # Enforce 100 m gap ‑ only if we have a non‑empty union
    if not train_polys.empty:
        train_union = unary_union(train_polys.geometry)
        if train_union and not train_union.is_empty:
            train_union = train_union.buffer(buffer_dist)
            mask_close  = val_polys.geometry.intersects(train_union)
            moved       = val_polys[mask_close]
            val_polys   = val_polys[~mask_close]
            train_polys = pd.concat([train_polys, moved])

    train_gdfs.append(train_polys)
    val_gdfs.append(val_polys)

# ------------------------------------------------------------------
# 3. Concatenate results
# ------------------------------------------------------------------
if not train_gdfs:
    raise RuntimeError("No polygons ended up in the training set!")
if not val_gdfs:
    raise RuntimeError("No polygons ended up in the validation set!")

train_res = gpd.GeoDataFrame(pd.concat(train_gdfs, ignore_index=True))
val_res   = gpd.GeoDataFrame(pd.concat(val_gdfs,   ignore_index=True))

# ------------------------------------------------------------------
# 4. Stats
# ------------------------------------------------------------------
summary = (
    pd.DataFrame({
        "train": train_res[classColumn].value_counts().sort_index(),
        "valid": val_res[classColumn].value_counts().sort_index(),
    })
    .fillna(0)
    .astype(int)
)
print("\nPolygons per class (train vs. validation):")
print(summary.to_string())

# ------------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------------
base = Path(shp_path).stem
train_path = out_dir / f"{base}_TrainSet.shp"
val_path   = out_dir / f"{base}_ValidSet.shp"

train_res.to_file(train_path)
val_res.to_file(val_path)

print(f"\n[✓] Training polygons  → {train_path}")
print(f"[✓] Validation polygons → {val_path}")
print("Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)
