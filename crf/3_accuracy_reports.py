# ------------------------------------------------------------
# Author: Benson Kenduiywo
# Accuracy assessment using pre-rasterized reference labels
# ------------------------------------------------------------
import os
from glob import glob

import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

import numpy as np
import geopandas as gpd
import pandas as pd

from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
# ------------------------------
# User Inputs
# ------------------------------
districts = ["Ruhango", "Nyagatare", "Musanze", "Nyabihu"]
eyear = 2025
season = "B"
file_ending = 'with_2025'
root = '/cluster/archiving/GIZ/data/'
outpath = f'/cluster/archiving/GIZ/accuracy/'
labels_path = '/cluster/archiving/GIZ/data/outputs/CRF/'
vector_path = f'{root}shapefiles/RWA_{season}{eyear}_Merge_v2_ValidSet.shp'

vector_base = os.path.splitext(os.path.basename(vector_path))[0]
reference_raster_path = f'{outpath}{vector_base}.tif'

prob_threshold = 0.4
out_xlsx = f'{outpath}CRF_Metrics_{file_ending}_threshold_{prob_threshold}.xlsx'
out_f1sc = f'{outpath}CRF_F1Score_{file_ending}_threshold_{prob_threshold}.png'
out_cmat = f'{outpath}CRF_Confmat_{file_ending}_threshold_{prob_threshold}.png'

# IMPORTANT: valid_classes define your class IDs (in band order if using probabilities)
valid_classes = [0, 1, 2, 3]
class_labels = ['Bean', 'Irish Potato', 'Maize', 'Rice']  # must align with valid_classes
nodata_value = 255
N_WORKERS = max(1, (os.cpu_count() or 2) - 2)

# ------------------------------
# Boolean flag — rasterize once or load existing
# ------------------------------
RASTERIZE_REFERENCE = False

os.makedirs(outpath, exist_ok=True)

def find_tiles_for_district(d: str, base: str):
    patterns = [
        f"{base}{d}/{d}_crf_prob_tile_*.tif", # e.g. /CRF/Ruhango/Ruhango_crf_label_tile_0001.tif   
    ]
    hits = []
    for pat in patterns:
        found = glob(pat)
        if found:
            hits.extend(found)
            break  # stop at first layout that matches
    hits = sorted(set(hits))
    print(f"[tile search] {d}: {len(hits)} files found (e.g. {hits[0] if hits else 'NONE'})", flush=True)
    return hits

# ------------------------------
# Merge Raster labels (uses the new pattern)
# ------------------------------
tile_paths = []
for d in districts:
    pats = find_tiles_for_district(d, labels_path)
    if not pats:
        raise FileNotFoundError(
            f"No tiles found for district '{d}'. "
            f"Expected pattern like: {labels_path}{d}_crf_label_tile_*.tif"
        )
    tile_paths.extend(pats)

if not tile_paths:
    raise FileNotFoundError("No tiles found for any district. Check labels_path and patterns.")

srcs = [rasterio.open(p) for p in tile_paths]
try:
    mosaic, transform = merge(srcs, nodata=nodata_value)
    crs = srcs[0].crs
    height, width = mosaic.shape[1], mosaic.shape[2]
finally:
    for s in srcs:
        s.close()

# mosaic shape: (bands, H, W)
# If bands == 1 => hard labels; if bands == len(valid_classes) => per-class probabilities.

# ------------------------------
# Rasterize reference polygons OR load an existing reference raster aligned to mosaic grid
# ------------------------------
if RASTERIZE_REFERENCE:
    print("Rasterizing reference polygons...", flush=True)
    gdf = gpd.read_file(vector_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[gdf.geometry.notnull()]
    # Expect label column named "code" matching valid_classes; change if needed
    label_col = "code"
    if label_col not in gdf.columns:
        raise KeyError(f"Expected column '{label_col}' in {vector_path}")

    gdf = gdf[gdf[label_col].isin(valid_classes)]
    shapes = ((geom, int(code)) for geom, code in zip(gdf.geometry, gdf[label_col]))

    ref_rasterized = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata_value,
        dtype="uint8",
        all_touched=False,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(reference_raster_path, "w", **profile) as dst:
        dst.write(ref_rasterized, 1)
    print(f"Reference raster saved: {reference_raster_path}", flush=True)

# Load (and align if needed)
with rasterio.open(reference_raster_path) as ref_src:
    if (ref_src.crs == crs) and (ref_src.transform == transform) and \
       (ref_src.width == width) and (ref_src.height == height):
        ref_labels = ref_src.read(1)
    else:
        print("Reprojecting reference raster to mosaic grid...", flush=True)
        ref_labels = np.full((height, width), nodata_value, dtype=np.uint8)
        reproject(
            source=rasterio.band(ref_src, 1),
            destination=ref_labels,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=nodata_value,
            resampling=Resampling.nearest,
            num_threads=N_WORKERS,
        )

# ------------------------------
# Predict map from mosaic
# ------------------------------
print(f"Processing threshold: {prob_threshold}", flush=True)

if mosaic.shape[0] == 1:
    # Hard labels
    predicted_map = mosaic[0].astype(np.int16)
else:
    # Multi-band probabilities — assume bands follow valid_classes order
    probs = mosaic.astype(np.float32)
    argmax_idx = np.argmax(probs, axis=0).astype(np.int16)  # 0..len(valid_classes)-1
    maxprob = np.max(probs, axis=0)

    # Optionally set low-confidence pixels to nodata
    if prob_threshold > 0.0:
        argmax_idx[maxprob < prob_threshold] = -1  # temp invalid

    idx_to_class = np.array(valid_classes, dtype=np.int16)
    predicted_map = np.where(argmax_idx == -1, nodata_value, idx_to_class[argmax_idx])

# ------------------------------
# Mask nodata in both reference and prediction
# ------------------------------
true_vals = ref_labels
pred_vals = predicted_map
mask = (true_vals != nodata_value) & (pred_vals != nodata_value)
true_vals = true_vals[mask].astype(int)
pred_vals = pred_vals[mask].astype(int)

if true_vals.size == 0:
    raise RuntimeError("No valid pixels after masking; check alignment, nodata, and thresholds.")

# ------------------------------
# Compute metrics (stable label order)
# ------------------------------
labels_sorted = list(valid_classes)

cm = confusion_matrix(true_vals, pred_vals, labels=labels_sorted)

overall_acc = accuracy_score(true_vals, pred_vals)
kappa_val = cohen_kappa_score(true_vals, pred_vals, labels=labels_sorted)
# F1 per class (aligned to labels_sorted)
f1s = f1_score(true_vals, pred_vals, labels=labels_sorted, average=None)

print(f"SKLEARN {prob_threshold} -> OA: {overall_acc:.3f}, Kappa: {kappa_val:.3f}", flush=True)

# Producer's (recall) and User's (precision) accuracy from CM
with np.errstate(divide='ignore', invalid='ignore'):
    producer_acc = np.divide(np.diag(cm), cm.sum(axis=1), where=cm.sum(axis=1) != 0)  # recall
    user_acc = np.divide(np.diag(cm), cm.sum(axis=0), where=cm.sum(axis=0) != 0)      # precision
    f1_scores = np.where(
        (producer_acc + user_acc) > 0,
        2 * producer_acc * user_acc / (producer_acc + user_acc),
        0.0
    )

# ------------------------------
# Print results
# ------------------------------
print("\nConfusion matrix (rows = reference, cols = predicted):", flush=True)
print(cm)

print("\nClass | ProducerAcc | UserAcc |   F1", flush=True)
print("--------------------------------------------")
for cls, pa, ua, f1v in zip(labels_sorted, producer_acc, user_acc, f1_scores):
    print(f"{cls:5d} |   {pa:.3f}     |  {ua:.3f} |  {f1v:.3f}")

print(f"\nOverall accuracy : {overall_acc:.3f}")
print(f"Kappa statistic  : {kappa_val:.3f}")

# ------------------------------
# Save results to Excel
# ------------------------------
cm_df = pd.DataFrame(
    cm,
    index=[f"Ref_{c}" for c in labels_sorted],
    columns=[f"Pred_{c}" for c in labels_sorted]
)
metrics_df = pd.DataFrame({
    "Class": labels_sorted,
    "ProducerAcc": producer_acc,
    "UserAcc": user_acc,
    "F1_score": f1_scores,
    "F1_sklearn": f1s,
})
overall_df = pd.DataFrame({
    "Overall_accuracy": [overall_acc],
    "Kappa": [kappa_val],
    "N_pixels": [int(true_vals.size)]
})

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
    metrics_df.to_excel(writer, sheet_name="PerClass", index=False)
    overall_df.to_excel(writer, sheet_name="Overall", index=False)
print(f"\nResults saved to: {out_xlsx}", flush=True)

# ------------------------------
# Save plots
# ------------------------------
label_map = dict(zip(valid_classes, class_labels))

# Percent CM by reference row
row_sums = cm.sum(axis=1, keepdims=True)
with np.errstate(divide='ignore', invalid='ignore'):
    cm_percent = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)
cm_percent = np.round(cm_percent, 1)

# ---- Confusion Matrix Heatmap ----
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.1)
ax = sns.heatmap(
    cm_percent,
    annot=False,
    fmt=".1f",
    cmap="YlGn",
    cbar_kws={'label': 'Percentage (%)'},
    xticklabels=[label_map[c] for c in labels_sorted],
    yticklabels=[label_map[c] for c in labels_sorted],
    linewidths=0.5,
    linecolor='gray'
)
# annotations
for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        value = cm_percent[i, j]
        color = "white" if value > 50 else "black"
        ax.text(j + 0.5, i + 0.5, f"{value:.1f}",
                ha='center', va='center', color=color, fontsize=10)

plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Reference Class", fontsize=12)
plt.title("Confusion Matrix (% of Reference Class)", fontsize=14)
plt.tight_layout()
plt.savefig(out_cmat, dpi=300)
plt.close()

# ---- F1 Score Bar Chart ----
plt.figure(figsize=(8, 5))
colors = sns.color_palette("Set2", len(labels_sorted))
plt.bar([label_map[c] for c in labels_sorted], f1_scores, color=colors)
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.title("F1 scores per crop type")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out_f1sc, dpi=300)
plt.close()

print(f"\nPlots saved as {out_f1sc} and {out_cmat} at 300 DPI.", flush=True)

