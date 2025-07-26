# ------------------------------------------------------------
# Author: Benson Kenduiywo
# The script assesess accuracy of a classification using data polygons not seen by the model during training
# ------------------------------------------------------------
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import rasterio
from rasterio.merge import merge
from concurrent.futures import ProcessPoolExecutor
import os
from shapely import wkb  
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ---------- 1. Paths -------------------------------------------------
outpath = '/cluster/archiving/GIZ/results/'
file_ending = 'all_with_2025'
out_xlsx = f"{outpath}Galileo_accuracy_{file_ending}.xlsx"
out_cmat = f"{outpath}Galileo_confmat_{file_ending}.png"
out_f1sc = f"{outpath}Galileo_f1score_{file_ending}.png"
root = '/cluster/archiving/GIZ/data/'
district1 = "Ruhango"
district2 = "Nyagatare"
district3 = "Musanze"
district4 = "Nyabihu"
eyear = 2025
season = "B"
vector_path = f"{root}shapefiles/RWA_{season}{eyear}_Merge_v2_ValidSet.shp"      
N_WORKERS = os.cpu_count() - 10
print(f"Logical cores: {N_WORKERS}")
# class value used for nodata (set None if none)
NODATA_VALUE = 255
'''
# ---------- 2. Read data --------------------------------------------
with rasterio.open(raster_path) as src:
    labels = src.read(1)                         # 2‑D array
    raster_crs = src.crs
    raster_transform = src.transform
'''
raster_paths = [
    f"{root}outputs/{d}_{season}{eyear}_merged_labels_all_with_2025.tif"
    for d in [district1, district2, district3, district4]
]



# ---------- 1. Merge rasters in memory ----------
sources = [rasterio.open(p) for p in raster_paths]
raster_crs = sources[0].crs
mosaic, raster_transform  = merge(sources)
for src in sources:
    src.close()
labels = mosaic[0, :, :]  # extract first band
# read polygons and re‑project to raster CRS
polys = gpd.read_file(vector_path).to_crs(raster_crs)
# ---------- 3. Collect pixel‑wise pairs -----------------------------
true_vals   = []
pred_vals   = []
'''
print("Extracting pixels per polygon ...")
for _, row in tqdm(polys.iterrows(), total=len(polys)):
    geom   = row.geometry
    ref_id = row["code"]

    # Build a mask of pixels whose centre is inside the polygon
    mask = features.geometry_mask([geom],
                                  out_shape=labels.shape,
                                  transform=raster_transform,
                                  invert=True)           # True = inside polygon

    # pixels inside polygon
    pix = labels[mask]

    # drop nodata
    if NODATA_VALUE is not None:
        pix = pix[pix != NODATA_VALUE]

    # append predictions and reference (same length)
    pred_vals.extend(pix.tolist())
    true_vals.extend([ref_id] * len(pix))

true_vals = np.array(true_vals, dtype=int)
pred_vals = np.array(pred_vals, dtype=int)

'''
polys = polys = polys[polys.geometry.notnull()] #Drop invalid or missing geometries
jobs = [{"wkb": row.geometry.wkb, "ref": int(row["code"])} for _, row in polys.iterrows()]
def extract_pixels(job):
    geom = wkb.loads(job["wkb"])  # correct way to load geometry
    mask = geometry_mask([geom], transform=raster_transform, invert=True, out_shape=labels.shape)
    pix = labels[mask]
    if NODATA_VALUE is not None:
        pix = pix[pix != NODATA_VALUE]
    return pix, np.full(pix.shape, job["ref"], dtype=np.int16)

jobs = [{"wkb": row.geometry.wkb, "ref": int(row["code"])} for _, row in polys.iterrows()]

# ---------- 4. Extract pixels in parallel ----------
pred_list, ref_list = [], []
with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
    results = list(tqdm(exe.map(extract_pixels, jobs), total=len(jobs)))
    for pred, ref in results:
        if pred.size > 0:
            pred_list.append(pred)
            ref_list.append(ref)

pred_vals = np.concatenate(pred_list)
true_vals = np.concatenate(ref_list)

# ---------- 4. Compute metrics --------------------------------------
classes = np.union1d(true_vals, pred_vals)       # all present class codes
cm = confusion_matrix(true_vals, pred_vals, labels=classes)

producer_acc = np.diag(cm) / cm.sum(axis=0)      # recall
user_acc     = np.diag(cm) / cm.sum(axis=1)      # precision
f1_scores    = 2 * producer_acc * user_acc / (producer_acc + user_acc)

overall_acc  = np.trace(cm) / cm.sum()
kappa_val    = cohen_kappa_score(true_vals, pred_vals, labels=classes)

# ---------- 5. Display ----------------------------------------------
print("\nConfusion matrix (rows = reference, cols = predicted):")
print(cm)

print("\nClass | ProducerAcc | UserAcc |   F1")
print("--------------------------------------------")
for cls, pa, ua, f1 in zip(classes, producer_acc, user_acc, f1_scores):
    print(f"{cls:5d} |   {pa:.3f}     |  {ua:.3f} |  {f1:.3f}")

print(f"\nOverall accuracy : {overall_acc:.3f}")
print(f"Kappa statistic  : {kappa_val:.3f}")


# ---------- 6. Save results to Excel --------------------------------

# 6a. Confusion‑matrix sheet
cm_df = pd.DataFrame(cm, index=[f"Ref_{c}" for c in classes],
                         columns=[f"Pred_{c}" for c in classes])

# 6b. Per‑class metrics sheet
metrics_df = pd.DataFrame({
    "Class":         classes,
    "ProducerAcc":   producer_acc,
    "UserAcc":       user_acc,
    "F1_score":      f1_scores
})

# 6c. Overall metrics sheet
overall_df = pd.DataFrame({
    "Overall_accuracy": [overall_acc],
    "Kappa":            [kappa_val]
})

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
    metrics_df.to_excel(writer, sheet_name="PerClass", index=False)
    overall_df.to_excel(writer, sheet_name="Overall", index=False)

print(f"\nResults saved to: {out_xlsx}")

# ---------- 7. Save plots to png --------------------------------
class_labels = ['Bean', 'Irish Potato', 'Maize', 'Rice', 'Bean and Maize']  # adjust as needed

# Ensure labels and classes match
label_map = dict(zip(classes, [class_labels[c] for c in classes]))

# ========== 1. Confusion Matrix Plot ==========
# Normalize to percentages per row (reference class)
cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
cm_percent = np.round(cm_percent, decimals=1)  # Round to 1 decimal

# Plot
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)
ax = sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="YlGn", cbar_kws={'label': 'Percentage (%)'},
                 xticklabels=[class_labels[i] for i in range(len(classes))],
                 yticklabels=[class_labels[i] for i in range(len(classes))],
                 linewidths=0.5, linecolor='gray')

# Axis labels and title
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Reference Class", fontsize=12)
plt.title("Confusion Matrix (% of Reference Class)", fontsize=14)

plt.tight_layout()
plt.savefig(out_cmat, dpi=300)
plt.close()

# ========== 2. F1 Score Bar Plot ==========
plt.figure(figsize=(8, 5))
colors = sns.color_palette("Set2", len(classes))
plt.bar([label_map[c] for c in classes], f1_scores, color=colors)
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.title("F1 scores per crop type")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out_f1sc, dpi=300)
plt.close()

print(f"\nPlots saved as {out_f1sc} and {out_cmat} at 300 DPI.")
