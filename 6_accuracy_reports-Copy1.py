# ------------------------------------------------------------
# Author: Benson Kenduiywo
# The script assesess accuracy of a classification using data polygons not seen by the model during training
# ------------------------------------------------------------
import rasterio
from rasterio.merge import merge
from rasterio.features import geometry_mask
from shapely import wkb
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os

# ------------------------------
# User Inputs
# ------------------------------
districts = ["Ruhango", "Nyagatare", "Musanze", "Nyabihu"]
eyear = 2025
season = "B"
prob_threshold = 0.7
file_ending = 'merged_probs_with_2025'
root = '/cluster/archiving/GIZ/data/'
outpath = f'/cluster/archiving/GIZ/accuracy/'
path_maps = '/cluster/archiving/GIZ/maps/'
filename_map = f'{path_maps}RWA_{season}{eyear}_confidence_{prob_threshold}.tif'
vector_path = f'{root}shapefiles/RWA_{season}{eyear}_Merge_v2_ValidSet.shp'

out_xlsx = f'{outpath}Metrics_{file_ending}_threshold_{prob_threshold}.xlsx'
out_f1sc =  f'{outpath}F1Score_{file_ending}_threshold_{prob_threshold}.png'
out_cmat = f'{outpath}Confmat_{file_ending}_threshold_{prob_threshold}.png'
valid_classes = [0, 1, 2, 3]
class_labels = ['Bean', 'Irish Potato', 'Maize', 'Rice']  # adjust as needed
nodata_value = 255
N_WORKERS = os.cpu_count() - 10
# ------------------------------
# Merge Raster Probabilities
# ------------------------------

raster_paths = [
    f"{root}outputs/{d}_{season}{eyear}_{file_ending}.tif"
    for d in districts
]
sources = [rasterio.open(p) for p in raster_paths]
mosaic, transform = merge(sources)
crs = sources[0].crs

# ------------------------------
# Read and Filter Vector Data
# ------------------------------
gdf = gpd.read_file(vector_path).to_crs(crs)
gdf = gdf[gdf.geometry.notnull()]
jobs = [{"wkb": row.geometry.wkb, "ref": int(row["code"])} for _, row in gdf.iterrows() if row["code"] in valid_classes]

# ------------------------------
# Define Pixel Extraction Function
# ------------------------------
def extract_pixels(job):
    geom = wkb.loads(job["wkb"])
    mask = geometry_mask([geom], transform=transform, invert=True, out_shape=prob_data.shape[1:])
    return mask, job["ref"]

# ------------------------------
# Run Accuracy Evaluation Per Threshold
# ------------------------------
results = []
print(f"Processing threshold: {prob_threshold}")
max_probs = np.max(prob_data, axis=0)
winners = np.argmax(prob_data, axis=0).astype(np.uint8)
predicted_map = np.where(max_probs >= prob_threshold, winners, nodata_value).astype(np.uint8)

#predicted_map = np.full_like(winners, nodata_value, dtype=np.uint8)
#predicted_map[max_probs >= prob_threshold] = winners[max_probs >= prob_threshold]
#predicted_map = predicted_map.astype(np.uint8)
#predicted_map[~np.isin(predicted_map, valid_classes)] = nodata_value


# ------------------------------
# Save predicted_map to GeoTIFF
# ------------------------------
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio import features
from rasterio.plot import reshape_as_image

label_colormap = {
    0: (85, 255, 0),     # Bean -> #55FF00
    1: (115, 38, 0),     # Irish Potato -> #732600
    2: (255, 212, 0),    # Maize -> #FFD400
    3: (0, 169, 230),    # Rice -> #00A9E6
    255: (255, 255, 255) # No Data (white)
}

# Metadata with class names
class_metadata = {
    "0": "Bean",
    "1": "Irish Potato",
    "2": "Maize",
    "3": "Rice",
    "255": "No Data"
}

with rasterio.open(
    filename_map,
    'w',
    driver='GTiff',
    height=predicted_map.shape[0],
    width=predicted_map.shape[1],
    count=1,
    dtype='uint8',
    crs=crs,
    transform=transform,
    compress='lzw',
    tiled=True,
    nodata=nodata_value
) as dst:
    dst.write(predicted_map, 1)
    
    # Apply color map
    dst.write_colormap(1, label_colormap)
    
    # Write class label metadata
    dst.update_tags(**{f'class_{k}': v for k, v in class_metadata.items()})

print(f"Saved predicted map with class labels to: {filename_map}")

#Clean-up
for src in sources:
    src.close()
prob_data = mosaic  # shape: (bands, height, width)

# ------------------------------
# Continue with accuracy assessment
# ------------------------------
pred_list, ref_list = [], []
with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
    masks_and_refs = list(tqdm(exe.map(extract_pixels, jobs), total=len(jobs)))
    for mask, ref in masks_and_refs:
        preds = predicted_map[mask]
        if nodata_value is not None:
            preds = preds[preds != nodata_value]
        if preds.size > 0:
            pred_list.append(preds)
            ref_list.append(np.full(preds.shape, ref, dtype=np.int16))

pred_vals = np.concatenate(pred_list)
true_vals = np.concatenate(ref_list)
classes = np.union1d(true_vals, pred_vals)       # all present class codes
cm = confusion_matrix(true_vals, pred_vals, labels=classes)

overall = accuracy_score(true_vals, pred_vals)
kappa = cohen_kappa_score(true_vals, pred_vals)
f1s = f1_score(true_vals, pred_vals, labels=valid_classes, average=None)

result = {
    'threshold': prob_threshold,
    'overall_accuracy': overall,
    'kappa': kappa,
}
for cls, f1 in zip(valid_classes, f1s):
    result[f'f1_class_{cls}'] = f1

print(f"SKLEARN {prob_threshold} -> OA: {overall:.3f}, Kappa: {kappa:.3f}")
# ---------- 4. Compute metrics --------------------------------------
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


