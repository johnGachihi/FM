# ------------------------------------------------------------
# Author: Benson Kenduiywo
# Accuracy assessment using pre-rasterized reference labels
# ------------------------------------------------------------
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
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
file_ending = 'merged_probs_with_2025'
root = '/cluster/archiving/GIZ/data/'
outpath = f'/cluster/archiving/GIZ/accuracy/'

vector_path = f'{root}shapefiles/RWA_{season}{eyear}_Merge_v2_ValidSet.shp'

vector_base = os.path.splitext(os.path.basename(vector_path))[0]
reference_raster_path = f'{outpath}{vector_base}.tif'

prob_threshold = 0.0
out_xlsx = f'{outpath}Metrics_{file_ending}_threshold_{prob_threshold}.xlsx'
out_f1sc = f'{outpath}F1Score_{file_ending}_threshold_{prob_threshold}.png'
out_cmat = f'{outpath}Confmat_{file_ending}_threshold_{prob_threshold}.png'

valid_classes = [0, 1, 2, 3]
class_labels = ['Bean', 'Irish Potato', 'Maize', 'Rice']  # adjust as needed
nodata_value = 255
N_WORKERS = os.cpu_count() - 10

# ------------------------------
# Boolean flag — rasterize once or load existing
# ------------------------------
RASTERIZE_REFERENCE = False 

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
height, width = mosaic.shape[1], mosaic.shape[2]
for src in sources:
    src.close()
prob_data = mosaic  # shape: (bands, height, width)

# ------------------------------
# Rasterize reference polygons once
# ------------------------------
if RASTERIZE_REFERENCE:
    print("Rasterizing reference polygons...", flush = True)
    gdf = gpd.read_file(vector_path).to_crs(crs)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf["code"].isin(valid_classes)]
    shapes = [(geom, int(code)) for geom, code in zip(gdf.geometry, gdf["code"])]

    ref_rasterized = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata_value,
        dtype="uint8"
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
        "tiled": True
    }
    with rasterio.open(reference_raster_path, "w", **profile) as dst:
        dst.write(ref_rasterized, 1)
    print(f"Reference raster saved: {reference_raster_path}", flush = True)

# ------------------------------
# Load reference raster
# ------------------------------
with rasterio.open(reference_raster_path) as ref_src:
    ref_labels = ref_src.read(1)

# ------------------------------
# Predict map from probability data
# ------------------------------
print(f"Processing threshold: {prob_threshold}", flush = True)
max_probs = np.max(prob_data, axis=0)
winners = np.argmax(prob_data, axis=0)
predicted_map = np.full_like(winners, nodata_value, dtype=np.uint8)
predicted_map[max_probs >= prob_threshold] = winners[max_probs >= prob_threshold]

# Mask nodata in both reference and prediction
pred_vals = predicted_map
true_vals = ref_labels
mask = (true_vals != nodata_value) & (pred_vals != nodata_value)
true_vals = true_vals[mask]
pred_vals = pred_vals[mask]


#mask_valid = (ref_labels != nodata_value)
#pred_vals = predicted_map[mask_valid]
#true_vals = ref_labels[mask_valid]

# ------------------------------
# Compute metrics
# ------------------------------
classes = np.union1d(true_vals, pred_vals)
cm = confusion_matrix(true_vals, pred_vals, labels=classes)

overall_acc = accuracy_score(true_vals, pred_vals)
kappa_val = cohen_kappa_score(true_vals, pred_vals)
f1s = f1_score(true_vals, pred_vals, labels=valid_classes, average=None)

print(f"SKLEARN {prob_threshold} -> OA: {overall_acc:.3f}, Kappa: {kappa_val:.3f}", flush = True)

producer_acc = np.diag(cm) / cm.sum(axis=0)  # recall
user_acc = np.diag(cm) / cm.sum(axis=1)      # precision
f1_scores = 2 * producer_acc * user_acc / (producer_acc + user_acc)

# ------------------------------
# Print results
# ------------------------------
print("\nConfusion matrix (rows = reference, cols = predicted):", flush =True)
print(cm)

print("\nClass | ProducerAcc | UserAcc |   F1", flush =True)
print("--------------------------------------------")
for cls, pa, ua, f1 in zip(classes, producer_acc, user_acc, f1_scores):
    print(f"{cls:5d} |   {pa:.3f}     |  {ua:.3f} |  {f1:.3f}")

print(f"\nOverall accuracy : {overall_acc:.3f}")
print(f"Kappa statistic  : {kappa_val:.3f}")

# ------------------------------
# Save results to Excel
# ------------------------------
cm_df = pd.DataFrame(cm, index=[f"Ref_{c}" for c in classes],
                     columns=[f"Pred_{c}" for c in classes])
metrics_df = pd.DataFrame({
    "Class": classes,
    "ProducerAcc": producer_acc,
    "UserAcc": user_acc,
    "F1_score": f1_scores
})
overall_df = pd.DataFrame({
    "Overall_accuracy": [overall_acc],
    "Kappa": [kappa_val]
})

with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
    metrics_df.to_excel(writer, sheet_name="PerClass", index=False)
    overall_df.to_excel(writer, sheet_name="Overall", index=False)
print(f"\nResults saved to: {out_xlsx}", flush =True)

# ------------------------------
# Save plots
# ------------------------------
# Filter to valid classes
classes = [c for c in classes if c in valid_classes]
label_map = dict(zip(valid_classes, class_labels))
cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
cm_percent = np.round(cm_percent, 1)

# ---- Confusion Matrix Heatmap ----
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)

ax = sns.heatmap(
    cm_percent,
    annot=False,  # We'll handle annotations manually
    fmt=".1f",
    cmap="YlGn",
    cbar_kws={'label': 'Percentage (%)'},
    xticklabels=[class_labels[i] for i in range(len(classes))],
    yticklabels=[class_labels[i] for i in range(len(classes))],
    linewidths=0.5,
    linecolor='gray'
)

# Manually add text annotations to ALL cells
for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        value = cm_percent[i, j]
        color = "white" if value > 50 else "black"  # High-contrast text
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
colors = sns.color_palette("Set2", len(classes))
plt.bar([label_map[c] for c in classes], f1_scores, color=colors)
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.title("F1 scores per crop type")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out_f1sc, dpi=300)
plt.close()


print(f"\nPlots saved as {out_f1sc} and {out_cmat} at 300 DPI.", flush =True)
