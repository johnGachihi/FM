import os
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.merge import merge
from rasterio.mask import mask
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import box

# ------------------------------
# User Inputs
# ------------------------------
district = "Nyagatare"
eyear = 2025
season = "B"
outpath = '/cluster/archiving/GIZ/confidence/'
file_ending = 'merged_probs_with_2025'
out_xlsx = f"{outpath}Galileo_accuracy_{file_ending}.xlsx"
out_cmat = f"{outpath}Galileo_confmat_{file_ending}.png"
root = '/cluster/archiving/GIZ/data/'
shapefile_path = f'{root}shapefiles/RWA_B2025_Merge_v2_ValidSet.shp'
prob_thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
valid_classes = [0, 1, 2, 3]
nodata_value = 255
districts = ["Ruhango", "Nyagatare", "Musanze", "Nyabihu"]

# ------------------------------
# Merge Raster Probabilities
# ------------------------------
raster_paths = [
    f"{root}outputs/{d}_{season}{eyear}_{file_ending}.tif" for d in districts
]
rasters = [rasterio.open(rp) for rp in raster_paths]
mosaic, out_trans = merge(rasters)
meta = rasters[0].meta.copy()
meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans
})
for r in rasters:
    r.close()

# ------------------------------
# Load Shapefile
# ------------------------------
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(meta["crs"])

# ------------------------------
# Accuracy Checks per Threshold
# ------------------------------
from rasterio.io import MemoryFile

results = []
writer = pd.ExcelWriter(out_xlsx, engine='xlsxwriter')

for threshold in prob_thresholds:
    all_preds, all_gts = [], []

    for _, row in gdf.iterrows():
        geom = [row["geometry"]]
        label = row["code"]
        if label not in valid_classes:
            continue

        try:
            with MemoryFile() as memfile:
                with memfile.open(**meta) as dataset:
                    dataset.write(mosaic)
                    out_image, _ = mask(dataset=dataset, shapes=geom, crop=True)
        except Exception as e:
            print(f"❌ Masking failed for label {label}: {e}")
            continue

        if out_image.shape[0] != len(valid_classes):
            print(f"⚠️ Unexpected band count for label {label}. Skipping.")
            continue

        probs = np.moveaxis(out_image, 0, -1)  # (H, W, C)
        max_probs = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        preds[max_probs < threshold] = nodata_value

        valid_mask = preds != nodata_value
        y_pred = preds[valid_mask].flatten()
        y_true = np.full_like(y_pred, label)

        if len(y_pred) == 0:
            continue

        all_preds.extend(y_pred)
        all_gts.extend(y_true)

    if not all_preds:
        print(f"⚠️ No predictions found at threshold {threshold}")
        continue

    cm = confusion_matrix(all_gts, all_preds, labels=valid_classes)
    acc = accuracy_score(all_gts, all_preds)
    kappa = cohen_kappa_score(all_gts, all_preds)
    f1s = f1_score(all_gts, all_preds, labels=valid_classes, average=None)

    result = {
        "threshold": threshold,
        "overall_accuracy": acc,
        "kappa": kappa
    }
    for i, cls in enumerate(valid_classes):
        result[f"f1_class_{cls}"] = f1s[i]
    results.append(result)

    df_cm = pd.DataFrame(cm, index=[f"GT_{i}" for i in valid_classes],
                            columns=[f"Pred_{i}" for i in valid_classes])
    df_cm.to_excel(writer, sheet_name=f"cmat_t{threshold}")

    plt.figure(figsize=(6,5))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix (Threshold {threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{outpath}threshold_confmat_{file_ending}_t{threshold}.png")
    plt.close()

# Plotting peformance in each metric
results_df = pd.DataFrame(results)

sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Overall Accuracy
sns.lineplot(ax=axes[0, 0], data=results_df, x='threshold', y='overall_accuracy', marker='o')
axes[0, 0].set_title("Overall Accuracy")

# Kappa
sns.lineplot(ax=axes[0, 1], data=results_df, x='threshold', y='kappa', marker='o')
axes[0, 1].set_title("Kappa Score")

# F1 per class
for cls in valid_classes:
    sns.lineplot(
        ax=axes[1, 0], data=results_df, x='threshold', y=f'f1_class_{cls}', marker='o', label=f'Class {cls}'
    )
axes[1, 0].set_title("F1 Score per Class")
axes[1, 0].legend()

# F1 mean
f1_cols = [f"f1_class_{c}" for c in valid_classes]
results_df['f1_mean'] = results_df[f1_cols].mean(axis=1)
sns.lineplot(ax=axes[1, 1], data=results_df, x='threshold', y='f1_mean', marker='o')
axes[1, 1].set_title("Mean F1 Score")

for ax in axes.flat:
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")

plt.tight_layout()
plot_path = f'{outpath}threshold_performance_plot_{file_ending}.png'
plt.savefig(plot_path, dpi=300)
plt.show()
# ------------------------------
# Save Overall Summary
# ------------------------------

results_df.to_excel(writer, sheet_name="metrics_summary", index=False)
writer.close()
print("All metrics and confusion matrices saved.")

'''
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
file_ending = 'merged_probs_with_2025'
root = '/cluster/archiving/GIZ/data/'
outpath = f'/cluster/archiving/GIZ/confidence/'
vector_path = f'{root}shapefiles/RWA_{season}{eyear}_Merge_v2_ValidSet.shp'
prob_thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
valid_classes = [0, 1, 2, 3]
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
for src in sources:
    src.close()
prob_data = mosaic  # shape: (bands, height, width)

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

for threshold in prob_thresholds:
    print(f"Processing threshold: {threshold}")
    max_probs = np.max(prob_data, axis=0)
    winners = np.argmax(prob_data, axis=0)
    predicted_map = np.full_like(winners, nodata_value, dtype=np.uint8)
    predicted_map[max_probs >= threshold] = winners[max_probs >= threshold]

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

    if not pred_list:
        print(f"Threshold {threshold}: No valid predictions")
        continue

    pred_vals = np.concatenate(pred_list)
    true_vals = np.concatenate(ref_list)

    overall = accuracy_score(true_vals, pred_vals)
    kappa = cohen_kappa_score(true_vals, pred_vals)
    f1s = f1_score(true_vals, pred_vals, labels=valid_classes, average=None)

    result = {
        'threshold': threshold,
        'overall_accuracy': overall,
        'kappa': kappa,
    }
    for cls, f1 in zip(valid_classes, f1s):
        result[f'f1_class_{cls}'] = f1

    results.append(result)
    print(f"Threshold {threshold} -> OA: {overall:.3f}, Kappa: {kappa:.3f}")

# ------------------------------
# Save and Plot
# ------------------------------
results_df = pd.DataFrame(results)
metrics_file = f'{outpath}threshold_accuracy_metrics.csv'
results_df.to_csv(metrics_file, index=False)

# Plotting
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Overall Accuracy
sns.lineplot(ax=axes[0, 0], data=results_df, x='threshold', y='overall_accuracy', marker='o')
axes[0, 0].set_title("Overall Accuracy")

# Kappa
sns.lineplot(ax=axes[0, 1], data=results_df, x='threshold', y='kappa', marker='o')
axes[0, 1].set_title("Kappa Score")

# F1 per class
for cls in valid_classes:
    sns.lineplot(
        ax=axes[1, 0], data=results_df, x='threshold', y=f'f1_class_{cls}', marker='o', label=f'Class {cls}'
    )
axes[1, 0].set_title("F1 Score per Class")
axes[1, 0].legend()

# F1 mean
f1_cols = [f"f1_class_{c}" for c in valid_classes]
results_df['f1_mean'] = results_df[f1_cols].mean(axis=1)
sns.lineplot(ax=axes[1, 1], data=results_df, x='threshold', y='f1_mean', marker='o')
axes[1, 1].set_title("Mean F1 Score")

for ax in axes.flat:
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")

plt.tight_layout()
plot_path = f'{outpath}threshold_performance_plot.png'
plt.savefig(plot_path, dpi=300)
plt.show()
'''