import timeit
start_time = timeit.default_timer()
import os
import torch
import torch.nn as nn
import rasterio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from rasterio.merge import merge
import rasterio.mask
from rasterio.warp import reproject, Resampling
from src.galileo import Encoder
from src.data.utils import construct_galileo_input
from osgeo import gdal
import os
from pathlib import Path
#from pixel_wise_train_classifier import PixelwisePatchClassifier
#from finetune_gfm_3 import PixelwisePatchClassifier

# --- SETTINGS ---
filename_ending = "with_2025"
model_filename = f'gfm_model_{filename_ending}.pt'
district = "Musanze"
num_classes = 4 #XXXXX
root = '/cluster/archiving/GIZ/data/'
eyear = 2025
season = "B"
tile_folder = Path(f"{root}/{district}_{season}{eyear}_v2_tiles/")
output_folder = Path(f"{root}outputs/{district}_tiles_{filename_ending}/")
output_folder.mkdir(parents=True, exist_ok=True)
final_output_path = f"{root}outputs/{district}_{season}{eyear}_merged_labels_{filename_ending}.tif"
masked_label_output_path = f"{root}outputs/{district}_{season}{eyear}_merged_masked_labels_{filename_ending}.tif"#"outputs/merged_prediction_masked.tif"
final_prob_output_path = f"{root}outputs/{district}_{season}{eyear}_merged_probs_{filename_ending}.tif"
masked_prob_output_path = f"{root}outputs/{district}_{season}{eyear}_merged_masked_probs_{filename_ending}.tif"
# Path to binary mask raster (0 and 1(retain value)
mask_raster_path = f"{root}masks/ESA_crop_lands_only_mask.tif" #/cluster/archiving/GIZ/data/masks/

encoder_ckpt = "models/nano/"
model_ckpt = f"/cluster/archiving/GIZ/data/checkpoints/{model_filename}" #Model directory
patch_size = 8
stride = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
ignore_value = 255
confidence_threshold = 0.0
nsteps = 5 #number of months or steps
nbands = 12 # number of bands
nstatic = 11

####C
class PixelwisePatchClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.classifier = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=num_classes,
            kernel_size=1
        )

    def encode_features(self, x):
        B, C, H, W = x.shape

        # Split temporal and static bands
        temporal = x[:, :nsteps * nbands, :, :].view(B, nsteps, nbands, H, W)  # [B, 5, 12, H, W]
        temporal = temporal.permute(0, 1, 3, 4, 2).contiguous()                # [B, 5, H, W, 12]

        inputs = []

        for b in range(B):
            # Sentinel-1: VV, VH (first 2 channels)
            s1 = temporal[b, ..., :2].permute(1, 2, 0, 3).float()  # [H, W, 5, 2]

            # Sentinel-2: next 10 channels
            s2 = temporal[b, ..., 2:].permute(1, 2, 0, 3).float()  # [H, W, 5, 10]

            # Static bands: elevation (1), slope (1), DW (9)
            static = x[b, -nstatic:, :, :]  # [11, H, W]
            srtm = static[:2, :, :].permute(1, 2, 0).float()  # [H, W, 2]
            dw = static[2:, :, :].permute(1, 2, 0).float()    # [H, W, 9]

            # Call construct_galileo_input
            masked = construct_galileo_input(s1=s1, s2=s2, srtm=srtm, dw=dw, normalize=True)
            inputs.append(masked)

        # Stack per-batch tensors into batched input
        batched_input = {
            k: torch.stack([getattr(i, k).float() if k != "months" else getattr(i, k).long() for i in inputs])
            for k in inputs[0]._fields
        }

        feats, *_ = self.encoder(
            batched_input["space_time_x"],
            batched_input["space_x"],
            batched_input["time_x"],
            batched_input["static_x"],
            batched_input["space_time_mask"],
            batched_input["space_mask"],
            batched_input["time_mask"],
            batched_input["static_mask"],
            batched_input["months"],
            patch_size=H,
        )
        return feats

    def forward(self, x):
        feats = self.encode_features(x)
        while feats.dim() > nsteps:
            feats = feats.squeeze(1)
        feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return self.classifier(feats)

# --- Load model ---
print("[INFO] Loading encoder and model")
encoder = Encoder.load_from_folder(Path(encoder_ckpt))
model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=True)
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model = model.to(device).eval()

# --- Process each tile ---
tile_paths = sorted(tile_folder.glob("*.tif"))
pred_tiles = []

print(f"[INFO] Found {len(tile_paths)} tiles.")

for tile_path in tqdm(tile_paths, desc="Processing tiles"):
    print(f"[DEBUG] Processing tile: {tile_path.name}")
    with rasterio.open(tile_path) as src:
        image = src.read()  # [C=60, H, W]
        profile = src.profile
        H, W = image.shape[1:]

        print(f"[DEBUG] Tile shape: {image.shape}")
        expected_channels = nsteps * nbands + nstatic  # 71
        if image.shape[0] != expected_channels:
            raise ValueError(f"[ERROR] Tile must have {expected_channels} channels ({nsteps} timesteps x {nbands} bands + {nstatic}). Got {image.shape[0]}")

        # Pad tile if needed
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
        H_pad, W_pad = image_padded.shape[1:]
        print(f"[DEBUG] Padded tile shape: {image_padded.shape}")

        pred_map = np.zeros((num_classes, H_pad, W_pad), dtype=np.float32)
        count_map = np.zeros((H_pad, W_pad), dtype=np.uint8)

        # Slide window and infer
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image_padded[:, y:y+patch_size, x:x+patch_size]  # [60, 8, 8]

                # Split into temporal and static
                temporal = patch[:nsteps * nbands]  # shape [60, H, W]
                static = patch[nsteps * nbands:]    # shape [11, H, W]

                # Reshape temporal to [T, B, H, W] → then to [H, W, T, B]
                temporal = temporal.reshape(nsteps, nbands, patch_size, patch_size).transpose(0, 2, 3, 1)  # [5, 8, 8, 12]
                s1 = temporal[..., :2].transpose(1, 2, 0, 3)  # [8, 8, 5, 2]
                s2 = temporal[..., 2:].transpose(1, 2, 0, 3)  # [8, 8, 5, 10]

                # Static bands
                srtm = static[:2].transpose(1, 2, 0)  # [8, 8, 2]
                dw = static[2:].transpose(1, 2, 0)    # [8, 8, 9]

                # Convert to tensors
                s1 = torch.from_numpy(s1).float()
                s2 = torch.from_numpy(s2).float()
                srtm = torch.from_numpy(srtm).float()
                dw = torch.from_numpy(dw).float()

                # Construct Galileo input
                masked = construct_galileo_input(s1=s1, s2=s2, srtm=srtm, dw=dw, normalize=True)

                batched_input = {
                    k: torch.stack([getattr(masked, k).float() if k != "months" else getattr(masked, k).long()])
                    for k in masked._fields
                }
                batched_input = {k: v.to(device) for k, v in batched_input.items()}

                with torch.no_grad():
                    feats, *_ = encoder(
                        batched_input["space_time_x"],
                        batched_input["space_x"],
                        batched_input["time_x"],
                        batched_input["static_x"],
                        batched_input["space_time_mask"],
                        batched_input["space_mask"],
                        batched_input["time_mask"],
                        batched_input["static_mask"],
                        batched_input["months"],
                        patch_size=patch_size,
                    )
                    feats = feats.squeeze(1)[:, -1, :, :, :]  # [1, C, H, W]
                    feats = feats.permute(0, 3, 1, 2).contiguous()
                    logits = model.classifier(feats)

                    probs = torch.softmax(logits, dim=1)
                    probs = F.interpolate(probs, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                    probs = probs.squeeze(0).cpu().numpy()  # [num_classes, 8, 8]

                pred_map[:, y:y+patch_size, x:x+patch_size] += probs
                count_map[y:y+patch_size, x:x+patch_size] += 1

        # --- Generate final prediction mask ---
        avg_probs = pred_map / np.clip(count_map, a_min=1, a_max=None)
        confidence = np.max(avg_probs, axis=0)
        final_mask = np.argmax(avg_probs, axis=0).astype(np.uint8)

        # Create valid data mask from original (unpadded) image
        valid_data_mask = np.isfinite(image).all(axis=0) & (np.abs(image).sum(axis=0) >= 1e-6)
        valid_data_mask_padded = np.pad(valid_data_mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=False)

        # Combine conditions: low confidence, no overlapping patches, or invalid input
        mask_low_conf = (confidence < confidence_threshold) | (count_map == 0) | (~valid_data_mask_padded)
        final_mask[mask_low_conf] = ignore_value

        # Remove padding from final labels
        final_mask = final_mask[:H, :W]

        ignore_ratio = (final_mask == ignore_value).sum() / final_mask.size
        
        # --- Save prediction mask ---
        out_tile_path = output_folder / tile_path.name.replace(".tif", "_pred.tif")
        profile.update(count=1, dtype='uint8', height=H, width=W)
        with rasterio.open(out_tile_path, "w", **profile) as dst:
            dst.write(final_mask, 1)

        pred_tiles.append(out_tile_path)

        # --- Save pixel-wise class probabilities ---
        probs_tile_path = output_folder / tile_path.name.replace(".tif", "_probs.tif")
        prob_profile = profile.copy()
        prob_profile.update(count=num_classes, dtype='float32', height=H, width=W)

        # Remove padding from avg_probs
        avg_probs = avg_probs[:, :H, :W]

        with rasterio.open(probs_tile_path, "w", **prob_profile) as dst_prob:
            dst_prob.write(avg_probs.astype(np.float32))

# --- Merge predicted/labels tiles ---
srcs = [rasterio.open(p) for p in pred_tiles]
mosaic, out_transform = merge(srcs)

profile = srcs[0].profile
profile.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
    "count": 1,
    "dtype": "uint8"
})

with rasterio.open(final_output_path, "w", **profile) as dst:
    dst.write(mosaic.astype(np.uint8))

print(f"[INFO] Merged prediction saved to: {final_output_path}")

###MASK OUT NON CROP AREAS from classification===========================================
# ───────────────────────────────────────────────────────────────
# 1. Load mask raster
# ───────────────────────────────────────────────────────────────
with rasterio.open(mask_raster_path) as mask_src:
    mask_data      = mask_src.read(1)          # single band
    mask_transform = mask_src.transform
    mask_crs       = mask_src.crs

# ───────────────────────────────────────────────────────────────
# 2. Load (merged) label raster
# ───────────────────────────────────────────────────────────────
with rasterio.open(final_output_path) as label_src:
    merged_labels  = label_src.read(1)         # shape [H, W]
    label_profile  = label_src.profile
    label_shape    = merged_labels.shape
    label_crs      = label_profile["crs"]
    label_transform= label_profile["transform"]

# ───────────────────────────────────────────────────────────────
# 3. Ensure mask matches label CRS / grid; reproject if necessary
# ───────────────────────────────────────────────────────────────
if (mask_crs != label_crs) or (mask_transform != label_transform):
    print("[INFO] Mask CRS/grid differs from labels – reprojecting mask…")

    mask_reproj = np.empty(label_shape, dtype=mask_data.dtype)

    reproject(
        source        = mask_data,
        destination   = mask_reproj,
        src_transform = mask_transform,
        src_crs       = mask_crs,
        dst_transform = label_transform,
        dst_crs       = label_crs,
        resampling    = Resampling.nearest   # preserve integer mask values
    )

    mask_data      = mask_reproj
    mask_crs       = label_crs
    mask_transform = label_transform
else:
    print("[INFO] Mask already aligned with labels – no reprojection needed.")

# ───────────────────────────────────────────────────────────────
# 4. Shape check (defensive)
# ───────────────────────────────────────────────────────────────
if mask_data.shape != merged_labels.shape:
    raise ValueError(
        f"Post‑reprojection mask shape {mask_data.shape} ≠ label shape {merged_labels.shape}"
    )

# ───────────────────────────────────────────────────────────────
# 5. Apply mask to labels (set to ignore_value where mask == 0)
# ───────────────────────────────────────────────────────────────
masked_labels = merged_labels.copy()
masked_labels[mask_data == 0] = ignore_value

# ───────────────────────────────────────────────────────────────
# 6. Save masked label raster
# ───────────────────────────────────────────────────────────────
label_profile.update(dtype="uint8",
                    compress='deflate',
                    tiled=True,
                    nodata=ignore_value
                    )

with rasterio.open(masked_label_output_path, "w", **label_profile) as dst:
    dst.write(masked_labels.astype(np.uint8), 1)

print(f"[INFO] Masked labels saved to: {masked_label_output_path}")

# ------------------------------------------------------------------
# 1. Merge probability tiles: Gather probability tiles
# ------------------------------------------------------------------
prob_tile_paths = sorted(output_folder.glob("*_probs.tif"))
print(f"[INFO] Found {len(prob_tile_paths)} probability tiles to merge.")

if len(prob_tile_paths) == 0:
    raise RuntimeError("No *_probs.tif files found!")

# ------------------------------------------------------------------
# 2. Build a VRT that references all tiles (allows overlaps)
# ------------------------------------------------------------------
tmp_vrt_path = output_folder / "tmp_probs.vrt"

vrt = gdal.BuildVRT(
    str(tmp_vrt_path),
    [str(p) for p in prob_tile_paths],
    options=gdal.BuildVRTOptions(resampleAlg="nearest", addAlpha=False)
)
vrt.FlushCache()

# ------------------------------------------------------------------
# 3. Warp (flatten) the VRT to a GeoTIFF, averaging overlaps
# ------------------------------------------------------------------
gdal.Warp(
    destNameOrDestDS=str(final_prob_output_path),
    srcDSOrSrcDSTab=str(tmp_vrt_path),
    format="GTiff",
    options=gdal.WarpOptions(
        resampleAlg="average",          # <-- averages overlapping pixels
        dstNodata="nan",                # keep NaN as nodata (float32)
        creationOptions=[
            "COMPRESS=DEFLATE",         # better for float32
            "PREDICTOR=3",
            "TILED=YES"
        ]
    )
)

print(f"[INFO] Merged probabilities saved to: {final_prob_output_path}")

# ------------------------------------------------------------------
# clean up temporary VRT
# ------------------------------------------------------------------
try:
    os.remove(tmp_vrt_path)
except OSError:
    pass

# ------------------------------------------------------------------
# Load merged probability raster
# ------------------------------------------------------------------
with rasterio.open(final_prob_output_path) as prob_src:
    mosaic_probs = prob_src.read()  # shape: [C, H, W]
    prob_profile = prob_src.profile
    prob_crs = prob_src.crs
    prob_transform = prob_src.transform
    prob_shape = mosaic_probs.shape[1:]  # (H, W)

# ------------------------------------------------------------------
# Validate shape
# ------------------------------------------------------------------
if mask_data.shape != prob_shape:
    raise ValueError(f"[ERROR] Mask shape {mask_data.shape} doesn't match probability mosaic shape {prob_shape}.")

# ------------------------------------------------------------------
# Apply mask — set all probabilities to np.nan where mask == 0
# ------------------------------------------------------------------
masked_probs = mosaic_probs.copy()
masked_probs[:, mask_data == 0] = np.nan

# ------------------------------------------------------------------
# STEP 6: Save masked probability raster
# ------------------------------------------------------------------
prob_profile.update(dtype="float32", nodata=np.nan)

with rasterio.open(masked_prob_output_path, "w", **prob_profile) as dst:
    dst.write(masked_probs.astype(np.float32))

print(f"[INFO] Masked probabilities saved to: {masked_prob_output_path}")


print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)

