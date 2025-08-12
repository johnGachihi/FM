# ------------------------------------------------------------
# Author: Joseph Chemut 
# Accuracy assessment using pre-rasterized reference labels
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

VALID_LABELS = {0, 1, 2, 3}
IGNORE_VALUE = 255

# Indices for first timestep Sentinel-2 bands
B2_IDX = 2  # Blue
B3_IDX = 3  # Green
B4_IDX = 4  # Red

# Directly specify your patch file path here
PATCH_PATH = "data/patches_21/train/patch_train_00055.npz"

def inspect_patch(patch_path):
    data = np.load(patch_path)

    image = data['image']  # shape: (C, H, W)
    mask = data['mask']    # shape: (H, W)

    print(f"Image shape: {image.shape}")
    print(f"Label shape: {mask.shape}")

    unique_labels = np.unique(mask)
    print(f"Unique label values in mask: {unique_labels}")

    # Only count valid label pixels
    covered_pixels = np.isin(mask, list(VALID_LABELS)).sum()
    total_pixels = mask.size
    print(f"Pixels with valid label: {covered_pixels}/{total_pixels} "
          f"({(covered_pixels / total_pixels) * 100:.2f}%)")

    # Prepare RGB for first timestep (B4, B3, B2)
    def normalize(band):
        return (band - band.min()) / (band.max() - band.min() + 1e-6)

    red = normalize(image[B4_IDX])
    green = normalize(image[B3_IDX])
    blue = normalize(image[B2_IDX])
    rgb = np.dstack([red, green, blue])

    # Visualize
    plt.figure(figsize=(10, 4))

    # First timestep Sentinel-2 RGB
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Sentinel-2 RGB (First timestep)")
    plt.axis("off")

    # Show label mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='tab10', vmin=0, vmax=max(VALID_LABELS))
    plt.title("Label Mask")
    plt.colorbar(label='Label')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_patch(PATCH_PATH)
