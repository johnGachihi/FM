import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Load .npz file ---
data = np.load("/cluster/archiving/GIZ/data/patches_SB/val/patch_val_00265.npz")
print("Available keys:", data.files)

# Inspect available keys
print("Available keys:", data.files)

# Assume the label array is stored under one of the keys, e.g., 'labels'
label_array = data['mask']  # Replace 'labels' with the actual key name

# --- Define a custom colormap for 4 classes ---
cmap = mcolors.ListedColormap(['black', 'green', 'blue', 'yellow'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Plot the label image ---
plt.figure(figsize=(6, 6))
plt.imshow(label_array, cmap=cmap, norm=norm)
plt.colorbar(ticks=[0, 1, 2, 3], label='Class Labels')
plt.title("Label Image from .npz")
plt.axis('off')
plt.show()
