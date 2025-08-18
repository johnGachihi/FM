# ------------------------------------------------------------
# Author: Benson Kenduiywo
# CRF
# ------------------------------------------------------------
import timeit
start_time = timeit.default_timer()
import numpy as np
import rasterio
import glob
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
beta = 10.0
eta = 1.0                  # exponent scale in Eq. 3.6
epsilon = 1e-6#1e-10
district = "Musanze"
root = '/cluster/archiving/GIZ/data/'
no_iterations = 20
NODATA_VALUE = -9999  # for probability rasters

# ---------------------------------------------------------------------
# Pairwise parameters: Eq. 3.6 (contrast-sensitive Potts)
#   w_same = β  * exp(-η * d_ij)
#   w_diff = β  / max(1 - exp(-η * d_ij), ε)
# where d_ij is Euclidean distance between features normalized by C (#bands).
# (Numerically-stable with expm1; optional cap for w_diff)
# ---------------------------------------------------------------------
def compute_pairwise_params_cs_potts(ref_image, pairwise_edges,
                                     beta=1.0, eta=1.0, epsilon=1e-10,
                                     wdiff_cap=None):
    print('Computing Eq. 3.6 pairwise parameters (w_same, w_diff)...', flush=True)
    im = ref_image
    if im.ndim == 2:
        im = im[..., np.newaxis]
    H, W, C = im.shape

    edges = np.asarray(pairwise_edges, dtype=np.int64)
    i1, j1, i2, j2 = edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3]

    f1 = im[i1, j1, :].astype(np.float64)
    f2 = im[i2, j2, :].astype(np.float64)

    finite = np.isfinite(f1) & np.isfinite(f2)
    diff = f1 - f2
    diff[~finite] = np.nan
    sum_diff2 = np.nansum(diff * diff, axis=1)  # (E,)

    # If an edge has 0 finite channels, treat distance as inf (max contrast)
    zero_finite = (finite.sum(axis=1) == 0)
    sum_diff2 = np.where(zero_finite, np.inf, sum_diff2)

    # Normalize by total number of bands/features C
    d_ij = np.sqrt(sum_diff2) # / np.sqrt(C)#float(C)

    # Stable denom: 1 - exp(-x) == -expm1(-x)
    exp_term = np.exp(-eta * d_ij)
    denom = -np.expm1(-eta * d_ij) # exactly equal to 1 - np.exp(-eta * d), but stable
    denom = np.maximum(denom, epsilon)

    w_same = beta * exp_term
    w_diff = beta / denom
    if wdiff_cap is not None:
        w_diff = np.minimum(w_diff, float(wdiff_cap))
    return w_same.astype(np.float64), w_diff.astype(np.float64)

# ---------------------------------------------------------------------
# Min-sum LBP with Eq. 3.6 pairwise potentials
# m_{p→q}(l_q) = min_{l_p}[ U_p(l_p) + Σ_{k∈N(p)\q} m_{k→p}(l_p) + V_{pq}(l_p,l_q) ]
# V_{pq}: diag=w_same(e), off-diag=w_diff(e) for edge e=(p,q)
# ---------------------------------------------------------------------
def loopy_belief_propagation(unary_potentials, pairwise_edges, w_same, w_diff,
                             height, width, n_classes, max_iter=no_iterations):
    print('LBP inference...', flush=True)
    edges = np.asarray(pairwise_edges, dtype=np.int64)
    E = edges.shape[0]

    messages = np.zeros((E, n_classes), dtype=np.float32)

    src = edges[:, :2]  # (i1, j1)
    dst = edges[:, 2:]  # (i2, j2)

    # Node -> indices of incoming edges
    node_to_in = {}
    for e, (i2, j2) in enumerate(dst):
        node_to_in.setdefault((i2, j2), []).append(e)

    # Reverse-edge index to exclude q→p when computing p→q
    edge_index = {(i1, j1, i2, j2): idx for idx, (i1, j1, i2, j2) in enumerate(edges)}
    rev_index = np.full(E, -1, dtype=np.int64)
    for e, (i1, j1, i2, j2) in enumerate(edges):
        rev_index[e] = edge_index.get((i2, j2, i1, j1), -1)

    # Precompute incoming-to-source lists
    incoming_to_src = []
    for e, (i1, j1) in enumerate(src):
        lst = node_to_in.get((i1, j1), [])
        r = rev_index[e]
        if r != -1 and r in lst:
            lst = [k for k in lst if k != r]
        incoming_to_src.append(np.array(lst, dtype=np.int64))

    for _ in range(max_iter):
        new_messages = np.zeros_like(messages)
        for e, (i1, j1, i2, j2) in enumerate(edges):
            inc_idx = incoming_to_src[e]
            if inc_idx.size:
                incoming = messages[inc_idx].sum(axis=0)
            else:
                incoming = np.zeros(n_classes, dtype=np.float32)  # <- FIX: vector, not scalar

            # Pairwise matrix for this edge (Eq. 3.6)
            pw = np.full((n_classes, n_classes), w_diff[e], dtype=np.float32)
            np.fill_diagonal(pw, w_same[e])

            energies = unary_potentials[i1, j1, :].astype(np.float32)[:, None] + \
                       incoming[:, None] + pw

            m = energies.min(axis=0)
            m -= m.min()  # stabilize
            new_messages[e] = m.astype(np.float32)

        messages = new_messages

    # Beliefs = unary + sum incoming messages
    beliefs = unary_potentials.copy().astype(np.float32)
    for e, (i1, j1, i2, j2) in enumerate(edges):
        beliefs[i2, j2, :] += messages[e, :]

    labels = np.argmin(beliefs, axis=2).astype(np.uint8)

    # Probabilities
    beliefs = np.clip(beliefs, -500, 500)
    probabilities = np.exp(-beliefs.astype(np.float32))
    sum_probs = probabilities.sum(axis=2, keepdims=True)
    probabilities = np.where(sum_probs > 1e-10, probabilities / (sum_probs + 1e-10), 1.0 / n_classes)
    probabilities[np.isnan(probabilities) | np.isinf(probabilities)] = 1.0 / n_classes

    print(f"Beliefs range: {beliefs.min():.3f}–{beliefs.max():.3f}", flush=True)
    print(f"Probabilities range: {probabilities.min():.3f}–{probabilities.max():.3f}", flush=True)
    return labels, probabilities

# ---------------------------------------------------------------------
# Worker: process one tile set (probs, initial labels, ref image)
# ---------------------------------------------------------------------
def process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path,
                              beta=1.0, eta=1.0, epsilon=1e-10, compression='lzw'):
    # Load probability image (expecting 4 bands)
    with rasterio.open(prob_image_path) as src:
        prob_image = src.read()  # (C, H, W)
        base_profile = src.profile
        if prob_image.shape[0] != 4:
            raise ValueError(f"Expected 4-channel probability image, got shape {prob_image.shape}")
        prob_image = np.transpose(prob_image, (1, 2, 0)).astype(np.float32)  # (H, W, C)

    # Load initial labeled image (single-band, uint8)
    with rasterio.open(label_image_path) as src:
        init_label_image = src.read(1)
        if init_label_image.ndim != 2:
            raise ValueError(f"Expected single-channel label image, got shape {init_label_image.shape}")

    # Load reference feature image
    with rasterio.open(ref_image_path) as src:
        ref_image = src.read()  # (B, H, W) or (H, W)
        print(f"Reference image {ref_image_path}: {src.count} bands, shape {ref_image.shape}", flush=True)
        if ref_image.ndim == 3:
            ref_image = np.transpose(ref_image, (1, 2, 0)).astype(np.float32)  # (H, W, B)
        else:
            ref_image = ref_image.astype(np.float32)[..., np.newaxis]         # (H, W, 1)

    # Dimensions & sanity checks
    height, width, n_classes = prob_image.shape
    if init_label_image.shape != (height, width):
        raise ValueError(f"Label image shape {init_label_image.shape} does not match probability image {(height, width)}")
    if ref_image.shape[:2] != (height, width):
        raise ValueError(f"Reference image shape {ref_image.shape[:2]} does not match probability image {(height, width)}")

    # Identify invalid probability vectors (any channel NaN / <0 / >1)
    prob_finite = np.isfinite(prob_image)
    invalid_val_mask = (~prob_finite) | (prob_image < 0.0) | (prob_image > 1.0)
    invalid_prob_any = np.any(invalid_val_mask, axis=2)
    if np.any(invalid_prob_any):
        print(f"Warning: {int(invalid_prob_any.sum())} invalid input probabilities in {prob_image_path}", flush=True)

    # No-data masks
    prob_nodata_mask = np.any(prob_image == NODATA_VALUE, axis=2)
    if ref_image.shape[2] > 1:
        ref_nodata_mask = np.any(~np.isfinite(ref_image), axis=2)
    else:
        ref_nodata_mask = ~np.isfinite(ref_image[:, :, 0])

    # Valid pixel mask (exclude any pixel with invalid probs or ref nodata)
    valid_mask = (init_label_image != 255) & (~invalid_prob_any) & (~prob_nodata_mask) & (~ref_nodata_mask)
    n_valid = int(valid_mask.sum())
    print(f"Valid pixels: {n_valid}, No-data in probs: {int(prob_nodata_mask.sum())}, "
          f"No-data in ref: {int(ref_nodata_mask.sum())}", flush=True)

    # If no valid pixels, skip tile gracefully
    if n_valid == 0:
        print("No valid pixels in this tile — skipping CRF and outputs.", flush=True)
        return None

    # Unary potentials = -log(prob) on valid pixels only (others set to huge)
    unary_potentials = np.full((height, width, n_classes), 1e10, dtype=np.float32)
    for c in range(n_classes):
        p = prob_image[:, :, c]
        pv = np.clip(p[valid_mask], 1e-10, 1.0).astype(np.float32)
        unary_potentials[:, :, c][valid_mask] = -np.log(pv).astype(np.float32)

    print(f"Unary potentials range: "
          f"{float(unary_potentials[valid_mask].min()):.3f}–{float(unary_potentials[valid_mask].max()):.3f}", flush=True)

    # Pairwise edges (bi-directional) over valid pixels
    pairwise_edges = []
    valid_idx = np.argwhere(valid_mask)
    vm = valid_mask  # local alias
    H, W = height, width
    for i, j in valid_idx:
        if i + 1 < H and vm[i + 1, j]:
            pairwise_edges.append((i, j, i + 1, j))  # down
            pairwise_edges.append((i + 1, j, i, j))  # up
        if j + 1 < W and vm[i, j + 1]:
            pairwise_edges.append((i, j, i, j + 1))  # right
            pairwise_edges.append((i, j + 1, i, j))  # left

    pairwise_edges = np.asarray(pairwise_edges, dtype=np.int64)
    if pairwise_edges.size == 0:
        print("No valid neighbor pairs — skipping CRF and outputs.", flush=True)
        return None

    # Eq. 3.6 pairwise parameters (diag=w_same, off-diag=w_diff)
    # Use a slightly larger epsilon and a soft cap to avoid runaway w_diff for tiny distances.
    w_same, w_diff = compute_pairwise_params_cs_potts(
        ref_image, pairwise_edges, beta=beta, eta=eta, epsilon=max(epsilon, 1e-6), wdiff_cap=1e6
    )
    print(f"w_same range: {w_same.min():.3f}–{w_same.max():.3f}; "
          f"w_diff range: {w_diff.min():.3f}–{w_diff.max():.3f}", flush=True)

    # Run Loopy Belief Propagation
    refined_labels, refined_probabilities = loopy_belief_propagation(
        unary_potentials, pairwise_edges, w_same, w_diff, height, width, n_classes, max_iter=no_iterations
    )

    # Apply no-data mask
    label_image = np.full((height, width), 255, dtype=np.uint8)
    label_image[valid_mask] = refined_labels[valid_mask]
    refined_probabilities = refined_probabilities.astype(np.float32)
    refined_probabilities[~valid_mask] = NODATA_VALUE

    # Validate probability sums on valid pixels
    prob_sums = np.sum(refined_probabilities, axis=2)
    invalid_probs = ((prob_sums < 0.99) | (prob_sums > 1.01)) & valid_mask
    if np.any(invalid_probs):
        print(f"Warning: Invalid probability sums at {int(invalid_probs.sum())} pixels, "
              f"sum range on valid: {float(prob_sums[valid_mask].min()):.3f}–{float(prob_sums[valid_mask].max()):.3f}", flush=True)

    # Prepare output profile for labels
    label_profile = base_profile.copy()
    for k in ("photometric", "colormap"):
        if k in label_profile:
            label_profile.pop(k)
    label_profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        height=height,
        width=width,
        compress=compression,
        tiled=True
    )

    # Save label image
    with rasterio.open(output_label_path, "w", **label_profile) as dst:
        dst.write(label_image, 1)
    print(f"Refined class label image saved to {output_label_path} with {compression} compression, "
          f"CRS {label_profile.get('crs')}", flush=True)

    # Prepare and save probability image
    output_prob_path = output_label_path.replace('_crf_label_tile_', '_crf_prob_tile_')
    prob_profile = base_profile.copy()
    for k in ("photometric", "colormap"):
        if k in prob_profile:
            prob_profile.pop(k)
    prob_profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=n_classes,
        nodata=NODATA_VALUE,
        height=height,
        width=width,
        compress="deflate",
        predictor=3,
        tiled=True
    )

    with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
        for c in range(n_classes):
            band_data = refined_probabilities[:, :, c]
            dst.write(band_data, c + 1)

    print(f"Refined probability image saved to {output_prob_path} with deflate compression, "
          f"CRS {prob_profile.get('crs')}", flush=True)
    return None

# ---------------------------------------------------------------------
# Main: dispatch tiles
# ---------------------------------------------------------------------
if __name__ == "__main__":
    probs_tiles = sorted(glob.glob(f'{root}outputs/{district}_tiles_with_2025/*_probs.tif'))
    label_tiles = sorted(glob.glob(f'{root}outputs/{district}_tiles_with_2025/*_pred.tif'))
    image_tiles = sorted(glob.glob(f'{root}{district}_B2025_v2_tiles/*.tif'))

    if not (len(probs_tiles) == len(label_tiles) == len(image_tiles)):
        raise ValueError(
            f"Tile count mismatch:\n"
            f"  probs_tiles: {len(probs_tiles)}\n"
            f"  label_tiles: {len(label_tiles)}\n"
            f"  image_tiles: {len(image_tiles)}"
        )

    n_cores = max(1, multiprocessing.cpu_count() - 10)
    print(f"Using {n_cores} CPU cores for parallel processing", flush=True)

    # Debug: Check input tile properties
    for i, (prob_tile, label_tile, ref_tile) in enumerate(zip(probs_tiles, label_tiles, image_tiles)):
        try:
            with rasterio.open(ref_tile) as src:
                print(f"Tile {i:04d} ({ref_tile}): {src.count} bands, shape {src.shape}, dtype {src.dtypes[0]}", flush=True)
            with rasterio.open(prob_tile) as src:
                print(f"Tile {i:04d} ({prob_tile}): {src.count} bands, shape {src.shape}, dtype {src.dtypes[0]}", flush=True)
            with rasterio.open(label_tile) as src:
                print(f"Tile {i:04d} ({label_tile}): {src.count} bands, shape {src.shape}, dtype {src.dtypes[0]}", flush=True)
        except Exception as e:
            print(f"Error inspecting tile {i:04d}: {e}", flush=True)

    tasks = []
    for i in range(len(image_tiles)):
        index = f"{i:04d}"
        prob_image_path = probs_tiles[i]
        label_image_path = label_tiles[i]
        ref_image_path = image_tiles[i]
        output_label_path = f'{root}outputs/CRF/{district}/{district}_crf_label_tile_{index}.tif'

        if not all(os.path.exists(p) for p in [prob_image_path, label_image_path, ref_image_path]):
            print(f"Skipping tile {index}: One or more input files missing", flush=True)
            continue

        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        tasks.append((prob_image_path, label_image_path, ref_image_path, output_label_path,
                      beta, eta, epsilon, 'lzw'))

    if not tasks:
        print("No tasks to run. Check your input folders and patterns.", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(process_probability_image, *task) for task in tasks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing tile: {e}", flush=True)

    print(f"Done! Elapsed time (hours): {(timeit.default_timer() - start_time) / 3600.0:.3f}", flush=True)

'''
if __name__ == "__main__":
    probs_tiles = glob.glob(f'{root}data/outputs/{district}_tiles_with_2025/*_probs.tif')
    label_tiles = glob.glob(f'{root}data/outputs/{district}_tiles_with_2025/*_pred.tif')
    image_tiles = glob.glob(f'{root}data/{district}_B2025_v2_tiles/*.tif')  # You forgot glob.glob here
    
    # Check that all lists have the same length
    if not (len(probs_tiles) == len(label_tiles) == len(image_tiles)):
        raise ValueError(
            f"Tile count mismatch:\n"
            f"  probs_tiles: {len(probs_tiles)}\n"
            f"  label_tiles: {len(label_tiles)}\n"
            f"  image_tiles: {len(image_tiles)}"
        )
    for i in range(image_tiles):
        index = f"{i:04d}"
        prob_image_path = f'{root}{district}_tiles_with_2025/tile_{index}_probs.tif'
        label_image_path = f'{root}{district}_tiles_with_2025/tile_{index}_pred.tif'
        ref_image_path = f'{root}{district}_B2025_v2_tiles/tile_{index}.tif'
        output_label_path = f'{root}CRF/{district}/{district}_crf_label_tile_{index}.tif'
        process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, beta, epsilon)
    print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0, flush=True)


import numpy as np
import rasterio
#inputs tile_0103.tif
district = "Musanze"
root = '/cluster/archiving/GIZ/'
data_tiles = f'{root}data/{district}_B2025_v2_tiles/tile_0103.tif'
node_probs = f'{root}data/outputs/{district}_tiles_with_2025/tile_0103_probs.tif'
node_label = f'{root}data/outputs/{district}_tiles_with_2025/tile_0103_pred.tif'

def compute_pairwise_weights(ref_image, pairwise_edges, beta=1.0, epsilon=1e-10):
    """
    Compute pairwise weights as beta / (Euclidean_distance + epsilon).
    
    Args:
        ref_image: Reference image (height x width, single or multi-channel, may contain NaNs).
        pairwise_edges: List of edge pairs [(i1, j1, i2, j2), ...].
        beta: Smoothness prior term.
        epsilon: Small constant to avoid division by zero.
    
    Returns:
        pairwise_weights: Array of weights for each edge.
    """
    print('Computing pairwise interactions as beta / (Euclidean_distance + epsilon).', flush =True)
    # Ensure ref_image is at least 3D (add channel dimension for single-channel)
    if len(ref_image.shape) == 2:
        ref_image = ref_image[..., np.newaxis]
    
    n_edges = len(pairwise_edges)
    pairwise_weights = np.zeros(n_edges)
    
    # Compute valid Euclidean distances to estimate max distance for NaN handling
    valid_dists = []
    for i1, j1, i2, j2 in pairwise_edges:
        val1, val2 = ref_image[i1, j1], ref_image[i2, j2]
        if np.all(np.isfinite(val1)) and np.all(np.isfinite(val2)):
            dist = np.sqrt(np.sum((val1 - val2) ** 2))
            valid_dists.append(dist)
    
    max_dist = max(valid_dists) if valid_dists else 1.0  # Default if no valid distances
    
    # Compute weights
    for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
        val1, val2 = ref_image[i1, j1], ref_image[i2, j2]
        if np.any(np.isnan(val1)) or np.any(np.isnan(val2)):
            dist = max_dist  # Use max distance for NaN pairs
        else:
            dist = np.sqrt(np.sum((val1 - val2) ** 2))
        pairwise_weights[e] = beta / (dist + epsilon)  # Inverse distance with epsilon
    
    return pairwise_weights

def loopy_belief_propagation(unary_potentials, pairwise_edges, pairwise_weights, height, width, n_classes, max_iter=10):
    """
    Perform Loopy Belief Propagation for CRF inference.
    
    Args:
        unary_potentials: Array (height, width, n_classes) with unary costs (-log(prob)).
        pairwise_edges: List of edge pairs [(i1, j1, i2, j2), ...].
        pairwise_weights: Array of weights for each edge.
        height, width: Image dimensions.
        n_classes: Number of classes (4: 0, 1, 2, 3).
        max_iter: Number of LBP iterations.
    
    Returns:
        labels: Array (height, width) with inferred class labels.
    """
    print('LBP inference...', flush =True)
    messages = np.zeros((len(pairwise_edges), n_classes))
    beliefs = unary_potentials.copy()
    
    for _ in range(max_iter):
        new_messages = np.zeros_like(messages)
        for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
            pairwise = np.ones((n_classes, n_classes)) * pairwise_weights[e]
            np.fill_diagonal(pairwise, 0)  # Same labels have zero cost
            
            incoming = np.zeros(n_classes)
            for e_other, (i3, j3, i4, j4) in enumerate(pairwise_edges):
                if (i4, j4) == (i2, j2) and (i3, j3) != (i1, j1):
                    incoming += messages[e_other]
            
            for l2 in range(n_classes):
                min_energy = np.inf
                for l1 in range(n_classes):
                    energy = unary_potentials[i1, j1, l1] + pairwise[l1, l2] + incoming[l2]
                    min_energy = min(min_energy, energy)
                new_messages[e, l2] = min_energy
        
        messages = new_messages
        beliefs = unary_potentials.copy()
        for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
            beliefs[i2, j2, :] += messages[e, :]
    
    return np.argmin(beliefs, axis=2).astype(np.uint8)

def process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, beta=1.0, epsilon=1e-10):
    # Load the 4-channel TIFF probability image
    with rasterio.open(prob_image_path) as src:
        prob_image = src.read()  # Shape: (channels, height, width)
        profile = src.profile  # Geospatial metadata
        if prob_image.shape[0] != 4:  # Channels-first format
            raise ValueError(f"Expected 4-channel probability image, got shape {prob_image.shape}")
        prob_image = np.transpose(prob_image, (1, 2, 0))  # Convert to (height, width, channels)
    
    # Load the initial labeled image
    with rasterio.open(label_image_path) as src:
        init_label_image = src.read(1)  # Read single band
        if len(init_label_image.shape) != 2:
            raise ValueError(f"Expected single-channel label image, got shape {init_label_image.shape}")
    
    # Load the reference image (single or multi-channel, may have NaNs)
    with rasterio.open(ref_image_path) as src:
        ref_image = src.read()  # Shape: (channels, height, width)
        if len(ref_image.shape) == 3:
            ref_image = np.transpose(ref_image, (1, 2, 0))  # Convert to (height, width, channels)
        else:
            ref_image = ref_image[..., np.newaxis]  # Add channel dimension if single band
    
    # Dynamically get dimensions from probability image
    height, width, n_classes = prob_image.shape
    if init_label_image.shape != (height, width):
        raise ValueError(f"Label image shape {init_label_image.shape} does not match probability image {height, width}")
    if ref_image.shape[:2] != (height, width):
        raise ValueError(f"Reference image shape {ref_image.shape[:2]} does not match probability image {height, width}")
    
    # Initialize the output label image with no-data value (255)
    label_image = np.full((height, width), 255, dtype=np.uint8)
    
    # Compute unary potentials: -log(prob) for valid pixels
    unary_potentials = np.zeros((height, width, n_classes))
    valid_mask = init_label_image != 255
    for c in range(n_classes):
        probs = prob_image[:, :, c].copy()
        probs[probs == 0] = 1e-10
        unary_potentials[:, :, c] = -np.log(probs)
        unary_potentials[~valid_mask, c] = 1e10
    
    # Define pairwise edges for 4-connected grid
    pairwise_edges = []
    for i in range(height):
        for j in range(width):
            if i < height - 1:
                pairwise_edges.append((i, j, i + 1, j))
            if j < width - 1:
                pairwise_edges.append((i, j, i, j + 1))
    
    # Compute pairwise weights using inverse Euclidean distance
    pairwise_weights = compute_pairwise_weights(ref_image, pairwise_edges, beta, epsilon)
    
    # Run Loopy Belief Propagation
    refined_labels = loopy_belief_propagation(unary_potentials, pairwise_edges, pairwise_weights, height, width, n_classes)
    
    # Apply no-data mask
    label_image[valid_mask] = refined_labels[valid_mask]
    
    # Prepare output profile with modified GeoTransform and projection
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.uint8,
        count=1,  # Single band for labels
        nodata=255,
        transform=rasterio.transform.from_origin(
            profile['transform'][0],  # Top-left x
            profile['transform'][3],  # Top-left y
            10.0,  # Pixel width
            10.0   # Pixel height (positive, as rasterio uses positive values)
        ),
        crs='EPSG:32736',  # WGS 84 / UTM Zone 36S
        compress='lzw'
    )
    
    # Save the labeled image as a GeoTIFF
    with rasterio.open(output_label_path, 'w', **output_profile) as dst:
        dst.write(label_image, 1)  # Write single-band result to first band
    
    print(f"Refined class label image saved to {output_label_path} with 10 m resolution, WGS 84 UTM Zone 36S", flush=True)
    
    return label_image

# Example usage
if __name__ == "__main__":
    # Input and output paths
    print('Starting CRF...', flush=True)
    prob_image_path = node_probs    # 4-channel probability TIFF (dynamic size)
    label_image_path = node_label   # Single-channel initial label TIFF
    ref_image_path = data_tiles     # Reference image (single or multi-channel)
    output_label_path = f"{district}_tiles_with_2025/tile_0103_crf_labels.tif"  # Output refined label TIFF
    beta = 1.0  # Smoothness prior
    epsilon = 1e-10  # Small constant to avoid division by zero
    
    # Process the image with CRF and LBP
    label_image = process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, beta, epsilon)
    print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0, flush=True)
'''