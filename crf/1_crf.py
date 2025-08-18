# ------------------------------------------------------------
# Author: Benson Kenduiywo
# CRF
# ------------------------------------------------------------
import timeit
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
eta = 1.0  # exponent scale in Eq. 3.6
epsilon = 1e-6  # updated from 1e-10 per your code
district = "Musanze"
root = '/cluster/archiving/GIZ/data/'
no_iterations = 20
NODATA_VALUE = -9999  # for probability rasters
MAX_WEIGHT = 1e6  # Cap for pairwise weights to prevent numerical issues

# ---------------------------------------------------------------------
# Pairwise parameters: Eq. 3.6 (contrast-sensitive Potts)
#   w_same = β * exp(-η * d_ij)
#   w_diff = β / max(1 - exp(-η * d_ij), ε)
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
    d_ij = np.sqrt(sum_diff2) / float(C)

    # Stable denom: 1 - exp(-x) == -expm1(-x)
    exp_term = np.exp(-eta * d_ij)
    denom = -np.expm1(-eta * d_ij)  # exactly equal to 1 - np.exp(-eta * d), but stable
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
                incoming = np.zeros(n_classes, dtype=np.float32)

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

    # Probabilities #CAPS PROBABILITIES
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
def process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, beta=1.0, eta=1.0, epsilon=1e-10, compression='lzw'):
    try:
        # Load probability image
        with rasterio.open(prob_image_path) as src:
            prob_image = src.read()
            base_profile = src.profile
            n_classes = prob_image.shape[0]  # Dynamically extract number of classes
            prob_image = np.transpose(prob_image, (1, 2, 0))
        
        # Load initial labeled image
        with rasterio.open(label_image_path) as src:
            init_label_image = src.read(1)
            if len(init_label_image.shape) != 2:
                raise ValueError(f"Expected single-channel label image, got shape {init_label_image.shape}")
        
        # Load reference image
        with rasterio.open(ref_image_path) as src:
            ref_image = src.read()
            print(f"Reference image {ref_image_path}: {src.count} bands, shape {src.shape}", flush=True)
            if len(ref_image.shape) == 3:
                ref_image = np.transpose(ref_image, (1, 2, 0))
            else:
                ref_image = ref_image[..., np.newaxis]
            # Replace no-data (NaN or -9999) with 0 in reference image
            ref_image[np.isnan(ref_image) | (ref_image == NODATA_VALUE)] = 0
        
        # Validate dimensions
        height, width = prob_image.shape[:2]
        if init_label_image.shape != (height, width):
            raise ValueError(f"Label image shape {init_label_image.shape} does not match probability image {height, width}")
        if ref_image.shape[:2] != (height, width):
            raise ValueError(f"Reference image shape {ref_image.shape[:2]} does not match probability image {height, width}")
        
        # Clean input probabilities
        invalid_probs = np.any((prob_image < 0) | (prob_image > 1) | np.isnan(prob_image) | (prob_image == NODATA_VALUE), axis=2)
        if np.any(invalid_probs):
            print(f"Warning: {np.sum(invalid_probs)} invalid or no-data probabilities in {prob_image_path}", flush=True)
            prob_image[invalid_probs] = 1.0 / n_classes  # Assign uniform probability for invalid/no-data pixels
            prob_image = np.clip(prob_image, 0, 1)
        
        # Initialize output label image
        label_image = np.full((height, width), 255, dtype=np.uint8)
        
        # Create valid pixel mask
        valid_mask = init_label_image != 255
        print(f"Valid pixels (based on labels != 255): {np.sum(valid_mask)}", flush=True)
        
        # Compute unary potentials
        unary_potentials = np.zeros((height, width, n_classes))
        for c in range(n_classes):
            probs = prob_image[:, :, c].copy()
            probs[probs == 0] = 1e-10
            probs = np.clip(probs, 1e-10, 1.0)
            unary_potentials[:, :, c] = -np.log(probs)
            unary_potentials[~valid_mask, c] = 1e10  # High cost for invalid pixels to preserve no-data labels
        
        print(f"Unary potentials range: {np.min(unary_potentials[valid_mask]):.3f}–{np.max(unary_potentials[valid_mask]):.3f}", flush=True)
        
        # Define pairwise edges (include all pixels, as ref_image no-data is handled)
        pairwise_edges = []
        for i in range(height):
            for j in range(width):
                if i < height - 1:
                    pairwise_edges.append((i, j, i + 1, j))
                if j < width - 1:
                    pairwise_edges.append((i, j, i, j + 1))
        
        # Compute pairwise weights
        w_same, w_diff = compute_pairwise_params_cs_potts(ref_image, pairwise_edges, beta=beta, eta=eta, epsilon=epsilon, wdiff_cap=MAX_WEIGHT)
        
        # Run Loopy Belief Propagation
        refined_labels, refined_probabilities = loopy_belief_propagation(unary_potentials, pairwise_edges, w_same, w_diff, height, width, n_classes)
        
        # Apply no-data mask (preserve 255 for invalid pixels based on label_image)
        label_image[valid_mask] = refined_labels[valid_mask]
        refined_probabilities[~valid_mask] = NODATA_VALUE
        
        # Validate probabilities
        prob_sums = np.sum(refined_probabilities, axis=2)
        invalid_probs = (prob_sums < 0.99) | (prob_sums > 1.01) & valid_mask
        if np.any(invalid_probs):
            print(f"Warning: Invalid probability sums at {np.sum(invalid_probs)} pixels, range: {np.min(prob_sums[valid_mask]):.3f}–{np.max(prob_sums[valid_mask]):.3f}", flush=True)
        
        # Save label image
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
        
        with rasterio.open(output_label_path, "w", **label_profile) as dst:
            dst.write(label_image, 1)
        print(f"Refined class label image saved to {output_label_path} with {compression} compression, CRS {label_profile['crs']}", flush=True)
        
        # Save probability image
        output_prob_path = output_label_path.replace('_crf_label_tile_', '_crf_prob_tile_')
        prob_profile = base_profile.copy()
        for k in ("photometric", "colormap"):
            if k in prob_profile:
                prob_profile.pop(k)
        prob_profile.update(
            driver="GTiff",
            dtype=rasterio.float32,
            count=n_classes,  # Use dynamic n_classes
            nodata=NODATA_VALUE,
            height=height,
            width=width,
            compress="deflate",
            predictor=3,
            tiled=True
        )
        
        try:
            with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
                for c in range(n_classes):
                    band_data = refined_probabilities[:, :, c].copy()
                    band_data[~valid_mask] = NODATA_VALUE
                    if np.any(np.isnan(band_data) | np.isinf(band_data)):
                        print(f"Warning: NaN/Inf in probability band {c+1} for {output_prob_path}", flush=True)
                        band_data[np.isnan(band_data) | np.isinf(band_data)] = NODATA_VALUE
                    dst.write(band_data, c + 1)
            print(f"Refined probability image saved to {output_prob_path} with deflate compression, CRS {prob_profile['crs']}", flush=True)
        except Exception as e:
            print(f"Error saving probability image {output_prob_path}: {e}", flush=True)
            with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
                for c in range(n_classes):
                    dst.write(np.full((height, width), NODATA_VALUE, dtype=np.float32), c + 1)
            print(f"Fallback probability image saved to {output_prob_path} with all no-data values", flush=True)
        
        return label_image, refined_probabilities
    except Exception as e:
        print(f"Error processing {prob_image_path}: {e}", flush=True)
        height, width = init_label_image.shape if 'init_label_image' in locals() else (256, 256)
        label_image = np.full((height, width), 255, dtype=np.uint8)
        label_profile = base_profile.copy() if 'base_profile' in locals() else {}
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
        with rasterio.open(output_label_path, "w", **label_profile) as dst:
            dst.write(label_image, 1)
        print(f"Default label image saved to {output_label_path} due to error", flush=True)
        
        output_prob_path = output_label_path.replace('_crf_label_tile_', '_crf_prob_tile_')
        prob_profile = base_profile.copy() if 'base_profile' in locals() else {}
        prob_profile.update(
            driver="GTiff",
            dtype=rasterio.float32,
            count=n_classes if 'n_classes' in locals() else 4,  # Fallback to 4 if n_classes not defined
            nodata=NODATA_VALUE,
            height=height,
            width=width,
            compress="deflate",
            predictor=3,
            tiled=True
        )
        with rasterio.open(output_prob_path, "w", **prob_profile) as dst:
            for c in range(prob_profile['count']):
                dst.write(np.full((height, width), NODATA_VALUE, dtype=np.float32), c + 1)
        print(f"Default probability image saved to {output_prob_path} due to error", flush=True)
        
        return None, None

# ---------------------------------------------------------------------
# Main: dispatch tiles
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start_time = timeit.default_timer()
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