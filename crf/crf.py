import timeit
start_time = timeit.default_timer()
import numpy as np
import rasterio
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

# Global variables
beta = 1.0
epsilon = 1e-10
district = "Musanze"
root = '/cluster/archiving/GIZ/data/outputs/'

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
    if len(ref_image.shape) == 2:
        ref_image = ref_image[..., np.newaxis]
    
    n_edges = len(pairwise_edges)
    pairwise_weights = np.zeros(n_edges)
    
    # Compute valid Euclidean distances
    valid_dists = []
    for i1, j1, i2, j2 in pairwise_edges:
        val1, val2 = ref_image[i1, j1], ref_image[i2, j2]
        if np.all(np.isfinite(val1)) and np.all(np.isfinite(val2)):
            dist = np.sqrt(np.sum((val1 - val2) ** 2))
            valid_dists.append(dist)
    
    max_dist = max(valid_dists) if valid_dists else 1.0
    
    # Vectorized weight computation
    for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
        val1, val2 = ref_image[i1, j1], ref_image[i2, j2]
        if np.any(np.isnan(val1)) or np.any(np.isnan(val2)):
            dist = max_dist
        else:
            dist = np.sqrt(np.sum((val1 - val2) ** 2))
        pairwise_weights[e] = beta / (dist + epsilon)
    
    return pairwise_weights

def loopy_belief_propagation(unary_potentials, pairwise_edges, pairwise_weights, height, width, n_classes, max_iter=10):
    """
    Perform Loopy Belief Propagation for CRF inference with partial vectorization.
    
    Args:
        unary_potentials: Array (height, width, n_classes) with unary costs (-log(prob)).
        pairwise_edges: List of edge pairs [(i1, j1, i2, j2), ...].
        pairwise_weights: Array of weights for each edge.
        height, width: Image dimensions.
        n_classes: Number of classes (4: 0, 1, 2, 3).
        max_iter: Number of LBP iterations (reduced to 5 for speed).
    
    Returns:
        labels: Array (height, width) with inferred class labels.
    """
    print('LBP inference...', flush =True)
    messages = np.zeros((len(pairwise_edges), n_classes))
    beliefs = unary_potentials.copy()
    
    # Precompute edge indices for faster neighbor lookup
    edge_map = {}
    for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
        edge_map.setdefault((i2, j2), []).append(e)
    
    for _ in range(max_iter):
        new_messages = np.zeros_like(messages)
        for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
            # Create pairwise potential matrix
            pairwise = np.ones((n_classes, n_classes)) * pairwise_weights[e]
            np.fill_diagonal(pairwise, 0)
            
            # Sum incoming messages for (i2, j2) excluding edge from (i1, j1)
            incoming = np.zeros(n_classes)
            if (i2, j2) in edge_map:
                for e_other in edge_map[(i2, j2)]:
                    i3, j3, _, _ = pairwise_edges[e_other]
                    if (i3, j3) != (i1, j1):
                        incoming += messages[e_other]
            
            # Vectorized energy computation for l2
            energies = unary_potentials[i1, j1, :, None] + pairwise + incoming[None, :]
            new_messages[e] = np.min(energies, axis=0)
        
        messages = new_messages
        beliefs = unary_potentials.copy()
        for e, (i1, j1, i2, j2) in enumerate(pairwise_edges):
            beliefs[i2, j2, :] += messages[e, :]
    
    return np.argmin(beliefs, axis=2).astype(np.uint8)

def process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, beta=1.0, epsilon=1e-10):
    NODATA_VALUE = -9999
    # Load the 4-channel TIFF probability image
    with rasterio.open(prob_image_path) as src:
        prob_image = src.read()  # Shape: (channels, height, width)
        profile = src.profile
        if prob_image.shape[0] != 4:
            raise ValueError(f"Expected 4-channel probability image, got shape {prob_image.shape}")
        prob_image = np.transpose(prob_image, (1, 2, 0))  # Convert to (height, width, channels)
    
    # Load the initial labeled image
    with rasterio.open(label_image_path) as src:
        init_label_image = src.read(1)
        if len(init_label_image.shape) != 2:
            raise ValueError(f"Expected single-channel label image, got shape {init_label_image.shape}")
    
    # Load the reference image
    with rasterio.open(ref_image_path) as src:
        ref_image = src.read()
        if len(ref_image.shape) == 3:
            ref_image = np.transpose(ref_image, (1, 2, 0))
        else:
            ref_image = ref_image[..., np.newaxis]
    
    # Dynamically get dimensions
    height, width, n_classes = prob_image.shape
    if init_label_image.shape != (height, width):
        raise ValueError(f"Label image shape {init_label_image.shape} does not match probability image {height, width}")
    if ref_image.shape[:2] != (height, width):
        raise ValueError(f"Reference image shape {ref_image.shape[:2]} does not match probability image {height, width}")
    
    # Initialize output label image
    label_image = np.full((height, width), 255, dtype=np.uint8)
    
    # Create valid pixel mask (no-data = 255 in label, -9999 in probs, or NaN in ref)
    valid_mask = init_label_image != 255
    prob_nodata_mask = np.any(prob_image == -NODATA_VALUE, axis=2)  
    if ref_image.shape[2] > 1:
        ref_nodata_mask = np.any(np.isnan(ref_image), axis=2)
    else:
        ref_nodata_mask = np.isnan(ref_image[:, :, 0])
    valid_mask = valid_mask & (~prob_nodata_mask) & (~ref_nodata_mask)
    
    # Compute unary potentials
    unary_potentials = np.zeros((height, width, n_classes))
    for c in range(n_classes):
        probs = prob_image[:, :, c].copy()
        probs[probs == -NODATA_VALUE] = 1e-10  # Replace no-data with small value for log
        probs[probs == 0] = 1e-10
        unary_potentials[:, :, c] = -np.log(probs)
        unary_potentials[~valid_mask, c] = 1e10
    
    # Define pairwise edges for valid pixels only
    pairwise_edges = []
    for i in range(height):
        for j in range(width):
            if not valid_mask[i, j]:
                continue
            if i < height - 1 and valid_mask[i + 1, j]:
                pairwise_edges.append((i, j, i + 1, j))
            if j < width - 1 and valid_mask[i, j + 1]:
                pairwise_edges.append((i, j, i, j + 1))
    
    # Compute pairwise weights
    pairwise_weights = compute_pairwise_weights(ref_image, pairwise_edges, beta, epsilon)
    
    # Run Loopy Belief Propagation
    refined_labels = loopy_belief_propagation(unary_potentials, pairwise_edges, pairwise_weights, height, width, n_classes)
    
    # Apply no-data mask
    label_image[valid_mask] = refined_labels[valid_mask]
    
    # Prepare output profile, retaining input CRS and transform
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=255,
        compress='lzw'
    )
    
    # Save the labeled image
    with rasterio.open(output_label_path, 'w', **output_profile) as dst:
        dst.write(label_image, 1)
    
    print(f"Refined class label image saved to {output_label_path} with input resolution and CRS {output_profile['crs']}, flush=True")
    
    #return label_image

if __name__ == "__main__":
    # Get list of tiles
    probs_tiles = sorted(glob.glob(f'{root}/{district}_tiles_with_2025/*_probs.tif'))
    label_tiles = sorted(glob.glob(f'{root}/{district}_tiles_with_2025/*_pred.tif'))
    image_tiles = sorted(glob.glob(f'{root}/../{district}_B2025_v2_tiles/*.tif'))

    # Check that all lists have the same length
    if not (len(probs_tiles) == len(label_tiles) == len(image_tiles)):
        raise ValueError(
            f"Tile count mismatch:\n"
            f"  probs_tiles: {len(probs_tiles)}\n"
            f"  label_tiles: {len(label_tiles)}\n"
            f"  image_tiles: {len(image_tiles)}"
        )

    # Determine number of CPU cores to use
    n_cores = max(1, multiprocessing.cpu_count() - 10)
    print(f"Using {n_cores} CPU cores for parallel processing", flush=True)

    # Prepare tasks for parallel execution
    tasks = []
    for i in range(len(image_tiles)):
        index = f"{i:04d}"
        prob_image_path = probs_tiles[i]
        label_image_path = label_tiles[i]
        ref_image_path = image_tiles[i]
        output_label_path = f'{root}/CRF/{district}/{district}_crf_label_tile_{index}.tif'
        
        # Verify that input files exist
        if not all(os.path.exists(p) for p in [prob_image_path, label_image_path, ref_image_path]):
            print(f"Skipping tile {index}: One or more input files missing", flush=True)
            continue
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        tasks.append((prob_image_path, label_image_path, ref_image_path, output_label_path, beta, epsilon))

    # Run tasks in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(process_probability_image, *task) for task in tasks]
        for future in futures:
            try:
                future.result()  # Wait for each task to complete and catch exceptions
            except Exception as e:
                print(f"Error processing tile: {e}", flush=True)

    print(f"Done! Elapsed time (hours): {(timeit.default_timer() - start_time) / 3600.0}", flush=True)


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