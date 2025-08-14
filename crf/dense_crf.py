'''
Installations:
1. conda install numpy cython
2. conda install -c conda-forge gcc gxx
3. conda install -c conda-forge pydensecrf
XXXXX NOT WORKIN YET
'''

import timeit
start_time = timeit.default_timer()
import numpy as np
import rasterio
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def process_probability_image(prob_image_path, label_image_path, ref_image_path, output_label_path, 
                             beta=1.0, epsilon=1e-10, gaussian_sxy=3, gaussian_compat=3, 
                             bilateral_sxy=80, bilateral_srgb=13, bilateral_compat=10):
    """
    Process a probability image using DenseCRF with pydensecrf, handling no-data pixels.

    Args:
        prob_image_path: Path to 4-channel probability TIFF (no-data = -9999).
        label_image_path: Path to single-channel initial label TIFF (no-data = 255).
        ref_image_path: Path to reference image TIFF (single or multi-channel, may have NaNs).
        output_label_path: Path to save output refined label TIFF (no-data = 255).
        beta: Smoothness prior (not used in DenseCRF, kept for compatibility).
        epsilon: Small constant (not used in DenseCRF, kept for compatibility).
        gaussian_sxy: Spatial standard deviation for Gaussian pairwise term.
        gaussian_compat: Compatibility for Gaussian pairwise term.
        bilateral_sxy: Spatial standard deviation for bilateral pairwise term.
        bilateral_srgb: RGB standard deviation for bilateral pairwise term.
        bilateral_compat: Compatibility for bilateral pairwise term.

    Returns:
        label_image: Array (height, width) with refined class labels.
    """
    # Load the 4-channel TIFF probability image
    with rasterio.open(prob_image_path) as src:
        prob_image = src.read()  # Shape: (channels, height, width)
        profile = src.profile
        nodata_value = profile.get('nodata', -9999)  # Default to -9999 if not specified
        if prob_image.shape[0] != 4:
            raise ValueError(f"Expected 4-channel probability image, got shape {prob_image.shape}")
        prob_image = np.ascontiguousarray(np.transpose(prob_image, (1, 2, 0)))  # (height, width, channels)

    # Load the initial labeled image
    with rasterio.open(label_image_path) as src:
        init_label_image = src.read(1)
        if len(init_label_image.shape) != 2:
            raise ValueError(f"Expected single-channel label image, got shape {init_label_image.shape}")

    # Load the reference image
    with rasterio.open(ref_image_path) as src:
        ref_image = src.read()
        if len(ref_image.shape) == 3:
            ref_image = np.ascontiguousarray(np.transpose(ref_image, (1, 2, 0)))  # (height, width, channels)
        else:
            ref_image = np.ascontiguousarray(ref_image[..., np.newaxis])  # (height, width, 1)

    # Dynamically get dimensions
    height, width, n_classes = prob_image.shape
    if init_label_image.shape != (height, width):
        raise ValueError(f"Label image shape {init_label_image.shape} does not match probability image {height, width}")
    if ref_image.shape[:2] != (height, width):
        raise ValueError(f"Reference image shape {ref_image.shape[:2]} does not match probability image {height, width}")

    # Initialize output label image
    label_image = np.full((height, width), 255, dtype=np.uint8)

    # Create valid pixel mask (no-data = 255 in label, nodata_value in probs, or NaN in ref)
    valid_mask = init_label_image != 255
    prob_nodata_mask = np.any(prob_image == nodata_value, axis=2)
    if ref_image.shape[2] > 1:
        ref_nodata_mask = np.any(np.isnan(ref_image), axis=2)
    else:
        ref_nodata_mask = np.isnan(ref_image[:, :, 0])
    valid_mask = valid_mask & (~prob_nodata_mask) & (~ref_nodata_mask)
    print(f"Valid pixels: {np.sum(valid_mask)} out of {height * width}", flush=True)

    # Prepare probability image for unary potentials
    valid_probs = prob_image.copy()
    valid_probs[valid_probs == nodata_value] = 0
    valid_probs[~valid_mask] = 0
    valid_probs[~valid_mask, 0] = 1  # Assign invalid pixels to class 0 with high confidence
    valid_probs = np.ascontiguousarray(valid_probs)  # Ensure C-contiguous
    valid_probs_transposed = np.ascontiguousarray(valid_probs.transpose(2, 0, 1))  # (n_classes, height, width)
    print(f"Probability array is C-contiguous: {valid_probs_transposed.flags['C_CONTIGUOUS']}", flush=True)

    # Prepare reference image for bilateral term (pydensecrf expects uint8 RGB)
    if ref_image.shape[2] == 1:
        # Replicate single channel to 3 channels
        ref_image_rgb = np.repeat(ref_image, 3, axis=2)
    elif ref_image.shape[2] >= 3:
        # Use first 3 channels for multichannel images
        ref_image_rgb = ref_image[:, :, :3]
    else:
        # Handle unexpected cases (e.g., 2 channels)
        raise ValueError(f"Reference image has unexpected number of channels: {ref_image.shape[2]}")
    
    # Replace NaNs and normalize to [0, 255]
    ref_image_rgb = np.nan_to_num(ref_image_rgb, nan=np.nanmean(ref_image_rgb))
    ref_image_rgb = (ref_image_rgb - np.min(ref_image_rgb)) / (np.max(ref_image_rgb) - np.min(ref_image_rgb) + 1e-10) * 255
    ref_image_rgb = np.ascontiguousarray(ref_image_rgb.astype(np.uint8))  # Ensure C-contiguous
    print(f"Reference image array is C-contiguous: {ref_image_rgb.flags['C_CONTIGUOUS']}", flush=True)

    # Set up DenseCRF
    print('DenseCRF inference...', flush=True)
    d = dcrf.DenseCRF2D(width, height, n_classes)
    d.setUnaryEnergy(unary_from_softmax(valid_probs_transposed))
    d.addPairwiseGaussian(sxy=gaussian_sxy, compat=gaussian_compat)
    d.addPairwiseBilateral(sxy=bilateral_sxy, srgb=bilateral_srgb, rgbim=ref_image_rgb, compat=bilateral_compat)

    # Run inference
    refined_probs = d.inference(5)  # 5 iterations for mean-field approximation
    refined_labels = np.argmax(refined_probs, axis=0).reshape(height, width).astype(np.uint8)

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

    print(f"Refined class label image saved to {output_label_path} with input resolution and CRS {output_profile['crs']}", flush=True)

    return label_image

# Example usage
if __name__ == "__main__":
    district = "Musanze"
    root = '/cluster/archiving/GIZ/'
    prob_image_path = f'{root}data/outputs/{district}_tiles_with_2025/tile_0103_probs.tif'
    label_image_path = f'{root}data/outputs/{district}_tiles_with_2025/tile_0103_pred.tif'
    ref_image_path = f'{root}data/{district}_B2025_v2_tiles/tile_0103.tif'
    output_label_path = f'{district}_densecrf_tile_0103_crf_labels_v2.tif'
    beta = 1.0  # Not used in DenseCRF, kept for compatibility
    epsilon = 1e-10  # Not used in Dense