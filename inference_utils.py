import numpy as np

def sliding_window(image, patch_size, stride):
    """
    Generator that yields image patches using a sliding window.

    Args:
        image (ndarray): Input image of shape [C, H, W].
        patch_size (int): Height/width of each square patch.
        stride (int): Stride for moving the window.

    Yields:
        Tuple[int, int, ndarray]: Top-left (y, x) and the patch [C, patch_size, patch_size].
    """
    C, H, W = image.shape
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size]
            yield y, x, patch

def stitch_output(pred_map, count_map, pred_patch, y, x):
    """
    Combines patch prediction into final output map with overlap averaging.

    Args:
        pred_map (ndarray): Accumulator for predicted class indices [H, W].
        count_map (ndarray): Accumulator for counting overlaps [H, W].
        pred_patch (ndarray): Patch prediction [patch_size, patch_size].
        y (int): Top-left Y-coordinate of the patch.
        x (int): Top-left X-coordinate of the patch.
    """
    h, w = pred_patch.shape
    pred_map[y:y+h, x:x+w] += pred_patch
    count_map[y:y+h, x:x+w] += 1

def normalize_output(pred_map, count_map, ignore_value=255):
    """
    Normalize the prediction map by dividing with count map.

    Args:
        pred_map (ndarray): Accumulated prediction map [H, W].
        count_map (ndarray): Count of predictions per pixel [H, W].
        ignore_value (int): Value to assign where count == 0.

    Returns:
        ndarray: Normalized output map [H, W].
    """
    normalized = np.full(pred_map.shape, ignore_value, dtype=np.uint8)
    valid = count_map > 0
    normalized[valid] = (pred_map[valid] / count_map[valid]).round().astype(np.uint8)
    return normalized
