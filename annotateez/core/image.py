"""Image conversion and mask boundary overlay utilities.

All functions operate on NumPy arrays and have no GUI dependencies,
making them straightforwardly testable.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maps a display color name to its RGB channel index.
# None means the channel contributes equally to all three (gray).
# -1 means the channel is skipped entirely (none).
_COLOR_TO_RGB_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "gray": None,
    "none": -1,
}

# Fixed palette of distinct RGB colors used for multi-cell mask overlays.
_CELL_COLORS: np.ndarray = np.array(
    [
        [255, 0, 0],    # red
        [0, 255, 0],    # green
        [0, 0, 255],    # blue
        [255, 255, 0],  # yellow
        [0, 255, 255],  # cyan
        [255, 0, 255],  # magenta
        [255, 128, 0],  # orange
        [128, 0, 255],  # purple
    ],
    dtype=np.uint8,
)


def channels_to_rgb8(
    images: np.ndarray,
    channel_colors: List[str],
) -> np.ndarray:
    """Convert multi-channel uint16 images to 8-bit RGB for display.

    Each channel is mapped to a display color and accumulated into the
    RGB output. Multiple channels can share the same color (their values
    are summed before clipping). Values are clipped to the uint16 range
    before downscaling to uint8 via integer division by 256.

    Args:
        images: Input array of shape (N, H, W, C) with dtype uint16.
        channel_colors: List of display colors of length C. Each entry
            must be one of "red", "green", "blue", "gray", or "none".

    Returns:
        RGB array of shape (N, H, W, 3) with dtype uint8.

    Raises:
        ValueError: If images is not 4-D or channel_colors length does
            not match the number of image channels.
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected 4-D array (N, H, W, C), got shape {images.shape}."
        )
    n, h, w, c = images.shape
    if len(channel_colors) != c:
        raise ValueError(
            f"channel_colors has {len(channel_colors)} entries but images "
            f"has {c} channels."
        )

    rgb = np.zeros((n, h, w, 3), dtype=np.float64)

    for i, color in enumerate(channel_colors):
        color = color.lower()
        if color not in _COLOR_TO_RGB_IDX:
            logger.warning(
                "Unknown display color '%s' for channel %d; skipping.", color, i
            )
            continue
        idx = _COLOR_TO_RGB_IDX[color]
        if idx == -1:
            continue
        channel = images[..., i].astype(np.float64)
        if idx is None:  # gray: contribute to all three channels equally
            rgb[..., 0] += channel
            rgb[..., 1] += channel
            rgb[..., 2] += channel
        else:
            rgb[..., idx] += channel

    np.clip(rgb, 0, 65535, out=rgb)
    return (rgb / 256).astype(np.uint8)


def extract_boundaries(mask: np.ndarray) -> np.ndarray:
    """Find boundary pixels in a 2-D integer mask.

    A foreground pixel (value > 0) is a boundary pixel if at least one of
    its 4-connected neighbors has a different value. Pixels on the image
    border are always considered boundary pixels when they are foreground.

    Args:
        mask: 2-D integer array where 0 is background and nonzero values
            identify cell regions.

    Returns:
        Boolean array of the same shape; True where boundary pixels are.

    Raises:
        ValueError: If mask is not 2-D.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}.")

    h, w = mask.shape
    fg = mask > 0

    # For edge pixels the "out-of-bounds" neighbor is treated as different,
    # so edge foreground pixels are always boundaries.
    diff_top = np.ones((h, w), dtype=bool)
    diff_top[1:, :] = mask[1:, :] != mask[:-1, :]

    diff_bottom = np.ones((h, w), dtype=bool)
    diff_bottom[:-1, :] = mask[:-1, :] != mask[1:, :]

    diff_left = np.ones((h, w), dtype=bool)
    diff_left[:, 1:] = mask[:, 1:] != mask[:, :-1]

    diff_right = np.ones((h, w), dtype=bool)
    diff_right[:, :-1] = mask[:, :-1] != mask[:, 1:]

    return fg & (diff_top | diff_bottom | diff_left | diff_right)


def overlay_mask_boundaries(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    binary_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw mask boundary lines on top of an RGB image.

    For binary masks (values 0 and 1) all boundaries are drawn in a
    single color (default: white). For instance-segmentation masks
    (values > 1) each cell ID is assigned a distinct color from a
    fixed palette.

    Args:
        rgb_image: Array of shape (H, W, 3) with dtype uint8.
        mask: 2-D integer array of shape (H, W).
        binary_color: RGB tuple used for binary-mask boundaries.
            Defaults to (255, 255, 255) (white).

    Returns:
        A copy of rgb_image with boundary pixels colored in place.

    Raises:
        ValueError: If rgb_image is not (H, W, 3) or mask is not 2-D.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            f"Expected (H, W, 3) image, got shape {rgb_image.shape}."
        )
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {mask.shape}.")

    result = rgb_image.copy()
    is_binary = mask.max() <= 1

    if is_binary:
        color = binary_color or (255, 255, 255)
        boundaries = extract_boundaries(mask)
        result[boundaries] = color
    else:
        all_boundaries = extract_boundaries(mask)
        for i, cell_id in enumerate(np.unique(mask)):
            if cell_id == 0:
                continue
            cell_boundaries = all_boundaries & (mask == cell_id)
            result[cell_boundaries] = _CELL_COLORS[i % len(_CELL_COLORS)]

    return result
