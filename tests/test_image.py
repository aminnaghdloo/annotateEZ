"""Tests for annotateez.core.image."""

import numpy as np
import pytest

from annotateez.core.image import (
    channels_to_rgb8,
    extract_boundaries,
    overlay_mask_boundaries,
)


# --- channels_to_rgb8 ---

def test_channels_to_rgb8_red_channel():
    """A red channel maps to the R component only."""
    images = np.full((1, 2, 2, 1), 512, dtype=np.uint16)  # 512 / 256 = 2
    result = channels_to_rgb8(images, ["red"])

    assert result.shape == (1, 2, 2, 3)
    assert result.dtype == np.uint8
    assert result[0, 0, 0, 0] == 2   # R
    assert result[0, 0, 0, 1] == 0   # G
    assert result[0, 0, 0, 2] == 0   # B


def test_channels_to_rgb8_green_channel():
    images = np.full((1, 1, 1, 1), 256, dtype=np.uint16)
    result = channels_to_rgb8(images, ["green"])

    assert result[0, 0, 0, 0] == 0   # R
    assert result[0, 0, 0, 1] == 1   # G
    assert result[0, 0, 0, 2] == 0   # B


def test_channels_to_rgb8_blue_channel():
    images = np.full((1, 1, 1, 1), 256, dtype=np.uint16)
    result = channels_to_rgb8(images, ["blue"])

    assert result[0, 0, 0, 2] == 1


def test_channels_to_rgb8_gray_contributes_to_all():
    """A gray channel contributes equally to R, G, and B."""
    images = np.full((1, 1, 1, 1), 256, dtype=np.uint16)
    result = channels_to_rgb8(images, ["gray"])

    assert result[0, 0, 0].tolist() == [1, 1, 1]


def test_channels_to_rgb8_none_channel_skipped():
    """A 'none' channel contributes nothing to the output."""
    images = np.full((1, 1, 1, 1), 65535, dtype=np.uint16)
    result = channels_to_rgb8(images, ["none"])

    assert result[0, 0, 0].tolist() == [0, 0, 0]


def test_channels_to_rgb8_accumulation():
    """Two red channels accumulate before clipping."""
    images = np.full((1, 1, 1, 2), 256, dtype=np.uint16)
    result = channels_to_rgb8(images, ["red", "red"])

    assert result[0, 0, 0, 0] == 2   # 512 / 256 = 2


def test_channels_to_rgb8_clips_overflow():
    """Accumulated values above 65535 are clipped, giving max uint8 output."""
    images = np.full((1, 1, 1, 2), 40000, dtype=np.uint16)
    result = channels_to_rgb8(images, ["green", "green"])

    assert result[0, 0, 0, 1] == 255   # 80000 → clipped to 65535 → 255


def test_channels_to_rgb8_gain_amplifies():
    """A gain > 1 multiplies the channel value before accumulation."""
    images = np.full((1, 1, 1, 1), 256, dtype=np.uint16)  # → 1 without gain
    result = channels_to_rgb8(images, ["red"], channel_gains=[2.0])

    assert result[0, 0, 0, 0] == 2   # 256 * 2 / 256 = 2


def test_channels_to_rgb8_gain_clips():
    """Gain that overflows uint16 range is clipped to 255 in uint8."""
    images = np.full((1, 1, 1, 1), 40000, dtype=np.uint16)
    result = channels_to_rgb8(images, ["red"], channel_gains=[10.0])

    assert result[0, 0, 0, 0] == 255   # 400000 → clipped to 65535 → 255


def test_channels_to_rgb8_raises_not_4d():
    images = np.zeros((4, 4, 1), dtype=np.uint16)
    with pytest.raises(ValueError, match="4-D"):
        channels_to_rgb8(images, ["red"])


def test_channels_to_rgb8_raises_color_length_mismatch():
    images = np.zeros((1, 4, 4, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="channel_colors"):
        channels_to_rgb8(images, ["red"])


# --- extract_boundaries ---

def test_extract_boundaries_all_background():
    mask = np.zeros((4, 4), dtype=np.int32)
    assert not extract_boundaries(mask).any()


def test_extract_boundaries_single_fg_pixel():
    """A foreground pixel surrounded by background is a boundary."""
    mask = np.zeros((3, 3), dtype=np.int32)
    mask[1, 1] = 1
    b = extract_boundaries(mask)

    assert b[1, 1]
    assert not b[0, 0]


def test_extract_boundaries_interior_not_boundary():
    """Interior pixel with all same-valued neighbors is not a boundary."""
    mask = np.ones((3, 3), dtype=np.int32)
    b = extract_boundaries(mask)

    assert not b[1, 1]   # center — all neighbors are 1


def test_extract_boundaries_edge_pixels_always_boundary():
    """Edge foreground pixels are always boundaries."""
    mask = np.ones((3, 3), dtype=np.int32)
    b = extract_boundaries(mask)

    assert b[0, 0]
    assert b[0, 2]
    assert b[2, 0]
    assert b[2, 2]


def test_extract_boundaries_two_cells():
    """Pixels on the border between two cells are boundaries for both."""
    mask = np.array([[1, 1, 2, 2]], dtype=np.int32)
    b = extract_boundaries(mask)

    # pixel at col 1 (cell 1, next to cell 2) is a boundary
    assert b[0, 1]
    # pixel at col 2 (cell 2, next to cell 1) is a boundary
    assert b[0, 2]


def test_extract_boundaries_raises_not_2d():
    with pytest.raises(ValueError, match="2-D"):
        extract_boundaries(np.zeros((3, 3, 1), dtype=np.int32))


# --- overlay_mask_boundaries ---

def test_overlay_binary_default_white():
    """Binary mask (max=1) boundaries are drawn in white by default."""
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[2, 2] = 1
    result = overlay_mask_boundaries(rgb, mask)

    assert result[2, 2].tolist() == [255, 255, 255]


def test_overlay_binary_custom_color():
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[2, 2] = 1
    result = overlay_mask_boundaries(rgb, mask, binary_color=(255, 0, 0))

    assert result[2, 2].tolist() == [255, 0, 0]


def test_overlay_returns_copy():
    """overlay_mask_boundaries does not modify the input image."""
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=np.int32)
    result = overlay_mask_boundaries(rgb, mask)

    assert not np.shares_memory(result, rgb)


def test_overlay_background_pixels_unchanged():
    """Background pixels (mask == 0) are not colored."""
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = np.zeros((3, 3), dtype=np.int32)
    mask[1, 1] = 1   # single foreground pixel
    result = overlay_mask_boundaries(rgb, mask)

    # background pixel should still be black
    assert result[0, 0].tolist() == [0, 0, 0]


def test_overlay_raises_wrong_image_ndim():
    with pytest.raises(ValueError):
        overlay_mask_boundaries(
            np.zeros((5, 5), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.int32),
        )


def test_overlay_raises_wrong_mask_ndim():
    with pytest.raises(ValueError):
        overlay_mask_boundaries(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.zeros((5, 5, 1), dtype=np.int32),
        )
