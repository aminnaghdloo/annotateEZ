"""Tests for annotateez.io.eventset."""

import h5py
import numpy as np
import pandas as pd
import pytest

from annotateez.io.eventset import EventSet


# --- EventSet.load ---

def test_load_images_shape_and_dtype(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")

    assert es.images.shape == (5, 8, 8, 2)
    assert es.images.dtype == np.uint16


def test_load_dataframe(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")

    assert len(es.df) == 5
    assert "label" in es.df.columns


def test_load_channel_names(hdf5_file):
    """Byte-encoded channel names are decoded to str."""
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")

    assert es.channel_names == ["DAPI", "CD45"]


def test_load_no_masks_by_default(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")

    assert es.masks is None


def test_load_masks_3d(hdf5_file):
    """3-D mask arrays (N, H, W) are loaded as-is."""
    with h5py.File(hdf5_file, "r+") as fh:
        fh.create_dataset("masks", data=np.zeros((5, 8, 8), dtype=np.int32))

    es = EventSet.load(
        hdf5_file, image_key="images", data_key="features", mask_key="masks"
    )

    assert es.masks.shape == (5, 8, 8)


def test_load_masks_4d_squeezed(hdf5_file):
    """4-D masks of shape (N, H, W, 1) are squeezed to (N, H, W)."""
    with h5py.File(hdf5_file, "r+") as fh:
        fh.create_dataset("masks", data=np.zeros((5, 8, 8, 1), dtype=np.int32))

    es = EventSet.load(
        hdf5_file, image_key="images", data_key="features", mask_key="masks"
    )

    assert es.masks.shape == (5, 8, 8)


def test_load_raises_missing_image_key(hdf5_file):
    with pytest.raises(KeyError, match="missing_img"):
        EventSet.load(hdf5_file, image_key="missing_img", data_key="features")


def test_load_raises_missing_data_key(hdf5_file):
    with pytest.raises(KeyError, match="missing_df"):
        EventSet.load(hdf5_file, image_key="images", data_key="missing_df")


def test_load_raises_non_4d_images(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as fh:
        fh.create_dataset("images", data=np.zeros((4, 4, 1), dtype=np.uint16))
    df = pd.DataFrame({"label": [0]})
    df.to_hdf(str(path), key="features", mode="r+")

    with pytest.raises(ValueError, match="4-D"):
        EventSet.load(path, image_key="images", data_key="features")


def test_load_stores_path_and_data_key(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")

    assert es.path == hdf5_file
    assert es.data_key == "features"


# --- EventSet.save ---

def test_save_writes_label_column(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")
    es.df["label"] = np.array([1, 0, 2, 1, 0], dtype=np.uint8)
    es.save(["class0", "class1", "class2"])

    es2 = EventSet.load(hdf5_file, image_key="images", data_key="features")
    assert es2.df["label"].tolist() == [1, 0, 2, 1, 0]


def test_save_writes_label_names(hdf5_file):
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")
    es.save(["neg", "pos"])

    with h5py.File(hdf5_file, "r") as fh:
        names = [v.decode("utf-8") for v in fh["labels"][:]]
    assert names == ["neg", "pos"]


def test_save_overwrites_existing_labels(hdf5_file):
    """Calling save twice replaces the labels dataset."""
    es = EventSet.load(hdf5_file, image_key="images", data_key="features")
    es.save(["a", "b"])
    es.save(["x", "y", "z"])

    with h5py.File(hdf5_file, "r") as fh:
        names = [v.decode("utf-8") for v in fh["labels"][:]]
    assert names == ["x", "y", "z"]
