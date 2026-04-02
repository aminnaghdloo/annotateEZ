"""Shared pytest fixtures for AnnotateEZ tests."""

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hdf5_file(tmp_path):
    """Minimal HDF5 file with images, channel names, and a features DataFrame.

    Layout:
        images   — (5, 8, 8, 2) uint16
        channels — [b"DAPI", b"CD45"]
        features — pandas DataFrame with columns: label (uint8), score (float)
    """
    path = tmp_path / "events.hdf5"
    n, h, w, c = 5, 8, 8, 2
    rng = np.random.default_rng(seed=0)
    images = rng.integers(0, 65535, (n, h, w, c), dtype=np.uint16)

    with h5py.File(path, "w") as fh:
        fh.create_dataset("images", data=images)
        fh.create_dataset("channels", data=[b"DAPI", b"CD45"])

    df = pd.DataFrame(
        {"label": np.zeros(n, dtype=np.uint8), "score": np.ones(n, dtype=np.float32)}
    )
    df.to_hdf(str(path), key="features", mode="r+")
    return path
