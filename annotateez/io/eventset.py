"""Load and save event sets stored in HDF5 format.

An event set is the combination of images, per-event features, optional
masks, and channel metadata that make up one annotatable dataset. All
direct h5py and pandas HDF5 interactions are contained here, keeping
the rest of the codebase free of low-level I/O details.
"""

import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EventSet:
    """An annotatable set of event images, features, and optional masks.

    The HDF5 file is kept open for the lifetime of the object so that
    images and masks can be read on demand rather than loaded all at once.
    Call ``close()`` (or use as a context manager) when done.

    Attributes:
        df: DataFrame with per-event features, including a "label" column.
        channel_names: Ordered list of channel name strings.
        path: Source HDF5 file path; used when saving.
        data_key: HDF5 key under which the DataFrame is stored.
        n_events: Total number of events (N).
        image_shape: Spatial and channel dimensions (H, W, C).
        image_dtype: numpy dtype of the raw image data.
    """

    def __init__(
        self,
        h5file: h5py.File,
        image_key: str,
        mask_key: Optional[str],
        df: pd.DataFrame,
        channel_names: List[str],
        path: Path,
        data_key: str,
    ) -> None:
        self._h5file = h5file
        self._image_key = image_key
        self._mask_key = mask_key
        self.df = df
        self.channel_names = channel_names
        self.path = path
        self.data_key = data_key

        ds = h5file[image_key]
        self.n_events: int = ds.shape[0]
        self.image_shape: tuple = ds.shape[1:]   # (H, W, C)
        self.image_dtype: np.dtype = ds.dtype

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path,
        image_key: str,
        data_key: str,
        mask_key: Optional[str] = None,
    ) -> "EventSet":
        """Open an HDF5 file and return a lazily-reading EventSet.

        Validation and channel-name metadata are read via a short-lived
        read-only handle; the DataFrame is then read while no h5py handle
        is open; finally the file is reopened in read-write mode for
        subsequent lazy image/mask access and in-place saving.

        Args:
            path: Path to the HDF5 file.
            image_key: Dataset key for the image array (N, H, W, C) uint16.
            data_key: Dataset key for the pandas DataFrame.
            mask_key: Dataset key for the mask array (N, H, W). None to skip.

        Returns:
            An EventSet with the HDF5 file open for reading.

        Raises:
            KeyError: If image_key or data_key is not found in the file.
            ValueError: If the image array is not 4-D.
        """
        # Phase 1: validate and read metadata with a short-lived handle.
        with h5py.File(path, "r") as fh:
            keys = list(fh.keys())
            logger.debug("HDF5 keys in %s: %s", path.name, keys)

            if image_key not in keys:
                raise KeyError(
                    f"Image key '{image_key}' not found in {path.name}. "
                    f"Available keys: {keys}"
                )
            if fh[image_key].ndim != 4:
                raise ValueError(
                    f"Expected 4-D image array (N, H, W, C), "
                    f"got shape {fh[image_key].shape}."
                )
            logger.info(
                "Opened images: shape=%s, dtype=%s",
                fh[image_key].shape,
                fh[image_key].dtype,
            )

            channel_names: List[str] = []
            if "channels" in keys:
                channel_names = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in fh["channels"][:]
                ]
                logger.debug("Channels: %s", channel_names)

            resolved_mask_key = None
            if mask_key and mask_key in keys:
                resolved_mask_key = mask_key
                logger.info("Masks available at key '%s'.", mask_key)

        # Phase 2: read DataFrame while h5py is closed (avoids file-lock conflict).
        if data_key not in keys:
            raise KeyError(
                f"Data key '{data_key}' not found in {path.name}. "
                f"Available keys: {keys}"
            )
        df = pd.read_hdf(str(path), key=data_key)
        logger.info("Loaded features: shape=%s", df.shape)

        # Phase 3: reopen in read-write mode for persistent lazy access.
        h5file = h5py.File(path, "r+")
        return cls(h5file, image_key, resolved_mask_key, df, channel_names, path, data_key)

    # ------------------------------------------------------------------
    # Lazy I/O
    # ------------------------------------------------------------------

    @property
    def has_masks(self) -> bool:
        """True if a mask dataset is available in the HDF5 file."""
        return self._mask_key is not None

    def read_images(self, indices: np.ndarray) -> np.ndarray:
        """Read a subset of images from the open HDF5 file.

        Args:
            indices: 1-D integer array of row indices to read.

        Returns:
            uint16 array of shape (len(indices), H, W, C).
        """
        sorted_idx = np.sort(indices)
        images = self._h5file[self._image_key][sorted_idx]
        restore = np.argsort(np.argsort(indices))
        return images[restore]

    def read_masks(self, indices: np.ndarray) -> Optional[np.ndarray]:
        """Read a subset of masks from the open HDF5 file.

        Args:
            indices: 1-D integer array of row indices to read.

        Returns:
            Integer array of shape (len(indices), H, W), or None if no masks.
        """
        if self._mask_key is None:
            return None
        sorted_idx = np.sort(indices)
        masks = self._h5file[self._mask_key][sorted_idx]
        if masks.ndim == 4 and masks.shape[3] == 1:
            masks = masks[..., 0]
        restore = np.argsort(np.argsort(indices))
        return masks[restore]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, label_names: List[str]) -> None:
        """Write annotated features and the label map back to the source file.

        The h5py handle is temporarily closed so that pandas (PyTables) can
        open the same file without a locking conflict, then reopened.

        Args:
            label_names: Ordered list of label class names.
        """
        # Close h5py so pandas can acquire the file lock.
        self._h5file.close()
        try:
            self.df.to_hdf(str(self.path), key=self.data_key, mode="r+")
            logger.debug("Wrote features to HDF5 key '%s'.", self.data_key)
        finally:
            self._h5file = h5py.File(self.path, "r+")

        if "labels" in self._h5file:
            del self._h5file["labels"]
        self._h5file.create_dataset(
            "labels",
            data=[name.encode("utf-8") for name in label_names],
        )
        self._h5file.flush()
        logger.info("Saved annotations and label map to %s.", self.path.name)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        if self._h5file and self._h5file.id.valid:
            self._h5file.close()
            logger.debug("Closed HDF5 file %s.", self.path.name)

    def __enter__(self) -> "EventSet":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
