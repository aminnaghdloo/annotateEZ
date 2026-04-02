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

    Attributes:
        images: uint16 array of shape (N, H, W, C).
        df: DataFrame with per-event features, including a "label" column.
        masks: Integer array of shape (N, H, W), or None if not present.
        channel_names: Ordered list of channel name strings.
        path: Source HDF5 file path; used when saving.
        data_key: HDF5 key under which the DataFrame is stored.
    """

    def __init__(
        self,
        images: np.ndarray,
        df: pd.DataFrame,
        masks: Optional[np.ndarray],
        channel_names: List[str],
        path: Path,
        data_key: str,
    ) -> None:
        self.images = images
        self.df = df
        self.masks = masks
        self.channel_names = channel_names
        self.path = path
        self.data_key = data_key

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
        """Load an event set from an HDF5 file.

        Args:
            path: Path to the HDF5 file.
            image_key: Dataset key for the image array, expected shape
                (N, H, W, C) with dtype uint16.
            data_key: Dataset key for the pandas DataFrame stored via
                df.to_hdf().
            mask_key: Dataset key for the mask array, expected shape
                (N, H, W) or (N, H, W, 1). Pass None to skip masks.

        Returns:
            A fully populated EventSet instance.

        Raises:
            KeyError: If image_key or data_key is not found in the file.
            ValueError: If the image array is not 4-D.
        """
        with h5py.File(path, "r") as fh:
            keys = list(fh.keys())
            logger.debug("HDF5 keys in %s: %s", path.name, keys)

            if image_key not in keys:
                raise KeyError(
                    f"Image key '{image_key}' not found in {path.name}. "
                    f"Available keys: {keys}"
                )
            images = fh[image_key][:]
            if images.ndim != 4:
                raise ValueError(
                    f"Expected 4-D image array (N, H, W, C), "
                    f"got shape {images.shape}."
                )
            logger.info(
                "Loaded images: shape=%s, dtype=%s", images.shape, images.dtype
            )

            channel_names: List[str] = []
            if "channels" in keys:
                channel_names = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in fh["channels"][:]
                ]
                logger.debug("Channels: %s", channel_names)

            masks: Optional[np.ndarray] = None
            if mask_key and mask_key in keys:
                masks = fh[mask_key][:]
                if masks.ndim == 4 and masks.shape[3] == 1:
                    masks = masks[..., 0]
                logger.info(
                    "Loaded masks: shape=%s, dtype=%s",
                    masks.shape,
                    masks.dtype,
                )

        if data_key not in keys:
            raise KeyError(
                f"Data key '{data_key}' not found in {path.name}. "
                f"Available keys: {keys}"
            )
        df = pd.read_hdf(str(path), key=data_key)
        logger.info("Loaded features: shape=%s", df.shape)

        return cls(images, df, masks, channel_names, path, data_key)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, label_names: List[str]) -> None:
        """Write annotated features and the label map back to the source file.

        Overwrites the DataFrame at self.data_key and refreshes the
        "labels" dataset with the current label name mapping.

        Args:
            label_names: Ordered list of label class names to store in
                the "labels" dataset as UTF-8-encoded strings.
        """
        self.df.to_hdf(str(self.path), key=self.data_key, mode="r+")
        logger.debug("Wrote features to HDF5 key '%s'.", self.data_key)

        with h5py.File(self.path, "r+") as fh:
            if "labels" in fh:
                del fh["labels"]
            fh.create_dataset(
                "labels",
                data=[name.encode("utf-8") for name in label_names],
            )
        logger.info("Saved annotations and label map to %s.", self.path.name)
