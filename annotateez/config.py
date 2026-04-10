"""Configuration management for AnnotateEZ.

Configuration is persisted in the package installation directory:
    <package_dir>/config.yml

The file is only created when the user saves settings. On first launch,
built-in defaults are used without writing any file.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH: Path = Path(__file__).parent / "config.yml"

#: Valid display colors for channel-to-RGB mapping.
DISPLAY_COLORS: List[str] = ["red", "green", "blue", "gray", "none"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "active_label": 1,
    "channels": [],
    "master_gain": 1.0,
    "max_undo_steps": 3,
    "theme": "dark",
    "data_key": "features",
    "image_key": "images",
    "mask_key": "masks",
    "show_masks": False,
    "labels": [
        {"name": "class 0", "color": "black", "active": False},
        {"name": "class 1", "color": "red", "active": True},
        {"name": "class 2", "color": "green", "active": True},
        {"name": "class 3", "color": "blue", "active": True},
        {"name": "class 4", "color": "yellow", "active": True},
        {"name": "class 5", "color": "magenta", "active": True},
        {"name": "class 6", "color": "cyan", "active": True},
    ],
    "output_dir": str(Path.home()),
    "tile_size": 85,
    "x_size": 15,
    "y_size": 15,
}


def load_config() -> Dict[str, Any]:
    """Load configuration from the package installation directory.

    Returns built-in defaults when no saved config exists; does not
    create a file on first launch. Ensures tile_size is odd, which is
    required for symmetric tile rendering.

    Returns:
        A dictionary containing the full application configuration.
    """
    if not CONFIG_PATH.exists():
        logger.info("No config file found; using built-in defaults")
        return copy.deepcopy(DEFAULT_CONFIG)

    with CONFIG_PATH.open("r") as fh:
        config = yaml.safe_load(fh)

    config["tile_size"] = _make_odd(
        config.get("tile_size", DEFAULT_CONFIG["tile_size"])
    )
    config.setdefault("master_gain", DEFAULT_CONFIG["master_gain"])
    config.setdefault("max_undo_steps", DEFAULT_CONFIG["max_undo_steps"])
    config.setdefault("theme", DEFAULT_CONFIG["theme"])
    logger.debug("Loaded config from %s", CONFIG_PATH)
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration to the package installation directory.

    Args:
        config: The configuration dictionary to save.
    """
    with CONFIG_PATH.open("w") as fh:
        yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)
    logger.debug("Saved config to %s", CONFIG_PATH)


def merge_channels(
    config: Dict[str, Any],
    channel_names: List[str],
) -> None:
    """Update the channel list in config to match names from a loaded file.

    Preserves existing display-color assignments for channels already in
    config. New channels default to "none". Modifies config in place.

    Args:
        config: The application configuration dictionary.
        channel_names: Ordered list of channel names read from an HDF5 file.
    """
    existing: Dict[str, Any] = {
        ch["name"]: ch
        for ch in config.get("channels", [])
    }
    config["channels"] = [
        {
            "name": name,
            "display_color": existing.get(name, {}).get("display_color", "none"),
            "gain": existing.get(name, {}).get("gain", 1.0),
        }
        for name in channel_names
    ]


def _make_odd(value: int) -> int:
    """Return value if odd, otherwise return value + 1."""
    return value if value % 2 == 1 else value + 1
