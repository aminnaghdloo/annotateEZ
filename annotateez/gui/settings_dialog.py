"""Settings dialog for AnnotateEZ.

Presents three sections — labels, channels, and configuration — and
persists the config to disk when the dialog is dismissed.
"""

import logging
from typing import Any, Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from annotateez.config import save_config
from annotateez.gui.widgets import ChannelWidget, LabelWidget, TextBox

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Modal dialog for editing labels, channel colors, and config values.

    All widgets mutate the config dict in place as the user types.
    Clicking Apply emits ``applied`` so the caller can re-render immediately
    without closing the dialog. Clicking Close (or the window X) saves config
    and dismisses the dialog.
    """

    applied = pyqtSignal()

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        self.setWindowTitle("Settings")
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # --- Labels section ---
        labels_title = QLabel("labels")
        labels_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(labels_title)
        for i in range(len(config["labels"])):
            layout.addWidget(LabelWidget(config, i))

        # --- Channels section (only when a file has been loaded) ---
        if config.get("channels"):
            channels_title = QLabel("channels")
            channels_title.setAlignment(Qt.AlignCenter)
            layout.addWidget(channels_title)
            for i in range(len(config["channels"])):
                layout.addWidget(ChannelWidget(config, i))

        # --- Configuration section ---
        config_title = QLabel("configuration")
        config_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(config_title)

        self._dir_dialog = QFileDialog()
        self._dir_dialog.setFileMode(QFileDialog.Directory)

        choose_btn = QPushButton("select output directory")
        choose_btn.setFixedHeight(32)
        choose_btn.pressed.connect(self._choose_output_dir)
        layout.addWidget(choose_btn)

        for key, title in [
            ("image_key", "image key"),
            ("data_key", "data key"),
            ("tile_size", "tile size [pixels]"),
            ("x_size", "horizontal tile count"),
            ("y_size", "vertical tile count"),
            ("max_undo_steps", "max undo steps"),
        ]:
            layout.addWidget(TextBox(config, key, title, config[key]))

        # --- Theme selector ---
        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("theme"))
        theme_combo = QComboBox()
        theme_combo.addItems(["dark", "light"])
        theme_combo.setCurrentText(config.get("theme", "dark"))
        theme_combo.currentTextChanged.connect(
            lambda t: config.__setitem__("theme", t)
        )
        theme_row.addWidget(theme_combo)
        layout.addLayout(theme_row)

        # --- Apply / Close buttons ---
        apply_btn = QPushButton("Apply")
        apply_btn.pressed.connect(self._on_apply)
        close_btn = QPushButton("Close")
        close_btn.pressed.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def _on_apply(self) -> None:
        self.applied.emit()
        logger.debug("Settings applied.")

    def _choose_output_dir(self) -> None:
        path = self._dir_dialog.getExistingDirectory(
            self, "Open Directory", "", QFileDialog.ShowDirsOnly
        )
        if path:
            self._config["output_dir"] = path
            logger.debug("Output directory set to: %s", path)

    def done(self, result: int) -> None:
        """Persist config to disk before closing, regardless of exit path."""
        save_config(self._config)
        logger.info("Settings saved.")
        super().done(result)
