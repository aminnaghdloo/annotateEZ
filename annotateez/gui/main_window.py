"""Main application window for AnnotateEZ."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QImage
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from annotateez.config import merge_channels
from annotateez.core.image import channels_to_rgb8, overlay_mask_boundaries
from annotateez.gui.settings_dialog import SettingsDialog
from annotateez.gui.widgets import Legend, Pos
from annotateez.io.eventset import EventSet

logger = logging.getLogger(__name__)

_ICON_DIR = Path(__file__).resolve().parent.parent / "icons"


def _tool_button(text: str, icon_name: str) -> QToolButton:
    """Create a standard fixed-size toolbar button with an icon."""
    btn = QToolButton()
    btn.setText(text)
    btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
    btn.setFixedSize(QSize(64, 64))
    btn.setIconSize(QSize(32, 32))
    btn.setIcon(QIcon(str(_ICON_DIR / icon_name)))
    return btn


class MainWindow(QMainWindow):
    """Main application window.

    Displays a paged grid of event-image tiles. Each tile shows a
    single event as an RGB image with a label-colored border.
    Toolbar buttons handle file I/O, page navigation, batch labeling,
    and settings.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        self._eventset: Optional[EventSet] = None
        self._rgb_images: Optional[np.ndarray] = None  # (N_padded, H, W, 3) uint8
        self._n_events: int = 0
        self._current_page: int = 0
        self._n_pages: int = 0
        self._f_name: str = "Empty"

        self._open_settings()
        self._build_ui()
        self._load_data()
        self.show()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _deploy_config(self) -> None:
        """Copy grid-dimension config values to instance attributes."""
        self._x_size: int = self._config["x_size"]
        self._y_size: int = self._config["y_size"]

    def _build_ui(self) -> None:
        self._deploy_config()

        self._file_dialog = QFileDialog()
        self._file_dialog.setFileMode(QFileDialog.AnyFile)

        self._grid = QGridLayout()
        self._grid.setSpacing(0)

        self._page_label = QLabel()
        self._page_label.setFixedSize(QSize(64, 64))
        self._page_label.setAlignment(Qt.AlignCenter)
        self._update_page_label()

        self._legend = Legend(self._config)

        select_all_btn = _tool_button("All", "check_all.png")
        select_all_btn.pressed.connect(self._select_all)

        select_none_btn = _tool_button("None", "uncheck_all.png")
        select_none_btn.pressed.connect(self._select_none)

        prev_btn = _tool_button("Prev", "Left.png")
        prev_btn.pressed.connect(self._prev_page)

        next_btn = _tool_button("Next", "Right.png")
        next_btn.pressed.connect(self._next_page)

        save_btn = _tool_button("Save", "Save.png")
        save_btn.pressed.connect(self._save_data)

        load_btn = _tool_button("Load", "Open.png")
        load_btn.pressed.connect(self._load_data)

        settings_btn = _tool_button("Settings", "Settings.png")
        settings_btn.pressed.connect(self._open_settings)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self._legend)
        toolbar.addWidget(select_all_btn)
        toolbar.addWidget(select_none_btn)
        toolbar.addWidget(prev_btn)
        toolbar.addWidget(self._page_label)
        toolbar.addWidget(next_btn)
        toolbar.addWidget(save_btn)
        toolbar.addWidget(load_btn)
        toolbar.addWidget(settings_btn)

        main_box = QVBoxLayout()
        main_box.addLayout(self._grid)
        main_box.addLayout(toolbar)

        central = QWidget()
        central.setLayout(main_box)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._config)
        dlg.applied.connect(self._apply_settings)
        dlg.exec_()
        self._deploy_config()

    def _apply_settings(self) -> None:
        """Re-render current images with updated channel settings."""
        if self._eventset is None:
            return
        self._deploy_config()
        self._rgb_images = self._render_rgb()
        self._reset_grid()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _tile_index(self, x: int, y: int) -> int:
        return (
            (self._current_page - 1) * self._x_size * self._y_size
            + x + self._x_size * y
        )

    def _make_qimage(self, event_id: int) -> QImage:
        img = self._rgb_images[event_id]
        h, w = img.shape[:2]
        return QImage(img.data, w, h, w * 3, QImage.Format_RGB888)

    def _get_label(self, event_id: int) -> int:
        if event_id >= self._n_events:
            return 0
        return int(self._eventset.df.label.iat[event_id])

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _init_grid(self) -> None:
        self._clear_grid()
        for x in range(self._x_size):
            for y in range(self._y_size):
                event_id = self._tile_index(x, y)
                w = Pos(
                    self._config,
                    event_id,
                    self._make_qimage(event_id),
                    self._get_label(event_id),
                )
                w.annotated.connect(self._on_annotated)
                self._grid.addWidget(w, y, x)

    def _reset_grid(self) -> None:
        for x in range(self._x_size):
            for y in range(self._y_size):
                event_id = self._tile_index(x, y)
                w = self._grid.itemAtPosition(y, x).widget()
                w.reset(event_id, self._make_qimage(event_id), self._get_label(event_id))

    def _update_page_label(self) -> None:
        self._page_label.setText(
            f"{self._f_name}\n\n{self._current_page} / {self._n_pages}"
        )

    # ------------------------------------------------------------------
    # Annotation sync
    # ------------------------------------------------------------------

    def _on_annotated(self, event_id: int, label: int) -> None:
        """Write a single tile annotation immediately to the DataFrame."""
        if event_id < self._n_events:
            self._eventset.df.label.iat[event_id] = label

    def _sync_grid_to_df(self) -> None:
        """Flush all visible tile labels to the DataFrame."""
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                if w.event_id < self._n_events:
                    self._eventset.df.label.iat[w.event_id] = w.label

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _next_page(self) -> None:
        if self._current_page < self._n_pages:
            self._current_page += 1
            logger.info("Page: %d", self._current_page)
        else:
            logger.warning("Already on the last page.")
        self._sync_grid_to_df()
        self._update_page_label()
        self._reset_grid()

    def _prev_page(self) -> None:
        if self._current_page > 1:
            self._current_page -= 1
            logger.info("Page: %d", self._current_page)
        else:
            logger.warning("Already on the first page.")
        self._sync_grid_to_df()
        self._update_page_label()
        self._reset_grid()

    # ------------------------------------------------------------------
    # Batch labeling
    # ------------------------------------------------------------------

    def _select_all(self) -> None:
        active = self._config["active_label"]
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                w.label = active
                w.update()
        self._sync_grid_to_df()

    def _select_none(self) -> None:
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                w.label = 0
                w.update()
        self._sync_grid_to_df()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_rgb(self) -> np.ndarray:
        """Convert raw images to padded uint8 RGB using current channel config."""
        channel_colors = [
            ch.get("display_color", "none") for ch in self._config["channels"]
        ]
        channel_gains = [
            float(ch.get("gain", 1.0)) for ch in self._config["channels"]
        ]
        rgb = channels_to_rgb8(self._eventset.images, channel_colors, channel_gains)

        if self._config.get("show_masks") and self._eventset.masks is not None:
            rgb = np.stack([
                overlay_mask_boundaries(rgb[i], self._eventset.masks[i])
                for i in range(len(rgb))
            ])

        n_padded = self._n_pages * self._x_size * self._y_size
        if n_padded > self._n_events:
            pad = np.zeros(
                (n_padded - self._n_events, rgb.shape[1], rgb.shape[2], 3),
                dtype=np.uint8,
            )
            rgb = np.concatenate((rgb, pad), axis=0)

        return rgb

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        f_path, _ = self._file_dialog.getOpenFileName(
            self, "Open File", "", "HDF files (*.hdf5)"
        )
        if not f_path:
            return

        path = Path(f_path)
        try:
            eventset = EventSet.load(
                path,
                image_key=self._config["image_key"],
                data_key=self._config["data_key"],
                mask_key=self._config.get("mask_key") or None,
            )
        except (KeyError, ValueError) as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        merge_channels(self._config, eventset.channel_names)

        if "label" not in eventset.df.columns:
            eventset.df["label"] = np.zeros(len(eventset.df), dtype=np.uint8)

        self._eventset = eventset
        self._n_events = len(eventset.df)
        self._n_pages = math.ceil(self._n_events / (self._x_size * self._y_size))
        self._rgb_images = self._render_rgb()
        self._f_name = path.stem
        self._current_page = 1
        self._update_page_label()
        self._init_grid()

    def _save_data(self) -> None:
        if self._eventset is None:
            logger.warning("No data loaded; nothing to save.")
            return
        self._sync_grid_to_df()
        label_names = [lbl["name"] for lbl in self._config["labels"]]
        self._eventset.save(label_names)

        export_path = Path(self._config["output_dir"]) / f"{self._f_name}.txt"
        self._eventset.df.to_csv(str(export_path), index=False, sep="\t")
        logger.info("Exported annotations to %s.", export_path)

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
