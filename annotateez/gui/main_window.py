"""Main application window for AnnotateEZ."""

import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QIcon, QImage, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QShortcut,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from annotateez.config import merge_channels
from annotateez.core.image import channels_to_rgb8, overlay_mask_boundaries
from annotateez.gui.settings_dialog import SettingsDialog
from annotateez.gui.widgets import Legend, Pos, SortPanel
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
        self._n_events: int = 0
        self._current_page: int = 0
        self._n_pages: int = 0
        self._f_name: str = ""
        self._display_order: Optional[np.ndarray] = None
        self._drag_in_progress: bool = False
        self._page_cache: Dict[int, np.ndarray] = {}
        self._cache_lock: threading.Lock = threading.Lock()
        self._undo_history: np.ndarray = np.empty((0, 0))
        self._redo_history: np.ndarray = np.empty((0, 0))

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
        self.setWindowTitle("AnnotateEZ")

        self._file_dialog = QFileDialog()
        self._file_dialog.setFileMode(QFileDialog.AnyFile)

        self._grid = QGridLayout()
        self._grid.setSpacing(0)

        self._page_label = QLabel()
        self._page_label.setFixedSize(QSize(64, 64))
        self._page_label.setAlignment(Qt.AlignCenter)
        self._update_page_label()

        self._legend = Legend(self._config)

        self._sort_panel = SortPanel()
        self._sort_panel.sort_requested.connect(self._apply_sort)

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
        toolbar.addWidget(self._sort_panel)

        main_box = QVBoxLayout()
        main_box.addLayout(self._grid)
        main_box.addLayout(toolbar)

        central = QWidget()
        central.setLayout(main_box)
        self.setCentralWidget(central)

        self._setup_shortcuts()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self._prev_page)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self._next_page)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_data)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self._redo)
        for i in range(7):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                lambda checked=False, label_id=i: self._set_label(label_id)
            )

    def _set_label(self, label_id: int) -> None:
        """Set active label from keyboard shortcut, syncing the Legend widget."""
        self._config["active_label"] = label_id
        self._legend.set_active_label(label_id)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._config)
        dlg.applied.connect(self._apply_settings)
        dlg.exec_()
        self._deploy_config()

    def _apply_settings(self) -> None:
        """Rebuild the grid with updated settings."""
        if self._eventset is None:
            return
        self._deploy_config()
        self._n_pages = math.ceil(self._n_events / (self._x_size * self._y_size))
        self._current_page = min(self._current_page, max(1, self._n_pages))
        self._clear_page_cache()
        self._update_page_label()
        self._init_grid()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def _tile_index(self, x: int, y: int) -> int:
        return (
            (self._current_page - 1) * self._x_size * self._y_size
            + x + self._x_size * y
        )

    def _event_id(self, display_idx: int) -> int:
        """Map display position to DataFrame row index, respecting sort order."""
        if self._display_order is not None and display_idx < self._n_events:
            return int(self._display_order[display_idx])
        return display_idx

    def _make_qimage(self, page_images: np.ndarray, local_idx: int) -> QImage:
        img = np.ascontiguousarray(page_images[local_idx])
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
        page_images = self._get_page_images(self._current_page)
        for x in range(self._x_size):
            for y in range(self._y_size):
                local_idx = x + self._x_size * y
                event_id = self._event_id(self._tile_index(x, y))
                w = Pos(
                    self._config,
                    event_id,
                    self._make_qimage(page_images, local_idx),
                    self._get_label(event_id),
                )
                w.annotated.connect(self._on_annotated)
                w.drag_started.connect(self._on_drag_start)
                w.drag_moved.connect(self._on_drag_move)
                w.drag_ended.connect(self._on_drag_end)
                self._grid.addWidget(w, y, x)
        self._prefetch_adjacent()

    def _reset_grid(self) -> None:
        page_images = self._get_page_images(self._current_page)
        for x in range(self._x_size):
            for y in range(self._y_size):
                local_idx = x + self._x_size * y
                event_id = self._event_id(self._tile_index(x, y))
                w = self._grid.itemAtPosition(y, x).widget()
                w.reset(event_id, self._make_qimage(page_images, local_idx), self._get_label(event_id))

    def _update_page_label(self) -> None:
        self._page_label.setText(f"{self._current_page} / {self._n_pages}")

    # ------------------------------------------------------------------
    # Annotation sync
    # ------------------------------------------------------------------

    def _on_annotated(self, event_id: int, label: int) -> None:
        """Write a single tile annotation immediately to the DataFrame."""
        if self._eventset is None or event_id >= self._n_events:
            return
        if not self._drag_in_progress:
            old_label = int(self._eventset.df.label.iat[event_id])
            if old_label != label:
                self._push_undo()
        self._eventset.df.label.iat[event_id] = label

    def _on_drag_start(self) -> None:
        if self._eventset is None:
            return
        self._push_undo()
        self._drag_in_progress = True

    def _on_drag_move(self, global_pos: QPoint, label: int) -> None:
        widget = QApplication.widgetAt(global_pos)
        if not isinstance(widget, Pos) or widget.event_id >= self._n_events:
            return
        if widget.label != label:
            widget.label = label
            widget.update()
            self._eventset.df.label.iat[widget.event_id] = label

    def _on_drag_end(self) -> None:
        self._drag_in_progress = False

    def _sync_grid_to_df(self) -> None:
        """Flush all visible tile labels to the DataFrame."""
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                if w.event_id < self._n_events:
                    self._eventset.df.label.iat[w.event_id] = w.label

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Snapshot current labels into the undo history."""
        max_steps = self._config.get("max_undo_steps", 3)
        snapshot = self._eventset.df["label"].to_numpy(copy=True)
        self._undo_history = np.vstack([self._undo_history, snapshot[np.newaxis, :]])
        if len(self._undo_history) > max_steps:
            self._undo_history = self._undo_history[-max_steps:]
        self._redo_history = np.empty((0, self._n_events), dtype=snapshot.dtype)

    def _undo(self) -> None:
        if self._eventset is None or len(self._undo_history) == 0:
            return
        current = self._eventset.df["label"].to_numpy(copy=True)
        self._redo_history = np.vstack([self._redo_history, current[np.newaxis, :]])
        snapshot = self._undo_history[-1]
        self._undo_history = self._undo_history[:-1]
        self._eventset.df["label"] = snapshot
        self._reset_grid()

    def _redo(self) -> None:
        if self._eventset is None or len(self._redo_history) == 0:
            return
        max_steps = self._config.get("max_undo_steps", 3)
        current = self._eventset.df["label"].to_numpy(copy=True)
        self._undo_history = np.vstack([self._undo_history, current[np.newaxis, :]])
        if len(self._undo_history) > max_steps:
            self._undo_history = self._undo_history[-max_steps:]
        snapshot = self._redo_history[-1]
        self._redo_history = self._redo_history[:-1]
        self._eventset.df["label"] = snapshot
        self._reset_grid()

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    def _apply_sort(self, column: str, ascending: bool) -> None:
        """Reorder event display by a DataFrame column without mutating the data."""
        if self._eventset is None:
            return
        self._sync_grid_to_df()
        order = np.argsort(self._eventset.df[column].to_numpy(), kind="stable")
        self._display_order = order if ascending else order[::-1]
        self._clear_page_cache()
        self._current_page = 1
        self._update_page_label()
        self._init_grid()

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
        if self._eventset is None:
            return
        self._push_undo()
        active = self._config["active_label"]
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                w.label = active
                w.update()
        self._sync_grid_to_df()

    def _select_none(self) -> None:
        if self._eventset is None:
            return
        self._push_undo()
        for x in range(self._x_size):
            for y in range(self._y_size):
                w = self._grid.itemAtPosition(y, x).widget()
                w.label = 0
                w.update()
        self._sync_grid_to_df()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _get_page_event_ids(self, page_num: int) -> np.ndarray:
        """Return the DataFrame row indices for events shown on page_num."""
        page_size = self._x_size * self._y_size
        start = (page_num - 1) * page_size
        end = min(start + page_size, self._n_events)
        if self._display_order is not None:
            return self._display_order[start:end]
        return np.arange(start, end)

    def _render_page_rgb(self, page_num: int) -> np.ndarray:
        """Read and render images for page_num into a (page_size, H, W, 3) array."""
        page_size = self._x_size * self._y_size
        event_ids = self._get_page_event_ids(page_num)

        channel_colors = [
            ch.get("display_color", "none") for ch in self._config["channels"]
        ]
        channel_gains = [
            float(ch.get("gain", 1.0)) for ch in self._config["channels"]
        ]

        raw = self._eventset.read_images(event_ids)
        rgb = channels_to_rgb8(raw, channel_colors, channel_gains)

        if self._config.get("show_masks"):
            masks = self._eventset.read_masks(event_ids)
            if masks is not None:
                rgb = np.stack([
                    overlay_mask_boundaries(rgb[i], masks[i])
                    for i in range(len(rgb))
                ])

        # Pad to a full page if this is the last (short) page.
        if len(event_ids) < page_size:
            h, w = rgb.shape[1:3]
            pad = np.zeros((page_size - len(event_ids), h, w, 3), dtype=np.uint8)
            rgb = np.concatenate((rgb, pad), axis=0)

        return rgb

    def _get_page_images(self, page_num: int) -> np.ndarray:
        """Return cached RGB images for page_num, loading if necessary."""
        with self._cache_lock:
            if page_num not in self._page_cache:
                self._page_cache[page_num] = self._render_page_rgb(page_num)
            return self._page_cache[page_num]

    def _prefetch_adjacent(self) -> None:
        """Load the previous and next pages into cache in background threads."""
        for page in (self._current_page - 1, self._current_page + 1):
            if 1 <= page <= self._n_pages:
                threading.Thread(
                    target=self._prefetch_page, args=(page,), daemon=True
                ).start()

    def _prefetch_page(self, page_num: int) -> None:
        with self._cache_lock:
            if page_num not in self._page_cache:
                self._page_cache[page_num] = self._render_page_rgb(page_num)

    def _clear_page_cache(self) -> None:
        with self._cache_lock:
            self._page_cache.clear()

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

        if self._eventset is not None:
            self._eventset.close()

        self._eventset = eventset
        self._n_events = eventset.n_events
        self._n_pages = math.ceil(self._n_events / (self._x_size * self._y_size))
        self._display_order = None
        self._clear_page_cache()
        label_dtype = eventset.df["label"].dtype
        self._undo_history = np.empty((0, self._n_events), dtype=label_dtype)
        self._redo_history = np.empty((0, self._n_events), dtype=label_dtype)
        self._sort_panel.set_columns(list(eventset.df.columns))
        self._f_name = path.stem
        self._current_page = 1
        self.setWindowTitle(f"AnnotateEZ — {path.stem}")
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
            if self._eventset is not None:
                self._eventset.close()
            event.accept()
        else:
            event.ignore()
