"""Reusable PyQt5 widgets for AnnotateEZ.

All widgets receive the live config dictionary and mutate it directly,
keeping the UI and config in sync without an intermediate model layer.
"""

import logging
from typing import Any, Dict, List

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QWidget,
)

from annotateez.config import DISPLAY_COLORS

logger = logging.getLogger(__name__)

# Maps label color names (as stored in config) to Qt global colors.
_COLOR_TO_QT: Dict[str, Qt.GlobalColor] = {
    "black": Qt.black,
    "red": Qt.red,
    "green": Qt.green,
    "blue": Qt.blue,
    "yellow": Qt.yellow,
    "magenta": Qt.magenta,
    "cyan": Qt.cyan,
    "white": Qt.white,
}


def _label_color(config: Dict[str, Any], label_id: int) -> QColor:
    """Return the QColor for a label ID looked up from config."""
    color_name = config["labels"][label_id]["color"]
    qt_color = _COLOR_TO_QT.get(color_name)
    if qt_color is None:
        logger.warning(
            "Unknown label color '%s' for label %d; defaulting to white.",
            color_name,
            label_id,
        )
        return QColor(Qt.white)
    return QColor(qt_color)


class Legend(QWidget):
    """Dropdown for selecting the active annotation label.

    Displays one entry per active label. Emits ``label_changed(label_id)``
    when the selection changes and updates ``config['active_label']`` in place.
    """

    label_changed = pyqtSignal(int)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        self._id_map: List[int] = []  # combo index → label_id

        self._combo = QComboBox()
        for i, label in enumerate(config["labels"]):
            if label["active"]:
                self._combo.addItem(label["name"])
                self._id_map.append(i)

        active = config.get("active_label", 1)
        if active in self._id_map:
            self._combo.setCurrentIndex(self._id_map.index(active))

        self._combo.currentIndexChanged.connect(self._on_changed)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Label:"))
        layout.addWidget(self._combo)
        self.setLayout(layout)

    def _on_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._id_map):
            label_id = self._id_map[idx]
            self._config["active_label"] = label_id
            self.label_changed.emit(label_id)

    def set_active_label(self, label_id: int) -> None:
        """Update the dropdown to reflect label_id if it is present."""
        if label_id in self._id_map:
            self._combo.setCurrentIndex(self._id_map.index(label_id))


class LabelWidget(QWidget):
    """Settings row for configuring a single label (name and active state).

    Shows a numeric ID, a color-styled name text box, and an active
    checkbox. Mutations go directly to ``config['labels'][label_id]``.
    """

    def __init__(self, config: Dict[str, Any], label_id: int) -> None:
        super().__init__()
        self._config = config
        self._id = label_id

        label_cfg = config["labels"][label_id]

        id_label = QLabel(str(label_id))
        id_label.setAlignment(Qt.AlignCenter)

        self.textbox = QLineEdit()
        self.textbox.setFixedSize(QSize(128, 24))
        self.textbox.setStyleSheet(
            f"QLineEdit{{background: {label_cfg['color']}}}"
        )
        self.textbox.setText(label_cfg["name"])
        self.textbox.textChanged.connect(self._on_text_changed)

        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet(
            "QCheckBox::indicator{width: 24px; height: 24px;}"
        )
        self.checkbox.setChecked(label_cfg["active"])
        self.checkbox.stateChanged.connect(self._on_state_changed)
        self._sync_enabled()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(id_label)
        layout.addWidget(self.textbox)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def _sync_enabled(self) -> None:
        self.textbox.setEnabled(self.checkbox.isChecked())

    def _on_state_changed(self) -> None:
        self._config["labels"][self._id]["active"] = self.checkbox.isChecked()
        self._sync_enabled()

    def _on_text_changed(self) -> None:
        self._config["labels"][self._id]["name"] = self.textbox.text()


class ChannelWidget(QWidget):
    """Settings row for mapping a single channel to a display color and gain.

    Shows the channel name, a display-color combo box, and a gain spin box.
    Updates ``config['channels'][channel_idx]`` in place.
    """

    def __init__(self, config: Dict[str, Any], channel_idx: int) -> None:
        super().__init__()
        self._config = config
        self._idx = channel_idx

        channel_cfg = config["channels"][channel_idx]

        name_label = QLabel(channel_cfg["name"])
        name_label.setFixedHeight(24)

        self.combo = QComboBox()
        self.combo.addItems(DISPLAY_COLORS)
        current = channel_cfg.get("display_color", "none")
        if current in DISPLAY_COLORS:
            self.combo.setCurrentText(current)
        self.combo.currentTextChanged.connect(self._on_color_changed)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.01, 1000.0)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(float(channel_cfg.get("gain", 1.0)))
        self.gain_spin.setFixedWidth(72)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(name_label)
        layout.addWidget(self.combo)
        layout.addWidget(self.gain_spin)
        self.setLayout(layout)

    def _on_color_changed(self, color: str) -> None:
        self._config["channels"][self._idx]["display_color"] = color

    def _on_gain_changed(self, value: float) -> None:
        self._config["channels"][self._idx]["gain"] = value


class TextBox(QWidget):
    """Labeled text input that updates a single config key on change.

    Preserves the existing value type: int config values are parsed as
    int; all others are stored as str. Invalid int input is silently
    ignored to avoid partial updates during typing.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        key: str,
        title: str,
        default: Any,
    ) -> None:
        super().__init__()
        self._config = config
        self._key = key

        title_label = QLabel(title)
        title_label.setFixedHeight(32)

        self.textbox = QLineEdit()
        self.textbox.setFixedSize(64, 32)
        self.textbox.setText(str(default))
        self.textbox.textChanged.connect(self._on_text_changed)

        layout = QHBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.textbox)
        self.setLayout(layout)

    def _on_text_changed(self) -> None:
        text = self.textbox.text()
        if not text:
            return
        if isinstance(self._config[self._key], int):
            try:
                self._config[self._key] = int(text)
            except ValueError:
                pass
        else:
            self._config[self._key] = text


class Pos(QWidget):
    """Tile widget displaying one event image with a label-colored border.

    Left-click assigns the current active label; right-click resets the
    label to 0 (class 0 / junk). Emits ``annotated(event_id, label)``
    after each click so the caller can update the backing DataFrame.
    """

    annotated = pyqtSignal(int, int)

    def __init__(
        self,
        config: Dict[str, Any],
        event_id: int,
        image: QImage,
        label: int,
    ) -> None:
        super().__init__()
        self._config = config
        self.event_id = event_id
        self.image = image
        self.label = label
        self.setFixedSize(QSize(config["tile_size"], config["tile_size"]))

    def reset(self, event_id: int, image: QImage, label: int) -> None:
        """Update tile content and trigger a repaint."""
        self.event_id = event_id
        self.image = image
        self.label = label
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = event.rect()
        painter.drawImage(r, self.image)
        pen = QPen(_label_color(self._config, self.label))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.drawRect(r)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self.label = 0
        elif event.button() == Qt.LeftButton:
            self.label = self._config["active_label"]
        else:
            return
        self.annotated.emit(self.event_id, self.label)
        self.update()


class SortPanel(QWidget):
    """Controls for sorting events by a DataFrame column.

    Provides a column dropdown, ascending/descending radio buttons,
    and an Apply button. Emits ``sort_requested(column, ascending)``
    when Apply is clicked.
    """

    sort_requested = pyqtSignal(str, bool)

    def __init__(self) -> None:
        super().__init__()

        self._col_combo = QComboBox()
        self._col_combo.setMinimumWidth(120)

        self._asc_btn = QRadioButton("Asc")
        self._desc_btn = QRadioButton("Desc")
        self._asc_btn.setChecked(True)

        apply_btn = QPushButton("Sort")
        apply_btn.pressed.connect(self._on_apply)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Sort by:"))
        layout.addWidget(self._col_combo)
        layout.addWidget(self._asc_btn)
        layout.addWidget(self._desc_btn)
        layout.addWidget(apply_btn)
        self.setLayout(layout)

    def set_columns(self, columns: List[str]) -> None:
        """Populate the column dropdown with DataFrame column names."""
        self._col_combo.clear()
        self._col_combo.addItems(columns)

    def _on_apply(self) -> None:
        col = self._col_combo.currentText()
        if col:
            self.sort_requested.emit(col, self._asc_btn.isChecked())
