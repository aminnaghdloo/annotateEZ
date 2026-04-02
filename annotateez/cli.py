"""Command-line entry point for AnnotateEZ.

Invoked as ``annotate-ez`` (after pip install) or ``python -m annotateez``.
"""

import logging
import sys

from PyQt5.QtWidgets import QApplication

from annotateez.config import load_config
from annotateez.gui.main_window import MainWindow

_LOG_FORMAT_CONSOLE = "[%(levelname)s] %(message)s"
_LOG_FORMAT_FILE = "%(asctime)s: [%(levelname)s] %(message)s"


def _setup_logging(log_path: str = "annotate-ez.log") -> None:
    """Configure root logger with a console handler and a file handler."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT_CONSOLE))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT_FILE))
    root.addHandler(file_handler)


def main() -> None:
    """Launch the AnnotateEZ GUI."""
    _setup_logging()
    config = load_config()
    app = QApplication(sys.argv)
    _window = MainWindow(config)
    sys.exit(app.exec_())
