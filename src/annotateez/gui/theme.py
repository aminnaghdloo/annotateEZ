"""Theme management for AnnotateEZ.

Provides dark and light QPalette definitions and an ``apply_theme``
function that sets the application-wide palette and style.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication


def _dark_palette() -> QPalette:
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(45,  45,  45))
    p.setColor(QPalette.WindowText,      QColor(220, 220, 220))
    p.setColor(QPalette.Base,            QColor(30,  30,  30))
    p.setColor(QPalette.AlternateBase,   QColor(55,  55,  55))
    p.setColor(QPalette.ToolTipBase,     QColor(45,  45,  45))
    p.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
    p.setColor(QPalette.Text,            QColor(220, 220, 220))
    p.setColor(QPalette.Button,          QColor(60,  60,  60))
    p.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
    p.setColor(QPalette.BrightText,      Qt.red)
    p.setColor(QPalette.Link,            QColor(100, 160, 240))
    p.setColor(QPalette.Highlight,       QColor(70,  120, 200))
    p.setColor(QPalette.HighlightedText, Qt.white)
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(120, 120, 120))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
    return p


def _light_palette() -> QPalette:
    return QApplication.style().standardPalette()


def apply_theme(app: QApplication, theme: str) -> None:
    """Apply the named theme ('dark' or 'light') to the application.

    Args:
        app: The running QApplication instance.
        theme: Either 'dark' or 'light'.
    """
    app.setStyle("Fusion")
    if theme == "dark":
        app.setPalette(_dark_palette())
    else:
        app.setPalette(_light_palette())
