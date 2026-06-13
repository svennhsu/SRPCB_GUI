import sys
import logging
from pathlib import Path

from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.main_window import APP_TITLE, APP_VERSION, MainWindow
from app.styles import application_stylesheet


def _build_light_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#F3F3F3"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#F6F8FA"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#007ACC"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#1E1E1E"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#007ACC"))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor("#5A4FCF"))
    return palette


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication([])
    app.setApplicationName(APP_TITLE)
    app.setApplicationVersion(APP_VERSION)

    app.setStyle("Fusion")
    app.setPalette(_build_light_palette())
    app.setStyleSheet(application_stylesheet())

    platform_name = app.platformName()
    logging.info("Qt platform: %s", platform_name)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
