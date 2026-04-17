"""
CameraProjekt1 — Entry point.
Launches the PyQt6 application with a dark Fusion palette.
"""

import sys

# torch musí byť importovaný pred PyQt6 — na Windows Qt zmení DLL search path
# a spôsobí WinError 1114 pri neskoršom načítaní c10.dll.
try:
    import torch as _torch  # noqa: F401
except Exception:
    pass  # GPU/torch nie je dostupný — chyba sa objaví až pri volaní run_dexined()

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt


def apply_dark_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
    # Disabled state
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.WindowText,
        QColor(100, 100, 100),
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled,
        QPalette.ColorRole.ButtonText,
        QColor(100, 100, 100),
    )
    app.setPalette(palette)


def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_palette(app)

    from core.profile_manager import ProfileManager
    from ui.main_window import MainWindow

    pm = ProfileManager()
    window = MainWindow(pm)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
