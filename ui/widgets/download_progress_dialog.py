"""
DownloadProgressDialog — modalne okno pre stiahnutie DexiNed modelu.

Stiahnutie prebieha v QThread, takze GUI zostane responzivne.
Podporuje zrusenie stiahnutia tlacidlom Cancel.

Pouzitie:
    dlg = DownloadProgressDialog(url, dest_path, parent=self)
    if dlg.exec() == QDialog.DialogCode.Accepted:
        print("Stiahnutie uspesne")
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class _DownloadThread(QThread):
    """QThread, ktory stiahne subor a reportuje priebeh."""

    progress_updated = pyqtSignal(int, int)   # (downloaded_bytes, total_bytes)
    download_finished = pyqtSignal()
    download_error = pyqtSignal(str)

    def __init__(self, url: str, dest_path: Path) -> None:
        super().__init__()
        self._url = url
        self._dest = dest_path
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        tmp_path = self._dest.with_suffix(".tmp")
        self._dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self._cancelled:
                raise InterruptedError("Stiahnutie zrusene uzivatelom.")

            def _reporthook(count: int, block_size: int, total_size: int) -> None:
                if self._cancelled:
                    raise InterruptedError("Stiahnutie zrusene uzivatelom.")
                downloaded = count * block_size
                self.progress_updated.emit(downloaded, total_size)

            urllib.request.urlretrieve(self._url, str(tmp_path), reporthook=_reporthook)

            if self._cancelled:
                raise InterruptedError("Stiahnutie zrusene uzivatelom.")

            if tmp_path.stat().st_size < 1_000_000:
                raise RuntimeError("Stiahnuty subor je prilis maly — chyba stiahnutia.")

            os.replace(tmp_path, self._dest)
            self.download_finished.emit()

        except Exception as exc:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            self.download_error.emit(str(exc))


class DownloadProgressDialog(QDialog):
    """
    Modalne okno so stiahnutim DexiNed modelu v pozadi.

    Vrati QDialog.DialogCode.Accepted po uspesnom stiahnutii,
    QDialog.DialogCode.Rejected pri zruseni alebo chybe.
    """

    def __init__(self, url: str, dest_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Stiahnutie DexiNed modelu")
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setFixedWidth(420)
        self.setModal(True)

        self._url = url
        self._dest = dest_path
        self._success = False
        self._thread: _DownloadThread | None = None

        self._build_ui()
        self._start_download()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._label_info = QLabel(
            "Stiahuje sa DexiNed model (~ 15 MB).\n"
            f"Zdroj: {self._url[:60]}..."
        )
        self._label_info.setWordWrap(True)
        layout.addWidget(self._label_info)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._label_size = QLabel("0 MB / ? MB")
        self._label_size.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._label_size)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self._on_cancel)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Stiahnutie
    # ------------------------------------------------------------------

    def _start_download(self) -> None:
        self._thread = _DownloadThread(self._url, self._dest)
        self._thread.progress_updated.connect(self._on_progress)
        self._thread.download_finished.connect(self._on_finished)
        self._thread.download_error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            pct = min(100, int(downloaded * 100 / total))
            self._progress.setValue(pct)
            dl_mb = downloaded / 1_048_576
            tot_mb = total / 1_048_576
            self._label_size.setText(f"{dl_mb:.1f} MB / {tot_mb:.1f} MB")
        else:
            dl_mb = downloaded / 1_048_576
            self._label_size.setText(f"{dl_mb:.1f} MB / ? MB")

    def _on_finished(self) -> None:
        self._success = True
        self._progress.setValue(100)
        self.accept()

    def _on_error(self, msg: str) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Chyba stiahnutia", msg)
        self.reject()

    def _on_cancel(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(3000)
        self.reject()

    def closeEvent(self, event) -> None:
        self._on_cancel()
        super().closeEvent(event)
