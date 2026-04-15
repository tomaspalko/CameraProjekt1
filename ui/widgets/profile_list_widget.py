"""
ProfileListWidget — QWidget so zoznamom profilov a CRUD tlačidlami.

Tlačidlá: New · Edit · Delete · Duplicate · Inspect
Delete zobrazuje QMessageBox potvrdenie.
refresh(profiles) repopuluje zoznam.

Signály:
  new_requested()
  edit_requested(int)       — id profilu
  delete_requested(int)     — id profilu
  duplicate_requested(int)  — id profilu
  inspect_requested(int)    — id profilu
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProfileListWidget(QWidget):
    """Zoznam profilov s CRUD akciami."""

    new_requested = pyqtSignal()
    edit_requested = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    duplicate_requested = pyqtSignal(int)
    inspect_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._profiles: list[dict] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # Verejné API
    # ------------------------------------------------------------------

    def refresh(self, profiles: list[dict]) -> None:
        """Repopuluje zoznam z listu profil-dictov (musia mať 'id' a 'name')."""
        self._profiles = list(profiles)
        self._list.clear()
        for p in self._profiles:
            item = QListWidgetItem(p["name"])
            item.setData(256, p["id"])  # UserRole = 256
            self._list.addItem(item)
        self._update_buttons()

    def selected_id(self) -> Optional[int]:
        """Vráti id aktuálne zvoleného profilu alebo None."""
        item = self._list.currentItem()
        if item is None:
            return None
        return item.data(256)

    # ------------------------------------------------------------------
    # Interné
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("Profily")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        # Tlačidlá — riadok 1
        row1 = QHBoxLayout()
        self._btn_new = QPushButton("Nový")
        self._btn_edit = QPushButton("Upraviť")
        self._btn_delete = QPushButton("Zmazať")
        for btn in (self._btn_new, self._btn_edit, self._btn_delete):
            row1.addWidget(btn)
        layout.addLayout(row1)

        # Tlačidlá — riadok 2
        row2 = QHBoxLayout()
        self._btn_duplicate = QPushButton("Duplikovať")
        self._btn_inspect = QPushButton("Inšpekcia")
        self._btn_inspect.setStyleSheet("background-color: #0078d7; color: white;")
        for btn in (self._btn_duplicate, self._btn_inspect):
            row2.addWidget(btn)
        layout.addLayout(row2)

        # Prepojenie signálov
        self._btn_new.clicked.connect(self._on_new)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_duplicate.clicked.connect(self._on_duplicate)
        self._btn_inspect.clicked.connect(self._on_inspect)

        self._update_buttons()

    def _update_buttons(self) -> None:
        has_selection = self._list.currentRow() >= 0
        for btn in (self._btn_edit, self._btn_delete, self._btn_duplicate, self._btn_inspect):
            btn.setEnabled(has_selection)

    def _on_selection_changed(self, _row: int) -> None:
        self._update_buttons()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        pid = item.data(256)
        if pid is not None:
            self.edit_requested.emit(pid)

    def _on_new(self) -> None:
        self.new_requested.emit()

    def _on_edit(self) -> None:
        pid = self.selected_id()
        if pid is not None:
            self.edit_requested.emit(pid)

    def _on_delete(self) -> None:
        pid = self.selected_id()
        if pid is None:
            return
        item = self._list.currentItem()
        name = item.text() if item else f"id={pid}"
        reply = QMessageBox.question(
            self,
            "Potvrdenie zmazania",
            f'Naozaj chces zmazat profil "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(pid)

    def _on_duplicate(self) -> None:
        pid = self.selected_id()
        if pid is not None:
            self.duplicate_requested.emit(pid)

    def _on_inspect(self) -> None:
        pid = self.selected_id()
        if pid is not None:
            self.inspect_requested.emit(pid)
