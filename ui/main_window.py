"""
MainWindow — hlavné okno aplikácie CameraProjekt1.

Rozloženie:
  ┌──────────────┬────────────────────────────────────┐
  │ ProfileList  │  StackedWidget                     │
  │  (~220 px)   │   - placeholder (pri štarte)       │
  │              │   - ProfileTab (edit/new)           │
  │              │   - InspectionTab (inšpekcia)       │
  └──────────────┴────────────────────────────────────┘

ProfileManager je zdieľaný — MainWindow ho vytvára a predáva tabom.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QWidget,
)

from core.profile_manager import ProfileManager
from ui.widgets.profile_list_widget import ProfileListWidget

# Lazy import tabov (aby sa vyhlo kruhovým importom)
def _import_profile_tab():
    from ui.profile_tab import ProfileTab
    return ProfileTab

def _import_inspection_tab():
    from ui.inspection_tab import InspectionTab
    return InspectionTab


# Index stránok v QStackedWidget
_PAGE_WELCOME = 0
_PAGE_PROFILE = 1
_PAGE_INSPECTION = 2


class MainWindow(QMainWindow):
    """Hlavné okno aplikácie."""

    def __init__(self, profile_manager: ProfileManager) -> None:
        super().__init__()
        self._pm = profile_manager
        self._current_profile_id: int | None = None
        self._build_ui()
        self._refresh_list()

    # ------------------------------------------------------------------
    # Zostrojenie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle("CameraProjekt1")
        self.setMinimumSize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Rozdeľovač: zoznam profilov | hlavný obsah
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        root_layout.addWidget(splitter)

        # Ľavý panel — zoznam profilov
        self._profile_list = ProfileListWidget()
        self._profile_list.setFixedWidth(220)
        self._profile_list.new_requested.connect(self._on_new_profile)
        self._profile_list.edit_requested.connect(self._on_edit_profile)
        self._profile_list.delete_requested.connect(self._on_delete_profile)
        self._profile_list.duplicate_requested.connect(self._on_duplicate_profile)
        self._profile_list.inspect_requested.connect(self._on_inspect_profile)
        splitter.addWidget(self._profile_list)

        # Pravý panel — stacked widget
        self._stack = QStackedWidget()
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Stránka 0 — uvítacia
        welcome = QLabel("Vyber profil alebo vytvor nový.")
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet("color: #888; font-size: 16px;")
        self._stack.addWidget(welcome)      # index 0

        # Stránky 1 a 2 sú tvorené dynamicky (lazy)
        self._profile_tab_widget: QWidget | None = None
        self._inspection_tab_widget: QWidget | None = None

    # ------------------------------------------------------------------
    # Refresh zoznamu
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        try:
            profiles = self._pm.list_profiles()
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa načítať profily:\n{exc}")
            profiles = []
        self._profile_list.refresh(profiles)

    # ------------------------------------------------------------------
    # Správa profilov
    # ------------------------------------------------------------------

    def _on_new_profile(self) -> None:
        try:
            profile = self._pm.create_profile()
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa vytvoriť profil:\n{exc}")
            return
        self._refresh_list()
        self._open_profile_tab(profile["id"])

    def _on_edit_profile(self, profile_id: int) -> None:
        self._open_profile_tab(profile_id)

    def _on_delete_profile(self, profile_id: int) -> None:
        try:
            self._pm.delete_profile(profile_id)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa zmazať profil:\n{exc}")
            return
        # Ak bol otvorený mazaný profil, choď na uvítaciu stránku
        if self._current_profile_id == profile_id:
            self._stack.setCurrentIndex(_PAGE_WELCOME)
            self._current_profile_id = None
        self._refresh_list()

    def _on_duplicate_profile(self, profile_id: int) -> None:
        try:
            new_profile = self._pm.duplicate_profile(profile_id)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa duplikovať profil:\n{exc}")
            return
        self._refresh_list()
        self._open_profile_tab(new_profile["id"])

    def _on_inspect_profile(self, profile_id: int) -> None:
        self._open_inspection_tab(profile_id)

    # ------------------------------------------------------------------
    # Prepínanie stránok
    # ------------------------------------------------------------------

    def _open_profile_tab(self, profile_id: int) -> None:
        self._current_profile_id = profile_id
        ProfileTab = _import_profile_tab()

        try:
            profile = self._pm.load_profile(profile_id)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa načítať profil:\n{exc}")
            return

        # Odstráň starý profile tab (ak existuje)
        if self._profile_tab_widget is not None:
            old_idx = self._stack.indexOf(self._profile_tab_widget)
            if old_idx >= 0:
                self._stack.removeWidget(self._profile_tab_widget)
            self._profile_tab_widget.deleteLater()
            self._profile_tab_widget = None

        tab = ProfileTab(profile, self._pm)
        tab.profile_saved.connect(self._on_profile_saved)
        self._profile_tab_widget = tab

        # Vložíme na index 1 (alebo kdekoľvek — addWidget vráti index)
        idx = self._stack.addWidget(tab)
        self._stack.setCurrentIndex(idx)

    def _open_inspection_tab(self, profile_id: int) -> None:
        self._current_profile_id = profile_id
        InspectionTab = _import_inspection_tab()

        try:
            profile = self._pm.load_profile(profile_id)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba", f"Nepodarilo sa načítať profil:\n{exc}")
            return

        # Odstráň starý inspection tab
        if self._inspection_tab_widget is not None:
            old_idx = self._stack.indexOf(self._inspection_tab_widget)
            if old_idx >= 0:
                self._stack.removeWidget(self._inspection_tab_widget)
            self._inspection_tab_widget.deleteLater()
            self._inspection_tab_widget = None

        tab = InspectionTab(profile, self._pm)
        self._inspection_tab_widget = tab

        idx = self._stack.addWidget(tab)
        self._stack.setCurrentIndex(idx)

    def _on_profile_saved(self, profile_id: int) -> None:
        """Slot volaný keď ProfileTab úspešne uloží profil."""
        self._refresh_list()
