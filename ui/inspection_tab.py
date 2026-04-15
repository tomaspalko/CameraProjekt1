"""
InspectionTab — Tab 2: Inšpekcia.
Implementácia: Fáza 8.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from core.profile_manager import ProfileManager


class InspectionTab(QWidget):
    """Záložka pre inšpekciu (stub — implementuje sa vo Fáze 8)."""

    def __init__(self, profile: dict, profile_manager: ProfileManager, parent=None) -> None:
        super().__init__(parent)
        self._profile = profile
        self._pm = profile_manager
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        name = self._profile.get("name", "?")
        pid = self._profile.get("id", "?")
        label = QLabel(f"InspectionTab — {name} (id={pid})\n[Fáza 8 — bude implementovaná]")
        label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(label)
