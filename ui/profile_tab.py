"""
ProfileTab — Tab 1: Konfigurácia referenčného profilu.

Stavy:
  IDLE         -- ziadny obrazok
  IMAGE_LOADED -- obrazok nacitany, bez ROI
  ROI_SELECTED -- obrazok + ROI (povolene vsetky ovladace)

Undo stack sleduje (frozenset[selected_indices], QRect|None) pred kazdou zmenou.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QComboBox,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.edge_detection import EdgeDetector
from core.profile_manager import ProfileManager
from core.segment_processor import Segment, SegmentProcessor
from ui.widgets.image_viewer import ImageViewer

# ---------------------------------------------------------------------------
# Stavy
# ---------------------------------------------------------------------------
_IDLE = "idle"
_IMAGE_LOADED = "image_loaded"
_ROI_SELECTED = "roi_selected"


# ---------------------------------------------------------------------------
# Pomocna trieda — synchronizovany par Slider + SpinBox
# ---------------------------------------------------------------------------

class _SliderSpinPair(QWidget):
    """Synchronizovany par QSlider + QDoubleSpinBox."""

    value_changed = pyqtSignal(float)

    def __init__(
        self,
        label_text: str,
        min_val: float,
        max_val: float,
        step: float = 1.0,
        decimals: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._step = step
        self._min = min_val
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(72)
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        n_steps = max(1, round((max_val - min_val) / step))
        self._slider.setRange(0, n_steps)
        layout.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(min_val, max_val)
        self._spin.setSingleStep(step)
        self._spin.setDecimals(decimals)
        self._spin.setFixedWidth(62)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._from_slider)
        self._spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, idx: int) -> None:
        if self._updating:
            return
        self._updating = True
        val = round(self._min + idx * self._step, 10)
        self._spin.setValue(val)
        self._updating = False
        self.value_changed.emit(val)

    def _from_spin(self, val: float) -> None:
        if self._updating:
            return
        self._updating = True
        idx = round((val - self._min) / self._step)
        self._slider.setValue(idx)
        self._updating = False
        self.value_changed.emit(val)

    def set_value(self, val: float) -> None:
        self._updating = True
        self._spin.setValue(val)
        self._slider.setValue(round((val - self._min) / self._step))
        self._updating = False

    def value(self) -> float:
        return self._spin.value()

    def setEnabled(self, enabled: bool) -> None:
        self._slider.setEnabled(enabled)
        self._spin.setEnabled(enabled)
        super().setEnabled(enabled)


def _section(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 6px; } "
                      "QGroupBox::title { subcontrol-origin: margin; left: 6px; }")
    return box


# ---------------------------------------------------------------------------
# ProfileTab
# ---------------------------------------------------------------------------

class ProfileTab(QWidget):
    """Zalozka pre konfiguraciu referencneho profilu."""

    profile_saved = pyqtSignal(int)  # emituje id ulozeneho profilu

    def __init__(
        self, profile: dict, profile_manager: ProfileManager, parent=None
    ) -> None:
        super().__init__(parent)
        self._profile = dict(profile)
        self._pm = profile_manager

        # Pracovny stav
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._roi: Optional[QRect] = None
        self._segments: list[Segment] = []
        self._selected_indices: set[int] = set()
        self._show_edges: bool = False
        self._scale: Optional[float] = None
        self._undo_stack: list[tuple[frozenset, Optional[QRect]]] = []
        self._state: str = _IDLE
        self._initializing: bool = False

        # Pomocnici
        self._detector = EdgeDetector()
        self._processor = SegmentProcessor()

        # Debounce casovac pre detekciu hran
        self._edge_timer = QTimer()
        self._edge_timer.setSingleShot(True)
        self._edge_timer.timeout.connect(self._run_edge_detection)

        self._build_ui()
        self._init_from_profile()

    # ------------------------------------------------------------------
    # Zostrojenie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Lavy panel (ovladace) ---
        ctrl_inner = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_inner)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(6)

        self._build_image_section(ctrl_layout)
        self._build_roi_section(ctrl_layout)
        self._build_edges_section(ctrl_layout)
        self._build_scale_section(ctrl_layout)
        self._build_segments_section(ctrl_layout)
        ctrl_layout.addStretch()
        self._build_save_section(ctrl_layout)

        scroll = QScrollArea()
        scroll.setWidget(ctrl_inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(286)
        splitter.addWidget(scroll)

        # --- Pravy panel (viewer + koordinaty) ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._viewer = ImageViewer()
        self._viewer.roi_selected.connect(self._on_roi_selected)
        self._viewer.pixel_clicked.connect(self._on_pixel_clicked)
        self._viewer.pixel_hovered.connect(self._on_pixel_hovered)
        self._viewer.scale_point_placed.connect(self._on_scale_point_placed)
        right_layout.addWidget(self._viewer, 1)

        self._coord_label = QLabel("x: --  y: --")
        self._coord_label.setStyleSheet(
            "color: #888; font-size: 11px; padding: 2px 6px;"
        )
        right_layout.addWidget(self._coord_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_image_section(self, parent: QVBoxLayout) -> None:
        box = _section("Obrazok")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_load = QPushButton("Nacitat obrazok...")
        self._btn_delete_img = QPushButton("Odstranit obrazok")
        self._btn_delete_img.setEnabled(False)

        self._btn_load.clicked.connect(self._on_load_image)
        self._btn_delete_img.clicked.connect(self._on_delete_image)

        layout.addWidget(self._btn_load)
        layout.addWidget(self._btn_delete_img)
        parent.addWidget(box)

    def _build_roi_section(self, parent: QVBoxLayout) -> None:
        box = _section("Oblast zaujmu (ROI)")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_draw_roi = QPushButton("Nakreslit ROI")
        self._btn_draw_roi.setEnabled(False)
        self._btn_clear_roi = QPushButton("Vymazat ROI")
        self._btn_clear_roi.setEnabled(False)

        self._btn_draw_roi.clicked.connect(self._on_draw_roi)
        self._btn_clear_roi.clicked.connect(self._on_clear_roi)

        layout.addWidget(self._btn_draw_roi)
        layout.addWidget(self._btn_clear_roi)
        parent.addWidget(box)

    def _build_edges_section(self, parent: QVBoxLayout) -> None:
        box = _section("Detekcia hran")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_toggle_edges = QPushButton("Zobrazit hrany")
        self._btn_toggle_edges.setCheckable(True)
        self._btn_toggle_edges.setEnabled(False)
        self._btn_toggle_edges.toggled.connect(self._on_toggle_edges)
        layout.addWidget(self._btn_toggle_edges)

        radio_row = QHBoxLayout()
        self._radio_canny = QRadioButton("Canny")
        self._radio_dexined = QRadioButton("DexiNed")
        self._radio_canny.setChecked(True)
        self._radio_canny.setEnabled(False)
        self._radio_dexined.setEnabled(False)
        self._radio_canny.toggled.connect(self._on_method_changed)
        self._radio_dexined.toggled.connect(self._on_method_changed)
        radio_row.addWidget(self._radio_canny)
        radio_row.addWidget(self._radio_dexined)
        layout.addLayout(radio_row)

        self._slider_t1 = _SliderSpinPair("Prah 1:", 0, 500, step=1, decimals=0)
        self._slider_t1.set_value(50)
        self._slider_t1.setEnabled(False)
        self._slider_t1.value_changed.connect(self._schedule_edge_refresh)
        layout.addWidget(self._slider_t1)

        self._slider_t2 = _SliderSpinPair("Prah 2:", 0, 500, step=1, decimals=0)
        self._slider_t2.set_value(150)
        self._slider_t2.setEnabled(False)
        self._slider_t2.value_changed.connect(self._schedule_edge_refresh)
        layout.addWidget(self._slider_t2)

        blur_row = QHBoxLayout()
        blur_lbl = QLabel("Blur:")
        blur_lbl.setFixedWidth(72)
        blur_row.addWidget(blur_lbl)
        self._combo_blur = QComboBox()
        self._combo_blur.addItem("Vypnuté (0)", 0)
        self._combo_blur.addItem("3 × 3", 3)
        self._combo_blur.addItem("5 × 5", 5)
        self._combo_blur.addItem("7 × 7", 7)
        self._combo_blur.setEnabled(False)
        self._combo_blur.currentIndexChanged.connect(self._schedule_edge_refresh)
        blur_row.addWidget(self._combo_blur, 1)
        layout.addLayout(blur_row)

        self._slider_conf = _SliderSpinPair("Spolah.:", 0.0, 1.0, step=0.01, decimals=2)
        self._slider_conf.set_value(0.5)
        self._slider_conf.setEnabled(False)
        self._slider_conf.value_changed.connect(self._schedule_edge_refresh)
        layout.addWidget(self._slider_conf)

        self._slider_min_len = _SliderSpinPair("Min. dlzka:", 1, 500, step=1, decimals=0)
        self._slider_min_len.set_value(20)
        self._slider_min_len.setEnabled(False)
        self._slider_min_len.value_changed.connect(self._schedule_edge_refresh)
        layout.addWidget(self._slider_min_len)

        parent.addWidget(box)

    def _build_scale_section(self, parent: QVBoxLayout) -> None:
        box = _section("Mierka")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_set_scale = QPushButton("Nastavit mierku (2 body)")
        self._btn_set_scale.setEnabled(False)
        self._btn_clear_scale = QPushButton("Vymazat mierku")
        self._btn_clear_scale.setEnabled(False)

        self._label_scale = QLabel("Nekalibrované")
        self._label_scale.setStyleSheet("color: #888; font-size: 11px;")

        self._btn_set_scale.clicked.connect(self._on_set_scale)
        self._btn_clear_scale.clicked.connect(self._on_clear_scale)

        layout.addWidget(self._btn_set_scale)
        layout.addWidget(self._btn_clear_scale)
        layout.addWidget(self._label_scale)
        parent.addWidget(box)

    def _build_segments_section(self, parent: QVBoxLayout) -> None:
        box = _section("Segmenty")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_undo = QPushButton("Spat (Undo)")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_undo)
        layout.addWidget(self._btn_undo)

        self._label_instr = QLabel("Nacitaj obrazok a nakresli ROI.")
        self._label_instr.setWordWrap(True)
        self._label_instr.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._label_instr)

        self._label_sel_count = QLabel("Vybranych: 0")
        self._label_sel_count.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._label_sel_count)

        parent.addWidget(box)

    def _build_save_section(self, parent: QVBoxLayout) -> None:
        self._btn_save = QPushButton("Ulozit profil")
        self._btn_save.setEnabled(False)
        self._btn_save.setStyleSheet(
            "background-color: #0078d7; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 3px;"
        )
        self._btn_save.clicked.connect(self._on_save)
        parent.addWidget(self._btn_save)

    # ------------------------------------------------------------------
    # Stavovy stroj
    # ------------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        self._state = state
        is_image = state in (_IMAGE_LOADED, _ROI_SELECTED)
        is_roi = state == _ROI_SELECTED

        self._btn_delete_img.setEnabled(is_image)
        self._btn_draw_roi.setEnabled(is_image)
        self._btn_clear_roi.setEnabled(is_roi)
        self._btn_toggle_edges.setEnabled(is_roi)
        self._btn_set_scale.setEnabled(is_roi)
        self._btn_save.setEnabled(is_roi)

        edge_controls_enabled = is_roi
        for w in (
            self._slider_t1, self._slider_t2, self._combo_blur,
            self._slider_conf, self._slider_min_len,
            self._radio_canny, self._radio_dexined,
        ):
            w.setEnabled(edge_controls_enabled)

        self._btn_clear_scale.setEnabled(is_roi and self._scale is not None)

        if not is_roi:
            self._label_instr.setText(
                "Nacitaj obrazok a nakresli ROI."
                if not is_image
                else "Nakresli oblast zaujmu (ROI)."
            )

    # ------------------------------------------------------------------
    # Obrazok
    # ------------------------------------------------------------------

    def _on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Nacitat obrazok",
            "",
            "Obrazky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Vsetky subory (*)",
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nepodarilo sa nacitat:\n{path}")
            return

        self._image = img
        self._image_path = path
        self._roi = None
        self._segments = []
        self._selected_indices.clear()
        self._undo_stack.clear()
        self._show_edges = False

        self._btn_toggle_edges.blockSignals(True)
        self._btn_toggle_edges.setChecked(False)
        self._btn_toggle_edges.setText("Zobrazit hrany")
        self._btn_toggle_edges.blockSignals(False)

        self._viewer.set_image(img)
        self._viewer.set_roi(None)
        self._viewer.set_overlay(None)
        self._viewer.set_mode("view")
        self._update_sel_count()
        self._set_state(_IMAGE_LOADED)

    def _on_delete_image(self) -> None:
        self._image = None
        self._image_path = None
        self._roi = None
        self._segments = []
        self._selected_indices.clear()
        self._undo_stack.clear()
        self._show_edges = False

        self._btn_toggle_edges.blockSignals(True)
        self._btn_toggle_edges.setChecked(False)
        self._btn_toggle_edges.setText("Zobrazit hrany")
        self._btn_toggle_edges.blockSignals(False)

        self._viewer.set_image(None)
        self._viewer.set_roi(None)
        self._viewer.set_overlay(None)
        self._viewer.set_mode("view")
        self._update_sel_count()
        self._set_state(_IDLE)

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _on_draw_roi(self) -> None:
        self._viewer.set_mode("roi")
        self._label_instr.setText("Nakresli oblast tahom mysi.")

    def _on_clear_roi(self) -> None:
        self._roi = None
        self._segments = []
        self._selected_indices.clear()
        self._undo_stack.clear()
        self._show_edges = False

        self._btn_toggle_edges.blockSignals(True)
        self._btn_toggle_edges.setChecked(False)
        self._btn_toggle_edges.setText("Zobrazit hrany")
        self._btn_toggle_edges.blockSignals(False)

        self._viewer.set_roi(None)
        self._viewer.set_overlay(None)
        self._viewer.set_mode("view")
        self._update_sel_count()
        self._set_state(_IMAGE_LOADED)

    def _on_roi_selected(self, rect: QRect) -> None:
        self._roi = rect
        self._segments = []
        self._selected_indices.clear()
        self._undo_stack.clear()

        self._viewer.set_roi(rect)
        self._set_state(_ROI_SELECTED)

        if self._show_edges:
            self._run_edge_detection()
            self._viewer.set_mode("click_segment")
        else:
            self._viewer.set_mode("view")

        self._label_instr.setText("Klikni na hranu pre vyber segmentu.")

    # ------------------------------------------------------------------
    # Hrany
    # ------------------------------------------------------------------

    def _on_toggle_edges(self, checked: bool) -> None:
        if self._initializing:
            return
        self._show_edges = checked
        self._btn_toggle_edges.setText("Skryt hrany" if checked else "Zobrazit hrany")

        if checked:
            self._run_edge_detection()
            self._viewer.set_mode("click_segment")
            self._label_instr.setText("Klikni na hranu pre vyber segmentu.")
        else:
            self._viewer.set_overlay(None)
            self._viewer.set_mode("view")
            self._label_instr.setText("Zobraz hrany pre vyber segmentov.")

    def _on_method_changed(self, checked: bool) -> None:
        if self._initializing or not checked:
            return
        self._schedule_edge_refresh()

    def _schedule_edge_refresh(self, *_args) -> None:
        if self._initializing:
            return
        if self._show_edges and self._roi is not None:
            self._edge_timer.start(350)

    def _ensure_dexined_weights(self) -> bool:
        """Vrati True ak vaha existuju alebo boli uspesne stiahnuté. Inak False."""
        from core.edge_detection import EdgeDetector
        if EdgeDetector.WEIGHTS_PATH.exists():
            return True
        from PyQt6.QtWidgets import QDialog
        from ui.widgets.download_progress_dialog import DownloadProgressDialog
        dlg = DownloadProgressDialog(
            EdgeDetector.WEIGHTS_URL, EdgeDetector.WEIGHTS_PATH, parent=self
        )
        return dlg.exec() == QDialog.DialogCode.Accepted

    def _run_edge_detection(self) -> None:
        if self._image is None or self._roi is None:
            return

        # DexiNed: stiahni vahy ak chybaju
        if self._radio_dexined.isChecked():
            if not self._ensure_dexined_weights():
                # Stiahnutie zlyhalo alebo bolo zrusene — prepni na Canny
                self._radio_canny.blockSignals(True)
                self._radio_canny.setChecked(True)
                self._radio_canny.blockSignals(False)
                return

        try:
            if self._radio_canny.isChecked():
                edge_map = self._detector.run_canny(
                    self._image,
                    threshold1=int(self._slider_t1.value()),
                    threshold2=int(self._slider_t2.value()),
                    blur_kernel=self._combo_blur.currentData(),
                )
            else:
                edge_map = self._detector.run_dexined(
                    self._image,
                    confidence=self._slider_conf.value(),
                )
        except Exception as exc:
            QMessageBox.warning(self, "Detekcia hran", f"Chyba: {exc}")
            return

        roi_t = (
            self._roi.x(), self._roi.y(),
            self._roi.width(), self._roi.height(),
        )
        new_segs = self._processor.extract_segments(
            edge_map,
            min_length=int(self._slider_min_len.value()),
            roi=roi_t,
        )
        valid = {s.index for s in new_segs}
        self._segments = new_segs
        self._selected_indices &= valid
        self._render_overlay()
        self._update_sel_count()

    # ------------------------------------------------------------------
    # Segmenty — klik + undo
    # ------------------------------------------------------------------

    def _on_pixel_clicked(self, point: QPoint) -> None:
        if not self._segments:
            return
        idx = self._processor.hit_test_segment(
            self._segments, (point.x(), point.y()), tolerance_px=5
        )
        if idx is not None:
            self._push_undo()
            if idx in self._selected_indices:
                self._selected_indices.discard(idx)
            else:
                self._selected_indices.add(idx)
            self._render_overlay()
            self._update_sel_count()

    def _on_pixel_hovered(self, point: QPoint) -> None:
        self._coord_label.setText(f"x: {point.x()}  y: {point.y()}")

    def _push_undo(self) -> None:
        self._undo_stack.append((
            frozenset(self._selected_indices),
            QRect(self._roi) if self._roi else None,
        ))
        self._btn_undo.setEnabled(True)

    def _on_undo(self) -> None:
        if not self._undo_stack:
            return
        indices, roi = self._undo_stack.pop()
        self._selected_indices = set(indices)
        if roi is not None:
            self._roi = QRect(roi)
            self._viewer.set_roi(roi)
        self._render_overlay()
        self._update_sel_count()
        self._btn_undo.setEnabled(bool(self._undo_stack))

    def _update_sel_count(self) -> None:
        self._label_sel_count.setText(f"Vybranych: {len(self._selected_indices)}")

    def _render_overlay(self) -> None:
        if self._image is None:
            self._viewer.set_overlay(None)
            return
        h, w = self._image.shape[:2]
        overlay = self._processor.render_segment_map(
            (h, w), self._segments, list(self._selected_indices)
        )
        self._viewer.set_overlay(overlay)

    # ------------------------------------------------------------------
    # Mierka
    # ------------------------------------------------------------------

    def _on_set_scale(self) -> None:
        self._viewer.clear_scale_points()
        self._viewer.set_mode("scale_tool")
        self._label_instr.setText("Klikni bod 1, potom bod 2 pre kalibraciu mierky.")

    def _on_scale_point_placed(self, idx: int, point: QPoint) -> None:
        if idx == 1:
            pts = self._viewer.get_scale_points()
            p0, p1 = pts[0], pts[1]
            if p0 is None or p1 is None:
                return

            dist_px = math.sqrt((p1.x() - p0.x()) ** 2 + (p1.y() - p0.y()) ** 2)
            if dist_px < 1.0:
                QMessageBox.warning(self, "Mierka", "Body su prilis blizko seba.")
                self._viewer.clear_scale_points()
                self._viewer.set_mode("view")
                return

            dist_mm, ok = QInputDialog.getDouble(
                self,
                "Mierka",
                f"Vzdialenost medzi bodmi ({dist_px:.1f} px) v milimetroch:",
                10.0, 0.001, 100000.0, 3,
            )
            if ok and dist_mm > 0:
                self._scale = dist_px / dist_mm
                self._label_scale.setText(f"{self._scale:.4f} px/mm")
                self._btn_clear_scale.setEnabled(True)

            self._viewer.clear_scale_points()
            if self._show_edges:
                self._viewer.set_mode("click_segment")
                self._label_instr.setText("Klikni na hranu pre vyber segmentu.")
            else:
                self._viewer.set_mode("view")
                self._label_instr.setText("Mierka nastavena.")

    def _on_clear_scale(self) -> None:
        self._scale = None
        self._label_scale.setText("Nekalibrované")
        self._btn_clear_scale.setEnabled(False)

    # ------------------------------------------------------------------
    # Ulozenie profilu
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "Ulozit", "Ziadny obrazok.")
            return
        if self._roi is None:
            QMessageBox.warning(self, "Ulozit", "ROI nie je nastavene.")
            return
        if not self._selected_indices:
            reply = QMessageBox.question(
                self,
                "Ulozit profil",
                "Nie su vybrate ziadne segmenty. Pokracovat?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        pid = self._profile["id"]
        profile_dir = Path(self._pm.base_dir) / str(pid)
        profile_dir.mkdir(parents=True, exist_ok=True)

        ref_path = str(profile_dir / "reference.png")
        seg_map_path = str(profile_dir / "segment_map.png")

        # Uloz referencny obrazok
        if self._image_path != ref_path:
            cv2.imwrite(ref_path, self._image)

        # Vypocitaj centroid
        if self._selected_indices and self._segments:
            centroid = self._processor.compute_combined_centroid(
                self._segments, list(self._selected_indices)
            )
        else:
            centroid = (
                self._roi.x() + self._roi.width() / 2.0,
                self._roi.y() + self._roi.height() / 2.0,
            )

        # Uloz segment mapu
        h, w = self._image.shape[:2]
        if self._segments:
            seg_map = self._processor.render_segment_map(
                (h, w), self._segments, list(self._selected_indices)
            )
        else:
            seg_map = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(seg_map_path, seg_map)

        # Aktualizuj profil dict
        edge_method = "dexined" if self._radio_dexined.isChecked() else "canny"
        self._profile.update({
            "roi": {
                "x": self._roi.x(), "y": self._roi.y(),
                "w": self._roi.width(), "h": self._roi.height(),
            },
            "edge_method": edge_method,
            "canny_params": {
                "threshold1": int(self._slider_t1.value()),
                "threshold2": int(self._slider_t2.value()),
                "blur_kernel": self._combo_blur.currentData(),
            },
            "dexined_params": {"confidence": self._slider_conf.value()},
            "min_segment_length": int(self._slider_min_len.value()),
            "scale_px_per_mm": self._scale,
            "centroid_ref": {"x": centroid[0], "y": centroid[1]},
            "segment_indices": sorted(self._selected_indices),
            "ecc_params": self._profile.get("ecc_params", {
                "motion_type": "MOTION_EUCLIDEAN",
                "max_iter": 200,
                "epsilon": 1e-5,
            }),
            "roi_inspection_offset": self._profile.get(
                "roi_inspection_offset", {"dx": 0, "dy": 0}
            ),
            "paths": {
                "reference_image": ref_path,
                "segment_map": seg_map_path,
            },
        })

        try:
            self._pm.save_profile(self._profile)
        except Exception as exc:
            QMessageBox.critical(self, "Chyba ulozenia", str(exc))
            return

        self._image_path = ref_path

        # Zobraz centroid
        self._viewer.set_crosshairs([(centroid[0], centroid[1])])

        QMessageBox.information(
            self, "Ulozene",
            f"Profil '{self._profile.get('name', '')}' bol uspesne ulozeny."
        )
        self.profile_saved.emit(pid)

    # ------------------------------------------------------------------
    # Inicializacia z existujuceho profilu
    # ------------------------------------------------------------------

    def _init_from_profile(self) -> None:
        self._initializing = True
        profile = self._profile

        # Nacitaj parametre hran
        canny = profile.get("canny_params", {})
        self._slider_t1.set_value(canny.get("threshold1", 50))
        self._slider_t2.set_value(canny.get("threshold2", 150))
        blur_k = canny.get("blur_kernel", 0)
        idx = self._combo_blur.findData(blur_k)
        self._combo_blur.setCurrentIndex(idx if idx >= 0 else 0)
        self._slider_conf.set_value(
            profile.get("dexined_params", {}).get("confidence", 0.5)
        )
        self._slider_min_len.set_value(profile.get("min_segment_length", 20))

        if profile.get("edge_method") == "dexined":
            self._radio_dexined.setChecked(True)
        else:
            self._radio_canny.setChecked(True)

        # Mierka
        scale = profile.get("scale_px_per_mm")
        if scale:
            self._scale = float(scale)
            self._label_scale.setText(f"{self._scale:.4f} px/mm")

        self._initializing = False

        # Referencny obrazok
        ref_path = profile.get("paths", {}).get("reference_image", "")
        if not ref_path or not Path(ref_path).exists():
            return

        img = cv2.imread(ref_path)
        if img is None:
            return

        self._image = img
        self._image_path = ref_path
        self._viewer.set_image(img)
        self._set_state(_IMAGE_LOADED)

        # ROI
        roi_data = profile.get("roi")
        if not roi_data:
            return
        rw, rh = roi_data.get("w", 0), roi_data.get("h", 0)
        if rw <= 0 or rh <= 0:
            return

        self._roi = QRect(roi_data["x"], roi_data["y"], rw, rh)
        self._viewer.set_roi(self._roi)
        self._set_state(_ROI_SELECTED)

        # Detekcia hran a obnova segmentov
        try:
            t1 = int(self._slider_t1.value())
            t2 = int(self._slider_t2.value())
            conf = self._slider_conf.value()
            min_len = int(self._slider_min_len.value())

            if profile.get("edge_method") == "dexined":
                edge_map = self._detector.run_dexined(img, confidence=conf)
            else:
                edge_map = self._detector.run_canny(img, threshold1=t1, threshold2=t2)

            roi_t = (self._roi.x(), self._roi.y(), rw, rh)
            self._segments = self._processor.extract_segments(
                edge_map, min_length=min_len, roi=roi_t
            )

            saved = set(profile.get("segment_indices", []))
            valid = {s.index for s in self._segments}
            self._selected_indices = saved & valid

            self._show_edges = True
            self._btn_toggle_edges.blockSignals(True)
            self._btn_toggle_edges.setChecked(True)
            self._btn_toggle_edges.setText("Skryt hrany")
            self._btn_toggle_edges.blockSignals(False)

            self._render_overlay()
            self._viewer.set_mode("click_segment")
            self._update_sel_count()

        except Exception:
            # DexiNed bez vahy alebo ina chyba — hrany sa nezobrazia
            pass

        # Obnov centroid krizik
        c = profile.get("centroid_ref")
        if c:
            self._viewer.set_crosshairs([(c.get("x", 0.0), c.get("y", 0.0))])

        self._label_instr.setText("Klikni na hranu pre vyber segmentu.")
        self._btn_clear_scale.setEnabled(self._scale is not None)
