"""
InspectionTab -- Tab 2: Inspekcia.

Rozlozenie:
  Horizontalny QSplitter: levy viewer (referencny) | pravy viewer (inspekcia)
  Koordinatovy label pod kazdym viewerom.
  Pravy panel (~260 px): nacitanie inspekcie, ECC parametre, ROI offset,
  zobrazenie segmentov, tlacidlo Run, vysledky.

ECC bezi v QThread (InspectionWorker) — GUI zostane responzivne.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QPoint, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.edge_detection import EdgeDetector
from core.inspection_engine import AlignmentStrategy, InspectionEngine, InspectionResult
from core.profile_manager import ProfileManager
from core.segment_processor import SegmentProcessor
from ui.widgets.image_viewer import ImageViewer


# ---------------------------------------------------------------------------
# QThread worker pre ECC
# ---------------------------------------------------------------------------

class InspectionWorker(QObject):
    """Spusta InspectionEngine.run() v separatnom vlakne."""

    result_ready = pyqtSignal(object)   # InspectionResult
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: InspectionEngine,
        ref_img: np.ndarray,
        seg_map: np.ndarray,
        insp_img: np.ndarray,
        centroid_ref: tuple,
        roi: tuple,
        roi_offset: tuple,
        ecc_params: dict,
        px_per_mm: Optional[float],
        roi_search_expansion: int = 0,
        edge_method: str = "canny",
        edge_params: Optional[dict] = None,
        alignment_strategy: AlignmentStrategy = AlignmentStrategy.ECC_ONLY,
        template_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._ref = ref_img
        self._seg = seg_map
        self._insp = insp_img
        self._centroid = centroid_ref
        self._roi = roi
        self._roi_offset = roi_offset
        self._roi_search_expansion = roi_search_expansion
        self._ecc_params = ecc_params
        self._px_per_mm = px_per_mm
        self._edge_method = edge_method
        self._edge_params = edge_params or {}
        self._alignment_strategy = alignment_strategy
        self._template_params = template_params or {}

    def run(self) -> None:
        try:
            result = self._engine.run(
                self._ref,
                self._seg,
                self._insp,
                self._centroid,
                self._roi,
                roi_offset=self._roi_offset,
                roi_search_expansion=self._roi_search_expansion,
                ecc_params=self._ecc_params,
                px_per_mm=self._px_per_mm,
                edge_method=self._edge_method,
                edge_params=self._edge_params,
                alignment_strategy=self._alignment_strategy,
                template_params=self._template_params,
            )
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# InspectionTab
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, str] = {
    "ECC only":                  "ecc_only",
    "Template matching only":    "template_only",
    "Template matching + ECC":   "template_then_ecc",
}
_STRATEGY_REVERSE_MAP: dict[str, str] = {v: k for k, v in _STRATEGY_MAP.items()}

_RELIABILITY_COLORS = {
    "HIGH": "#00c853",
    "MEDIUM": "#ffd600",
    "LOW": "#d50000",
}


class InspectionTab(QWidget):
    """Zalozka pre inspekciu — porovnanie referencneho a inspekcionneho obrazka."""

    def __init__(
        self, profile: dict, profile_manager: ProfileManager, parent=None
    ) -> None:
        super().__init__(parent)
        self._profile = dict(profile)
        self._pm = profile_manager

        self._ref_image: Optional[np.ndarray] = None
        self._seg_map: Optional[np.ndarray] = None
        self._insp_image: Optional[np.ndarray] = None
        self._insp_path: Optional[str] = None

        self._processor = SegmentProcessor()
        self._engine = InspectionEngine()

        self._thread: Optional[QThread] = None
        self._worker: Optional[InspectionWorker] = None

        self._build_ui()
        self._load_profile_assets()

    # ------------------------------------------------------------------
    # Zostrojenie UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Horizontalny splitter: [ref viewer | insp viewer] | pravy panel
        outer_splitter = QSplitter()
        root.addWidget(outer_splitter)

        # Viewer splitter
        viewer_splitter = QSplitter()
        outer_splitter.addWidget(viewer_splitter)

        # Levy viewer (referencny)
        left_wrap = QWidget()
        left_layout = QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self._viewer_ref = ImageViewer()
        self._viewer_ref.pixel_hovered.connect(
            lambda p: self._coord_ref.setText(f"x: {p.x()}  y: {p.y()}")
        )
        left_layout.addWidget(self._viewer_ref, 1)
        self._coord_ref = QLabel("x: --  y: --")
        self._coord_ref.setStyleSheet("color: #888; font-size: 11px; padding: 2px 6px;")
        left_layout.addWidget(self._coord_ref)
        viewer_splitter.addWidget(left_wrap)

        # Pravy viewer (inspekcia)
        right_wrap = QWidget()
        right_layout = QVBoxLayout(right_wrap)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self._viewer_insp = ImageViewer()
        self._viewer_insp.pixel_hovered.connect(
            lambda p: self._coord_insp.setText(f"x: {p.x()}  y: {p.y()}")
        )
        right_layout.addWidget(self._viewer_insp, 1)
        self._coord_insp = QLabel("x: --  y: --")
        self._coord_insp.setStyleSheet("color: #888; font-size: 11px; padding: 2px 6px;")
        right_layout.addWidget(self._coord_insp)
        viewer_splitter.addWidget(right_wrap)

        # Pravy ovladaci panel
        ctrl_inner = QWidget()
        ctrl_inner.setFixedWidth(265)
        ctrl_layout = QVBoxLayout(ctrl_inner)
        ctrl_layout.setContentsMargins(6, 6, 6, 6)
        ctrl_layout.setSpacing(8)

        self._build_insp_image_section(ctrl_layout)
        self._build_ecc_section(ctrl_layout)
        self._build_overlay_section(ctrl_layout)
        self._build_run_section(ctrl_layout)
        self._build_results_section(ctrl_layout)
        ctrl_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(ctrl_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(275)
        outer_splitter.addWidget(scroll)

        outer_splitter.setStretchFactor(0, 1)
        outer_splitter.setStretchFactor(1, 0)

    def _build_insp_image_section(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Inspekcia")
        box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 6px; } "
                          "QGroupBox::title { subcontrol-origin: margin; left: 6px; }")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._btn_load_insp = QPushButton("Nacitat inspekciu...")
        self._btn_delete_insp = QPushButton("Odstranit inspekciu")
        self._btn_delete_insp.setEnabled(False)

        self._btn_load_insp.clicked.connect(self._on_load_insp)
        self._btn_delete_insp.clicked.connect(self._on_delete_insp)

        layout.addWidget(self._btn_load_insp)
        layout.addWidget(self._btn_delete_insp)
        parent.addWidget(box)

    def _build_ecc_section(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Zarovnanie")
        box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 6px; } "
                          "QGroupBox::title { subcontrol-origin: margin; left: 6px; }")
        layout = QFormLayout(box)
        layout.setSpacing(4)

        # --- Voľba stratégie ---
        self._combo_strategy = QComboBox()
        self._combo_strategy.addItems(list(_STRATEGY_MAP.keys()))
        self._combo_strategy.setCurrentText("ECC only")
        self._combo_strategy.currentIndexChanged.connect(self._on_strategy_changed)
        layout.addRow("Strategia:", self._combo_strategy)

        # --- ECC parametre (skrytelný blok) ---
        self._ecc_params_widget = QWidget()
        ecc_form = QFormLayout(self._ecc_params_widget)
        ecc_form.setContentsMargins(0, 0, 0, 0)
        ecc_form.setSpacing(4)

        self._combo_motion = QComboBox()
        self._combo_motion.addItems([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
        ])
        self._combo_motion.setCurrentText("MOTION_EUCLIDEAN")
        ecc_form.addRow("Typ pohybu:", self._combo_motion)

        self._spin_max_iter = QSpinBox()
        self._spin_max_iter.setRange(10, 5000)
        self._spin_max_iter.setValue(200)
        ecc_form.addRow("Max iteracii:", self._spin_max_iter)

        self._edit_epsilon = QLineEdit("1e-5")
        ecc_form.addRow("Epsilon:", self._edit_epsilon)

        layout.addRow(self._ecc_params_widget)

        # --- Template matching parametre (skrytelný blok) ---
        self._tm_params_widget = QWidget()
        tm_form = QFormLayout(self._tm_params_widget)
        tm_form.setContentsMargins(0, 0, 0, 0)
        tm_form.setSpacing(4)

        self._spin_tm_expansion = QDoubleSpinBox()
        self._spin_tm_expansion.setRange(0.1, 5.0)
        self._spin_tm_expansion.setSingleStep(0.1)
        self._spin_tm_expansion.setValue(0.5)
        self._spin_tm_expansion.setToolTip(
            "Rozšírenie oblasti hľadania.\n"
            "0.5 = hľadá sa v okruhu ±50 % rozmeru ROI od stredu."
        )
        tm_form.addRow("Search expansion:", self._spin_tm_expansion)

        self._combo_tm_method = QComboBox()
        self._combo_tm_method.addItems(["TM_CCOEFF_NORMED", "TM_CCORR_NORMED"])
        tm_form.addRow("TM metoda:", self._combo_tm_method)

        layout.addRow(self._tm_params_widget)

        # --- ROI offset ---
        offset_row = QHBoxLayout()
        self._spin_dx = QSpinBox()
        self._spin_dx.setRange(-4096, 4096)
        self._spin_dx.setValue(0)
        self._spin_dy = QSpinBox()
        self._spin_dy.setRange(-4096, 4096)
        self._spin_dy.setValue(0)
        offset_row.addWidget(QLabel("dX:"))
        offset_row.addWidget(self._spin_dx)
        offset_row.addWidget(QLabel("dY:"))
        offset_row.addWidget(self._spin_dy)
        layout.addRow("ROI offset:", offset_row)

        self._spin_expansion = QSpinBox()
        self._spin_expansion.setRange(0, 512)
        self._spin_expansion.setValue(0)
        self._spin_expansion.setToolTip(
            "Rozšírenie ROI pre vyhľadávanie.\n"
            "Zväčší oblasť výrezu na všetkých stranách o zadaný počet pixelov.\n"
            "Nemá vplyv na výpočet polohy."
        )
        self._spin_expansion.valueChanged.connect(self._on_expansion_changed)
        layout.addRow("ROI expanzia:", self._spin_expansion)

        # --- Obnova hodnôt z profilu ---
        ecc = self._profile.get("ecc_params", {})
        if ecc.get("motion_type") in ("MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE"):
            self._combo_motion.setCurrentText(ecc["motion_type"])
        self._spin_max_iter.setValue(int(ecc.get("max_iter", 200)))
        self._edit_epsilon.setText(str(ecc.get("epsilon", "1e-5")))

        offset = self._profile.get("roi_inspection_offset", {})
        self._spin_dx.setValue(int(offset.get("dx", 0)))
        self._spin_dy.setValue(int(offset.get("dy", 0)))
        self._spin_expansion.setValue(int(self._profile.get("roi_search_expansion", 0)))

        strategy_str = self._profile.get("alignment_strategy", "ecc_only")
        self._combo_strategy.setCurrentText(
            _STRATEGY_REVERSE_MAP.get(strategy_str, "ECC only")
        )

        tm_p = self._profile.get("template_params", {})
        self._spin_tm_expansion.setValue(float(tm_p.get("search_expansion", 0.5)))
        tm_method = tm_p.get("method", "TM_CCOEFF_NORMED")
        if tm_method in ("TM_CCOEFF_NORMED", "TM_CCORR_NORMED"):
            self._combo_tm_method.setCurrentText(tm_method)

        # Nastaví počiatočnú viditeľnosť blokoch
        self._on_strategy_changed(self._combo_strategy.currentIndex())

        parent.addWidget(box)

    def _on_strategy_changed(self, _index: int) -> None:
        """Zobrazí / skryje ECC a TM bloky podľa zvolenej stratégie."""
        strategy = self._combo_strategy.currentText()
        show_ecc = strategy in ("ECC only", "Template matching + ECC")
        show_tm  = strategy in ("Template matching only", "Template matching + ECC")
        self._ecc_params_widget.setVisible(show_ecc)
        self._tm_params_widget.setVisible(show_tm)

    def _build_overlay_section(self, parent: QVBoxLayout) -> None:
        self._chk_show_segments = QCheckBox("Zobrazit segmenty na vieweroch")
        self._chk_show_segments.setChecked(True)
        self._chk_show_segments.toggled.connect(self._on_toggle_segments)
        parent.addWidget(self._chk_show_segments)

    def _build_run_section(self, parent: QVBoxLayout) -> None:
        self._btn_run = QPushButton("Spustit inspekciu")
        self._btn_run.setEnabled(False)
        self._btn_run.setStyleSheet(
            "background-color: #0078d7; color: white; "
            "font-weight: bold; padding: 8px; border-radius: 3px;"
        )
        self._btn_run.clicked.connect(self._on_run)
        parent.addWidget(self._btn_run)

    def _build_results_section(self, parent: QVBoxLayout) -> None:
        box = QGroupBox("Vysledky")
        box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 6px; } "
                          "QGroupBox::title { subcontrol-origin: margin; left: 6px; }")
        form = QFormLayout(box)
        form.setSpacing(4)

        def _lbl() -> QLabel:
            l = QLabel("--")
            l.setStyleSheet("font-family: monospace;")
            return l

        self._lbl_shift_px   = _lbl()
        self._lbl_shift_mm   = _lbl()
        self._lbl_rot        = _lbl()
        self._lbl_ncc        = _lbl()
        self._lbl_reliability = _lbl()
        self._lbl_duration   = _lbl()
        self._lbl_centroid_ref  = _lbl()
        self._lbl_centroid_insp = _lbl()

        form.addRow("Posun (px):", self._lbl_shift_px)
        form.addRow("Posun (mm):", self._lbl_shift_mm)
        form.addRow("Rotacia:", self._lbl_rot)
        form.addRow("NCC skore:", self._lbl_ncc)
        form.addRow("Spolahliv.:", self._lbl_reliability)
        form.addRow("Trvanie:", self._lbl_duration)
        form.addRow("Centroid ref:", self._lbl_centroid_ref)
        form.addRow("Centroid insp:", self._lbl_centroid_insp)

        parent.addWidget(box)

    # ------------------------------------------------------------------
    # Nacitanie profilovych assetov
    # ------------------------------------------------------------------

    def _load_profile_assets(self) -> None:
        paths = self._profile.get("paths", {})

        ref_path = paths.get("reference_image", "")
        if ref_path and Path(ref_path).exists():
            img = cv2.imread(ref_path)
            if img is not None:
                self._ref_image = img
                self._viewer_ref.set_image(img)

        seg_path = paths.get("segment_map", "")
        if seg_path and Path(seg_path).exists():
            seg = cv2.imread(seg_path)
            if seg is not None:
                self._seg_map = seg
                if self._chk_show_segments.isChecked():
                    self._viewer_ref.set_overlay(seg)

        # ROI na ref vieweri
        roi = self._profile.get("roi")
        if roi:
            from PyQt6.QtCore import QRect
            self._viewer_ref.set_roi(QRect(roi["x"], roi["y"], roi["w"], roi["h"]))

        # Centroid ref krizik
        c = self._profile.get("centroid_ref")
        if c:
            self._viewer_ref.set_crosshairs([(c.get("x", 0.0), c.get("y", 0.0))])

        # Zobraziť rozšírenú vyhľadávaciu ROI na inspekčnom vieweri
        self._on_expansion_changed(self._spin_expansion.value())

        self._update_run_button()

    # ------------------------------------------------------------------
    # Inspekcia — nacitanie obrazka
    # ------------------------------------------------------------------

    def _on_load_insp(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Nacitat inspekciu", "",
            "Obrazky (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;Vsetky subory (*)",
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Chyba", f"Nepodarilo sa nacitat:\n{path}")
            return

        # Overenie rozmerov
        if self._ref_image is not None:
            if img.shape[:2] != self._ref_image.shape[:2]:
                QMessageBox.warning(
                    self,
                    "Rozmerova nezhoda",
                    f"Inspekcia {img.shape[:2]} != referencny {self._ref_image.shape[:2]}.\n"
                    "Obrazky musia mat rovnake rozmery.",
                )
                return

        self._insp_image = img
        self._insp_path = path
        self._viewer_insp.set_image(img)
        self._btn_delete_insp.setEnabled(True)
        self._update_run_button()

    def _on_delete_insp(self) -> None:
        self._insp_image = None
        self._insp_path = None
        self._viewer_insp.set_image(None)
        self._viewer_insp.set_overlay(None)
        self._viewer_insp.set_crosshairs([])
        self._btn_delete_insp.setEnabled(False)
        self._update_run_button()
        self._clear_results()

    def _update_run_button(self) -> None:
        can_run = (
            self._ref_image is not None
            and self._seg_map is not None
            and self._insp_image is not None
        )
        self._btn_run.setEnabled(can_run)

    # ------------------------------------------------------------------
    # Overlay segmentov
    # ------------------------------------------------------------------

    def _on_toggle_segments(self, checked: bool) -> None:
        if checked and self._seg_map is not None:
            self._viewer_ref.set_overlay(self._seg_map)
        else:
            self._viewer_ref.set_overlay(None)
        if not checked:
            self._viewer_insp.set_overlay(None)

    def _on_expansion_changed(self, value: int) -> None:
        """Aktualizuje zobrazenie vyhľadávacej ROI na inspekčnom vieweri a uloží do profilu."""
        roi_data = self._profile.get("roi", {})
        rx = int(roi_data.get("x", 0))
        ry = int(roi_data.get("y", 0))
        rw = int(roi_data.get("w", 0))
        rh = int(roi_data.get("h", 0))
        e = max(0, value)
        from PyQt6.QtCore import QRect
        self._viewer_insp.set_roi(QRect(rx - e, ry - e, rw + 2 * e, rh + 2 * e))
        self._profile["roi_search_expansion"] = e
        self._pm.save_profile(self._profile)

    # ------------------------------------------------------------------
    # Spustenie inspekcie (QThread)
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        if self._ref_image is None or self._seg_map is None or self._insp_image is None:
            return

        # Zozbieraj ECC parametre
        try:
            epsilon = float(self._edit_epsilon.text())
        except ValueError:
            epsilon = 1e-5

        ecc_params = {
            "motion_type": self._combo_motion.currentText(),
            "max_iter": self._spin_max_iter.value(),
            "epsilon": epsilon,
        }

        roi_data = self._profile.get("roi", {})
        roi = (
            roi_data.get("x", 0), roi_data.get("y", 0),
            roi_data.get("w", self._ref_image.shape[1]),
            roi_data.get("h", self._ref_image.shape[0]),
        )

        roi_offset = (self._spin_dx.value(), self._spin_dy.value())

        c = self._profile.get("centroid_ref", {})
        centroid_ref = (c.get("x", 0.0), c.get("y", 0.0))

        px_per_mm = self._profile.get("scale_px_per_mm")

        edge_method = self._profile.get("edge_method", "canny")
        if edge_method == "dexined":
            edge_params = self._profile.get("dexined_params", {})
        else:
            edge_params = self._profile.get("canny_params", {})

        # Stratégia zarovnania
        strategy_str = _STRATEGY_MAP.get(self._combo_strategy.currentText(), "ecc_only")
        alignment_strategy = AlignmentStrategy(strategy_str)

        template_params = {
            "search_expansion": self._spin_tm_expansion.value(),
            "method": self._combo_tm_method.currentText(),
        }

        # Zablokuj tlacidlo pocas behu
        self._btn_run.setEnabled(False)
        self._btn_run.setText("Prebieha...")

        # Vytvor worker + thread
        self._thread = QThread()
        self._worker = InspectionWorker(
            engine=self._engine,
            ref_img=self._ref_image,
            seg_map=self._seg_map,
            insp_img=self._insp_image,
            centroid_ref=centroid_ref,
            roi=roi,
            roi_offset=roi_offset,
            ecc_params=ecc_params,
            px_per_mm=px_per_mm,
            roi_search_expansion=self._spin_expansion.value(),
            edge_method=edge_method,
            edge_params=edge_params,
            alignment_strategy=alignment_strategy,
            template_params=template_params,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.result_ready.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_done)
        self._thread.start()

    def _on_result(self, result: InspectionResult) -> None:
        self._populate_results(result)

        # Kriziky na vieweroch
        self._viewer_ref.set_crosshairs([result.centroid_ref_px])
        self._viewer_insp.set_crosshairs([result.centroid_insp_px])

        # Overlay na inspekcionom vieweri — zarovnané segmenty
        if self._chk_show_segments.isChecked():
            overlay = self._build_aligned_overlay(result)
            self._viewer_insp.set_overlay(overlay)

    def _build_aligned_overlay(self, result) -> Optional[np.ndarray]:
        """
        Spustí detekciu hrán na inšpekčnom obrázku, extrahuje segmenty a vráti
        BGR overlay s iba vybranými segmentmi (definovanými v profile).
        Fallback na statickú segment_map pri chybe.
        """
        if self._insp_image is None:
            return None

        try:
            edge_method = self._profile.get("edge_method", "canny")
            if edge_method == "dexined":
                edge_params = self._profile.get("dexined_params", {})
            else:
                edge_params = self._profile.get("canny_params", {})

            min_seg_len = float(self._profile.get("min_segment_length", 20))
            segment_indices = set(self._profile.get("segment_indices", []))

            roi_data = self._profile.get("roi", {})
            rx = int(roi_data.get("x", 0))
            ry = int(roi_data.get("y", 0))
            rw = int(roi_data.get("w", self._insp_image.shape[1]))
            rh = int(roi_data.get("h", self._insp_image.shape[0]))
            ox, oy = self._spin_dx.value(), self._spin_dy.value()

            # Grayscale konverzia
            gray = (
                cv2.cvtColor(self._insp_image, cv2.COLOR_BGR2GRAY)
                if self._insp_image.ndim == 3
                else self._insp_image
            )

            # Detekcia hrán na inšpekčnom obrázku
            detector = EdgeDetector()
            if edge_method == "dexined":
                edge_map = detector.run_dexined(
                    gray, confidence=edge_params.get("confidence", 0.5)
                )
            else:
                edge_map = detector.run_canny(
                    gray,
                    threshold1=edge_params.get("threshold1", 50.0),
                    threshold2=edge_params.get("threshold2", 150.0),
                )

            # Extrakcia segmentov z ROI inšpekčného obrázka (s offsetom)
            insp_roi = (rx + ox, ry + oy, rw, rh)
            segments = self._processor.extract_segments(edge_map, min_seg_len, roi=insp_roi)

            # Filtrovanie iba vybraných segmentov (podľa indexov z profilu)
            selected = [s for s in segments if s.index in segment_indices]

            # Renderovanie na prázdne plátno
            H, W = self._insp_image.shape[:2]
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for seg in selected:
                cv2.drawContours(canvas, [seg.contour], -1, (118, 230, 0), 1)

            return canvas
        except Exception:
            return self._seg_map

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Chyba inspekcie", msg)

    def _on_thread_done(self) -> None:
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Spustit inspekciu")
        self._thread = None
        self._worker = None

    # ------------------------------------------------------------------
    # Vysledky
    # ------------------------------------------------------------------

    def _populate_results(self, r: InspectionResult) -> None:
        dx, dy = r.shift_px
        self._lbl_shift_px.setText(f"dx={dx:+.2f}  dy={dy:+.2f}")

        if r.shift_mm is not None:
            mx, my = r.shift_mm
            self._lbl_shift_mm.setText(f"dx={mx:+.3f}  dy={my:+.3f}")
        else:
            self._lbl_shift_mm.setText("--")

        self._lbl_rot.setText(f"{r.rotation_deg:+.3f} deg")
        self._lbl_ncc.setText(f"{r.ncc_score:.4f}")
        self._lbl_duration.setText(f"{r.duration_ms:.1f} ms")

        color = _RELIABILITY_COLORS.get(r.reliability, "#888")
        self._lbl_reliability.setText(r.reliability)
        self._lbl_reliability.setStyleSheet(
            f"font-family: monospace; color: {color}; font-weight: bold;"
        )

        cx_r, cy_r = r.centroid_ref_px
        self._lbl_centroid_ref.setText(f"({cx_r:.1f}, {cy_r:.1f})")
        cx_i, cy_i = r.centroid_insp_px
        self._lbl_centroid_insp.setText(f"({cx_i:.1f}, {cy_i:.1f})")

    def _clear_results(self) -> None:
        for lbl in (
            self._lbl_shift_px, self._lbl_shift_mm, self._lbl_rot,
            self._lbl_ncc, self._lbl_reliability, self._lbl_duration,
            self._lbl_centroid_ref, self._lbl_centroid_insp,
        ):
            lbl.setText("--")
            lbl.setStyleSheet("font-family: monospace;")
