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

from core.inspection_engine import InspectionEngine, InspectionResult
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
    ) -> None:
        super().__init__()
        self._engine = engine
        self._ref = ref_img
        self._seg = seg_map
        self._insp = insp_img
        self._centroid = centroid_ref
        self._roi = roi
        self._roi_offset = roi_offset
        self._ecc_params = ecc_params
        self._px_per_mm = px_per_mm

    def run(self) -> None:
        try:
            result = self._engine.run(
                self._ref,
                self._seg,
                self._insp,
                self._centroid,
                self._roi,
                roi_offset=self._roi_offset,
                ecc_params=self._ecc_params,
                px_per_mm=self._px_per_mm,
            )
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# InspectionTab
# ---------------------------------------------------------------------------

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
        box = QGroupBox("ECC parametre")
        box.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 6px; } "
                          "QGroupBox::title { subcontrol-origin: margin; left: 6px; }")
        layout = QFormLayout(box)
        layout.setSpacing(4)

        self._combo_motion = QComboBox()
        self._combo_motion.addItems([
            "MOTION_TRANSLATION",
            "MOTION_EUCLIDEAN",
            "MOTION_AFFINE",
        ])
        self._combo_motion.setCurrentText("MOTION_EUCLIDEAN")
        layout.addRow("Typ pohybu:", self._combo_motion)

        self._spin_max_iter = QSpinBox()
        self._spin_max_iter.setRange(10, 5000)
        self._spin_max_iter.setValue(200)
        layout.addRow("Max iteracii:", self._spin_max_iter)

        self._edit_epsilon = QLineEdit("1e-5")
        layout.addRow("Epsilon:", self._edit_epsilon)

        # ROI offset
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

        # Restore ECC params from profile
        ecc = self._profile.get("ecc_params", {})
        if ecc.get("motion_type") in ("MOTION_TRANSLATION", "MOTION_EUCLIDEAN", "MOTION_AFFINE"):
            self._combo_motion.setCurrentText(ecc["motion_type"])
        self._spin_max_iter.setValue(int(ecc.get("max_iter", 200)))
        self._edit_epsilon.setText(str(ecc.get("epsilon", "1e-5")))

        offset = self._profile.get("roi_inspection_offset", {})
        self._spin_dx.setValue(int(offset.get("dx", 0)))
        self._spin_dy.setValue(int(offset.get("dy", 0)))

        parent.addWidget(box)

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

        # Overlay na inspekcionom vieweri (segment mapa pre vizualnu kontrolu)
        if self._seg_map is not None and self._chk_show_segments.isChecked():
            self._viewer_insp.set_overlay(self._seg_map)

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
