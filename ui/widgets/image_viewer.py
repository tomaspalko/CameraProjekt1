"""
ImageViewer — QWidget pre zobrazenie obrázka so zoom/pan, ROI výberom,
segmentovými overlaymi a nástrojmi pre klikanie.

Módy:
  "view"           — len zoom/pan
  "roi"            — rubber-band výber ROI (ľavý klik + ťahanie)
  "click_segment"  — kliknutie vysiela pixel_clicked
  "scale_tool"     — umiestňovanie kalibračných bodov (1. a 2. klik)

Signály:
  roi_selected(QRect)              — po dokončení rubber-band výberu
  pixel_clicked(QPoint)            — kliknutie v móde click_segment
  pixel_hovered(QPoint)            — pohyb myšou (vždy, vo všetkých módoch)
  scale_point_placed(int, QPoint)  — (index=0/1, bod) v móde scale_tool
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import (
    QPoint,
    QPointF,
    QRect,
    QRectF,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

_MODE_CURSOR = {
    "view": Qt.CursorShape.OpenHandCursor,
    "roi": Qt.CursorShape.CrossCursor,
    "click_segment": Qt.CursorShape.PointingHandCursor,
    "scale_tool": Qt.CursorShape.CrossCursor,
}

_ROI_COLOR = QColor(0, 120, 215)       # #0078d7
_SCALE_COLOR = QColor(255, 220, 0)     # žltá pre kalibračné body
_CROSSHAIR_COLOR = QColor(0, 230, 118) # #00e676 zelená

_MIN_ZOOM = 0.05
_MAX_ZOOM = 20.0


class ImageViewer(QWidget):
    """Interaktívny viewer s zoom/pan, ROI a overlaymi."""

    roi_selected = pyqtSignal(QRect)
    pixel_clicked = pyqtSignal(QPoint)
    pixel_hovered = pyqtSignal(QPoint)
    scale_point_placed = pyqtSignal(int, QPoint)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)

        self._pixmap: Optional[QPixmap] = None
        self._overlay_pixmap: Optional[QPixmap] = None

        # Zoom & pan
        self._zoom: float = 1.0
        self._pan_offset: QPointF = QPointF(0.0, 0.0)
        self._pan_start: Optional[QPointF] = None
        self._pan_origin: Optional[QPointF] = None

        # ROI rubber-band
        self._roi_start: Optional[QPoint] = None
        self._roi_current: Optional[QRect] = None
        self._roi_rect: Optional[QRect] = None       # potvrdený ROI v image coords

        # Scale body (image coords)
        self._scale_points: list[Optional[QPoint]] = [None, None]
        self._scale_next_idx: int = 0

        # Crosshairs (image coords)
        self._crosshairs: list[tuple[float, float]] = []

        # Mód
        self._mode: str = "view"
        self._update_cursor()

    # ------------------------------------------------------------------
    # Verejné API
    # ------------------------------------------------------------------

    def set_image(self, image: np.ndarray) -> None:
        """Nastav obrázok (uint8 BGR alebo grayscale)."""
        if image is None:
            self._pixmap = None
            self.update()
            return
        self._pixmap = self._ndarray_to_pixmap(image)
        self._fit_to_view()
        self.update()

    def set_overlay(self, overlay: Optional[np.ndarray]) -> None:
        """Nastav overlay obrázok (BGR, rovnaké rozmery ako hlavný obrázok). None = bez overlay."""
        if overlay is None:
            self._overlay_pixmap = None
        else:
            self._overlay_pixmap = self._ndarray_to_pixmap(overlay)
        self.update()

    def set_roi(self, roi: Optional[QRect]) -> None:
        """Nastav zobrazovaný ROI (image coords). None = skryť."""
        self._roi_rect = roi
        self.update()

    def set_zoom(self, zoom: float) -> None:
        """Nastav zoom faktor (1.0 = 100 %)."""
        self._zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, zoom))
        self.update()

    def set_mode(self, mode: str) -> None:
        """Prepni mód: "view" | "roi" | "click_segment" | "scale_tool"."""
        assert mode in _MODE_CURSOR, f"Neznámy mód: {mode}"
        self._mode = mode
        if mode == "scale_tool":
            self._scale_points = [None, None]
            self._scale_next_idx = 0
        self._update_cursor()
        self.update()

    def set_crosshairs(self, points: list[tuple[float, float]]) -> None:
        """Nastav zoznam bodov pre krížiky (image coords)."""
        self._crosshairs = list(points)
        self.update()

    def get_roi(self) -> Optional[QRect]:
        return self._roi_rect

    def get_scale_points(self) -> list[Optional[QPoint]]:
        return list(self._scale_points)

    def clear_scale_points(self) -> None:
        self._scale_points = [None, None]
        self._scale_next_idx = 0
        self.update()

    # ------------------------------------------------------------------
    # Interné — koordináty
    # ------------------------------------------------------------------

    def _image_to_widget(self, pt: QPointF) -> QPointF:
        """Prevedie bod z image súradníc na widget súradnice."""
        return QPointF(
            pt.x() * self._zoom + self._pan_offset.x(),
            pt.y() * self._zoom + self._pan_offset.y(),
        )

    def _widget_to_image(self, pt: QPointF) -> QPointF:
        """Prevedie bod z widget súradníc na image súradnice."""
        return QPointF(
            (pt.x() - self._pan_offset.x()) / self._zoom,
            (pt.y() - self._pan_offset.y()) / self._zoom,
        )

    def _fit_to_view(self) -> None:
        """Nastaví zoom a pan tak, aby obrázok vyplnil widget."""
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0 or ww == 0 or wh == 0:
            return
        self._zoom = min(ww / pw, wh / ph)
        self._pan_offset = QPointF(
            (ww - pw * self._zoom) / 2.0,
            (wh - ph * self._zoom) / 2.0,
        )

    # ------------------------------------------------------------------
    # Interné — konverzia ndarray → QPixmap
    # ------------------------------------------------------------------

    @staticmethod
    def _ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
        img = np.ascontiguousarray(image)
        if img.ndim == 2:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        # Udržiame referenciu na dáta
        pixmap = QPixmap.fromImage(qimg.copy())
        return pixmap

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        if self._pixmap is not None:
            self._fit_to_view()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1 / 1.15)
        cursor_pos = QPointF(event.position())
        img_pt = self._widget_to_image(cursor_pos)

        new_zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self._zoom * factor))
        self._pan_offset = QPointF(
            cursor_pos.x() - img_pt.x() * new_zoom,
            cursor_pos.y() - img_pt.y() * new_zoom,
        )
        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        img_pt = self._widget_to_image(pos)
        img_qpt = QPoint(int(img_pt.x()), int(img_pt.y()))

        if event.button() == Qt.MouseButton.MiddleButton:
            # Pan — stredné tlačidlo
            self._pan_start = pos
            self._pan_origin = QPointF(self._pan_offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == "view":
                self._pan_start = pos
                self._pan_origin = QPointF(self._pan_offset)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

            elif self._mode == "roi":
                self._roi_start = img_qpt
                self._roi_current = QRect(img_qpt, img_qpt)

            elif self._mode == "click_segment":
                self.pixel_clicked.emit(img_qpt)

            elif self._mode == "scale_tool":
                if self._scale_next_idx < 2:
                    idx = self._scale_next_idx
                    self._scale_points[idx] = img_qpt
                    self._scale_next_idx += 1
                    self.scale_point_placed.emit(idx, img_qpt)
                    self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        img_pt = self._widget_to_image(pos)
        img_qpt = QPoint(int(img_pt.x()), int(img_pt.y()))
        self.pixel_hovered.emit(img_qpt)

        # Pan
        if self._pan_start is not None:
            delta = pos - self._pan_start
            self._pan_offset = self._pan_origin + delta
            self.update()
            return

        # ROI rubber-band
        if self._mode == "roi" and self._roi_start is not None:
            self._roi_current = QRect(self._roi_start, img_qpt).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            if self._pan_start is not None:
                self._pan_start = None
                self._pan_origin = None
                self._update_cursor()

            if self._mode == "roi" and self._roi_current is not None:
                if self._roi_current.width() > 5 and self._roi_current.height() > 5:
                    self._roi_rect = self._roi_current
                    self.roi_selected.emit(self._roi_rect)
                self._roi_start = None
                self._roi_current = None
                self.update()

    # ------------------------------------------------------------------
    # Kreslenie
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Pozadie
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self._pixmap is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Žiadny obrázok")
            return

        # Hlavný obrázok
        img_w = self._pixmap.width() * self._zoom
        img_h = self._pixmap.height() * self._zoom
        dst = QRectF(self._pan_offset.x(), self._pan_offset.y(), img_w, img_h)
        painter.drawPixmap(dst, self._pixmap, QRectF(self._pixmap.rect()))

        # Overlay (segment mapa)
        if self._overlay_pixmap is not None:
            painter.setOpacity(0.6)
            painter.drawPixmap(dst, self._overlay_pixmap, QRectF(self._overlay_pixmap.rect()))
            painter.setOpacity(1.0)

        # ROI obdĺžnik
        if self._roi_rect is not None:
            self._draw_roi(painter, self._roi_rect, _ROI_COLOR)

        # Rubber-band v priebehu
        if self._roi_current is not None:
            self._draw_roi(painter, self._roi_current, _ROI_COLOR, dashed=True)

        # Scale body
        for idx, pt in enumerate(self._scale_points):
            if pt is not None:
                w_pt = self._image_to_widget(QPointF(pt))
                self._draw_crosshair(painter, w_pt, _SCALE_COLOR, size=10, label=str(idx + 1))

        # Krížiky (centroids)
        for cx, cy in self._crosshairs:
            w_pt = self._image_to_widget(QPointF(cx, cy))
            self._draw_crosshair(painter, w_pt, _CROSSHAIR_COLOR, size=12)

        painter.end()

    def _draw_roi(
        self,
        painter: QPainter,
        rect: QRect,
        color: QColor,
        dashed: bool = False,
    ) -> None:
        tl = self._image_to_widget(QPointF(rect.topLeft()))
        br = self._image_to_widget(QPointF(rect.bottomRight()))
        w_rect = QRectF(tl, br)

        pen = QPen(color, 2)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(w_rect)

    def _draw_crosshair(
        self,
        painter: QPainter,
        center: QPointF,
        color: QColor,
        size: int = 10,
        label: Optional[str] = None,
    ) -> None:
        pen = QPen(color, 2)
        painter.setPen(pen)
        cx, cy = center.x(), center.y()
        painter.drawLine(
            QPointF(cx - size, cy), QPointF(cx + size, cy)
        )
        painter.drawLine(
            QPointF(cx, cy - size), QPointF(cx, cy + size)
        )
        if label:
            painter.setPen(QPen(color))
            painter.drawText(QPointF(cx + size + 2, cy - 2), label)

    def _update_cursor(self) -> None:
        self.setCursor(QCursor(_MODE_CURSOR.get(self._mode, Qt.CursorShape.ArrowCursor)))
