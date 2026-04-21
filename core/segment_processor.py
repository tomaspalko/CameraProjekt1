"""
SegmentProcessor — extrakcia kontúr, výpočet tažísk a renderovanie máp segmentov.

Dôležité:
  - Súradnice kontúr sú VŽDY v súradnicovom priestore celého obrázka (full-image space).
    Ak je zadané ROI, kontúry sa okamžite odsadia o (roi_x, roi_y).
  - Kombinované tažisko = vážený priemer individuálnych tažísk (váha = dĺžka oblúka).
  - Mapa segmentov: čierne pozadie, vybrané segmenty = zelené, ostatné = sivé.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Segment:
    """Jeden detekovaný segment (kontúra) v obrázku."""

    index: int
    contour: np.ndarray                       # shape (N, 1, 2), dtype int32
    centroid: tuple[float, float]             # (cx, cy) v full-image súradniciach
    arc_length: float                         # obvod kontúry v pixeloch
    bounding_rect: tuple[int, int, int, int]  # (x, y, w, h) v full-image súradniciach


class SegmentProcessor:
    """
    Spracovanie segmentov z binárnej mapy hrán.

    Použitie:
        proc = SegmentProcessor()
        segments = proc.extract_segments(edge_map, min_length=20, roi=(x,y,w,h))
        centroid = proc.compute_combined_centroid(segments, selected_indices=[0, 2])
        map_img  = proc.render_segment_map(image_shape, segments, selected_indices=[0, 2])
    """

    # ------------------------------------------------------------------
    # Extrakcia segmentov
    # ------------------------------------------------------------------

    def extract_segments(
        self,
        edge_map: np.ndarray,
        min_length: float = 20.0,
        roi: Optional[tuple[int, int, int, int]] = None,
    ) -> list[Segment]:
        """
        Extrahuje kontúry z binárnej mapy hrán.

        Args:
            edge_map:   Binárna uint8 mapa hrán (hodnoty 0/255), môže byť celý obrázok.
            min_length: Minimálna dĺžka oblúka kontúry v pixeloch (kratšie sú vynechané).
            roi:        Voliteľné (x, y, w, h) — ak zadané, edge_map sa orezáva na ROI
                        a súradnice kontúr sú preložené do full-image priestoru.

        Returns:
            Zoznam Segment objektov zoradených zostupne podľa arc_length.
        """
        if roi is not None:
            rx, ry, rw, rh = roi
            crop = edge_map[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry
        else:
            crop = edge_map
            offset_x, offset_y = 0, 0

        contours, _ = cv2.findContours(
            crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        segments: list[Segment] = []
        for idx, cnt in enumerate(contours):
            length = cv2.arcLength(cnt, closed=False)
            if length < min_length:
                continue

            # Preloženie súradníc do full-image priestoru
            shifted = cnt.copy()
            shifted[:, :, 0] += offset_x
            shifted[:, :, 1] += offset_y

            # Tažisko cez momenty
            M = cv2.moments(shifted)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = float(shifted[:, 0, 0].mean())
                cy = float(shifted[:, 0, 1].mean())

            bx, by, bw, bh = cv2.boundingRect(shifted)

            segments.append(
                Segment(
                    index=idx,
                    contour=shifted,
                    centroid=(cx, cy),
                    arc_length=length,
                    bounding_rect=(bx, by, bw, bh),
                )
            )

        # Zoradiť zostupne podľa dĺžky oblúka
        segments.sort(key=lambda s: s.arc_length, reverse=True)

        # Preindexovanie (index = poradie po zoradení)
        for new_idx, seg in enumerate(segments):
            seg.index = new_idx

        return segments

    # ------------------------------------------------------------------
    # Výpočet tažiska
    # ------------------------------------------------------------------

    def compute_combined_centroid(
        self,
        segments: list[Segment],
        selected_indices: list[int],
        trimmed_contours: Optional[dict[int, np.ndarray]] = None,
    ) -> tuple[float, float]:
        """
        Vypočíta vážené tažisko vybraných segmentov.

        Váha každého segmentu = jeho arc_length (alebo dĺžka orezanej kontúry).
        Ak je pre segment dostupná orezaná kontúra, použije sa na výpočet tažiska.
        Raises ValueError ak selected_indices je prázdny.
        """
        if not selected_indices:
            raise ValueError("Musí byť vybraný aspoň jeden segment.")

        selected = [s for s in segments if s.index in selected_indices]
        if not selected:
            raise ValueError("Žiadny zo zadaných indexov nebol nájdený v zozname segmentov.")

        tc = trimmed_contours or {}
        total_weight = 0.0
        cx_acc = 0.0
        cy_acc = 0.0

        for s in selected:
            if s.index in tc:
                trimmed = tc[s.index]
                M = cv2.moments(trimmed)
                if M["m00"] != 0:
                    cx_i = M["m10"] / M["m00"]
                    cy_i = M["m01"] / M["m00"]
                else:
                    cx_i = float(trimmed[:, 0, 0].mean())
                    cy_i = float(trimmed[:, 0, 1].mean())
                weight = float(cv2.arcLength(trimmed, closed=False))
            else:
                cx_i, cy_i = s.centroid
                weight = s.arc_length
            cx_acc += cx_i * weight
            cy_acc += cy_i * weight
            total_weight += weight

        if total_weight == 0:
            cx = sum(s.centroid[0] for s in selected) / len(selected)
            cy = sum(s.centroid[1] for s in selected) / len(selected)
        else:
            cx = cx_acc / total_weight
            cy = cy_acc / total_weight

        return (cx, cy)

    # ------------------------------------------------------------------
    # Renderovanie mapy segmentov
    # ------------------------------------------------------------------

    def render_segment_map(
        self,
        image_shape: tuple[int, int],
        segments: list[Segment],
        selected_indices: list[int],
        trimmed_contours: Optional[dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Vykreslí farebnú mapu segmentov na čiernom pozadí.

        - Vybrané segmenty: zelená (0, 255, 0)
        - Ostatné segmenty: sivá (117, 117, 117)
        - Ak je pre segment dostupná orezaná kontúra, vykreslí sa namiesto plnej.

        Args:
            image_shape:       (H, W) výstupného obrázka.
            segments:          Zoznam všetkých segmentov.
            selected_indices:  Indexy vybraných segmentov.
            trimmed_contours:  Voliteľný slovník {index: orezaná kontúra}.

        Returns:
            BGR uint8 obrázok tvaru (H, W, 3).
        """
        h, w = image_shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        selected_set = set(selected_indices)
        tc = trimmed_contours or {}

        # Najprv nevybrané (sivé), potom vybrané (zelené) — vybrané sú navrchu
        for seg in segments:
            if seg.index not in selected_set:
                contour = tc.get(seg.index, seg.contour)
                cv2.drawContours(canvas, [contour], -1, (117, 117, 117), 1)

        for seg in segments:
            if seg.index in selected_set:
                contour = tc.get(seg.index, seg.contour)
                cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 1)

        return canvas

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def hit_test_segment(
        self,
        segments: list[Segment],
        click_point: tuple[int, int],
        tolerance_px: int = 5,
    ) -> Optional[int]:
        """
        Vráti index segmentu najbližšieho ku kliknutému bodu (v rámci tolerancie).

        Používa cv2.pointPolygonTest pre každý segment.
        Vráti None ak žiadny segment nie je v rámci tolerance_px.
        """
        cx, cy = click_point
        best_index: Optional[int] = None
        best_dist = float("inf")

        for seg in segments:
            # pointPolygonTest: kladné = vnútri, záporné = vonku, absolútna hodnota = vzdialenosť
            dist = abs(cv2.pointPolygonTest(seg.contour, (float(cx), float(cy)), measureDist=True))
            if dist <= tolerance_px and dist < best_dist:
                best_dist = dist
                best_index = seg.index

        return best_index

    @staticmethod
    def trim_contour_to_rect(
        contour: np.ndarray,
        rect: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Oreže kontúru na body, ktoré ležia vnútri obdĺžnika rect.

        Args:
            contour: Kontúra tvaru (N, 1, 2), dtype int32.
            rect:    (x, y, w, h) v súradniciach celého obrázka.

        Returns:
            Orezaná kontúra tvaru (M, 1, 2) alebo None ak zostanú menej ako 2 body.
        """
        x0, y0, rw, rh = rect
        x1, y1 = x0 + rw, y0 + rh
        pts = contour[:, 0, :]  # (N, 2)
        mask = (
            (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
            (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        )
        kept = pts[mask]
        if len(kept) < 2:
            return None
        return kept.reshape(-1, 1, 2).astype(np.int32)

    @staticmethod
    def fit_contour_to_line(contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Fituje kontúru na priamku pomocou cv2.fitLine (DIST_L2).

        Returns:
            Hustá kontúra (N, 1, 2) int32 pozdĺž priamky, alebo None ak < 2 bodov.
        """
        pts = contour[:, 0, :].astype(np.float32)  # (N, 2)
        if len(pts) < 2:
            return None
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy = float(line[0]), float(line[1])
        x0, y0 = float(line[2]), float(line[3])
        # Project all points onto line direction → find extent [t_min, t_max]
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        t_min, t_max = float(t.min()), float(t.max())
        n_pts = max(2, int(round(t_max - t_min)))
        ts = np.linspace(t_min, t_max, n_pts)
        xs = np.round(x0 + ts * vx).astype(np.int32)
        ys = np.round(y0 + ts * vy).astype(np.int32)
        return np.stack([xs, ys], axis=1).reshape(-1, 1, 2)

    def hit_test_area(
        self,
        segments: list[Segment],
        rect: tuple[int, int, int, int],
    ) -> list[int]:
        """
        Vráti indexy všetkých segmentov, ktorých bounding_rect sa prekrýva so zadaným rect.

        Args:
            segments: Zoznam segmentov.
            rect:     (x, y, w, h) oblasť výberu.

        Returns:
            Zoznam indexov prekrývajúcich sa segmentov.
        """
        rx, ry, rw, rh = rect
        rx2, ry2 = rx + rw, ry + rh

        result: list[int] = []
        for seg in segments:
            sx, sy, sw, sh = seg.bounding_rect
            sx2, sy2 = sx + sw, sy + sh
            # AABB prekrytie
            if sx < rx2 and sx2 > rx and sy < ry2 and sy2 > ry:
                result.append(seg.index)

        return result
