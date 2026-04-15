"""
InspectionEngine — vyhľadanie segmentov v inšpekčnom obrázku pomocou ECC.

Algoritmus:
  1. Orezanie referenčného obrázka a inšpekčného obrázka na ROI (+ offset).
  2. Konverzia na float32 grayscale.
  3. cv2.findTransformECC(template=segment_map_crop, input=inspection_crop).
  4. Aplikácia warp matice na referenčné tažisko → inšpekčné tažisko.
  5. Extrakcia rotácie z warp matice (pre MOTION_EUCLIDEAN).
  6. Výpočet NCC skóre cez cv2.matchTemplate po zarovnaní.
  7. Klasifikácia spoľahlivosti: HIGH ≥ 0.8, MEDIUM ≥ 0.5, LOW < 0.5.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.scale_calculator import ScaleCalculator

# Podporované typy pohybu pre ECC
MOTION_TYPES: dict[str, int] = {
    "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
    "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
    "MOTION_AFFINE": cv2.MOTION_AFFINE,
}

_RELIABILITY_HIGH = 0.8
_RELIABILITY_MEDIUM = 0.5


@dataclass
class InspectionResult:
    """Výsledok inšpekcie jedného profilu."""

    centroid_ref_px: tuple[float, float]
    centroid_insp_px: tuple[float, float]
    centroid_ref_mm: Optional[tuple[float, float]]
    centroid_insp_mm: Optional[tuple[float, float]]
    shift_px: tuple[float, float]           # (dx, dy) inšp - ref
    shift_mm: Optional[tuple[float, float]]
    rotation_deg: float                     # v stupňoch, kladné = proti smeru hodinových ručičiek
    ncc_score: float                        # [-1, 1], bližšie k 1 = lepšia zhoda
    duration_ms: float
    reliability: str                        # "HIGH" | "MEDIUM" | "LOW"


class InspectionEngine:
    """
    ECC-based vyhľadávanie segmentov v inšpekčnom obrázku.

    Args:
        scale_calculator: Voliteľný kalibrovaný ScaleCalculator pre mm výstupy.
    """

    def __init__(self, scale_calculator: Optional[ScaleCalculator] = None) -> None:
        self._scale = scale_calculator

    def run(
        self,
        reference_image: np.ndarray,
        segment_map: np.ndarray,
        inspection_image: np.ndarray,
        centroid_ref: tuple[float, float],
        roi: tuple[int, int, int, int],
        roi_offset: tuple[int, int] = (0, 0),
        ecc_params: Optional[dict] = None,
        px_per_mm: Optional[float] = None,
    ) -> InspectionResult:
        """
        Spustí inšpekciu.

        Args:
            reference_image:  Referenčný obrázok (uint8, grayscale alebo BGR).
            segment_map:      BGR mapa segmentov rovnakej veľkosti ako referenčný obrázok.
            inspection_image: Inšpekčný obrázok — musí mať rovnaké rozmery ako reference_image.
            centroid_ref:     (cx, cy) tažisko v referenčnom obrázku (full-image px).
            roi:              (x, y, w, h) oblasť záujmu.
            roi_offset:       (dx, dy) posun ROI pre inšpekčný obrázok.
            ecc_params:       Slovník s kľúčmi: motion_type (str), max_iter (int), epsilon (float).
            px_per_mm:        Ak zadané, vypočítajú sa aj mm výstupy.

        Returns:
            InspectionResult

        Raises:
            ValueError: ak rozmery inšpekčného obrázka sa líšia od referenčného.
        """
        # Validácia rozmerov
        if reference_image.shape[:2] != inspection_image.shape[:2]:
            raise ValueError(
                f"Inšpekčný obrázok má iné rozmery "
                f"({inspection_image.shape[:2]}) ako referenčný "
                f"({reference_image.shape[:2]})."
            )

        params = ecc_params or {}
        motion_type_str = params.get("motion_type", "MOTION_EUCLIDEAN")
        motion_type = MOTION_TYPES.get(motion_type_str, cv2.MOTION_EUCLIDEAN)
        max_iter = int(params.get("max_iter", 200))
        epsilon = float(params.get("epsilon", 1e-5))

        rx, ry, rw, rh = roi
        ox, oy = roi_offset

        t_start = time.perf_counter()

        # --- Orezanie na ROI ---
        ref_crop = self._crop_gray(reference_image, rx, ry, rw, rh)
        seg_crop = self._crop_gray(segment_map, rx, ry, rw, rh)
        insp_crop = self._crop_gray(
            inspection_image,
            rx + ox, ry + oy, rw, rh,
            clamp_shape=reference_image.shape[:2],
        )

        # --- ECC registrácia ---
        warp_matrix = self._init_warp_matrix(motion_type)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            epsilon,
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=seg_crop.astype(np.float32),
                inputImage=insp_crop.astype(np.float32),
                warpMatrix=warp_matrix,
                motionType=motion_type,
                criteria=criteria,
            )
        except cv2.error:
            # ECC môže zlyháť pri veľmi nízkej konvergencii — použijeme identitu
            warp_matrix = self._init_warp_matrix(motion_type)

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        # --- Transformácia tažiska ---
        # Centroid je v full-image priestore; prevedieme do ROI priestoru,
        # aplikujeme warp (template→inspection), potom prevedieme späť.
        centroid_ref_roi = (
            centroid_ref[0] - rx,
            centroid_ref[1] - ry,
        )
        centroid_insp_roi = self._transform_point(centroid_ref_roi, warp_matrix, motion_type)
        centroid_insp_px = (
            centroid_insp_roi[0] + rx + ox,
            centroid_insp_roi[1] + ry + oy,
        )

        # --- Rotácia ---
        rotation_deg = self._extract_rotation(warp_matrix, motion_type)

        # --- NCC skóre — porovnáme zarovnaný insp_crop s ref_crop ---
        ncc_score = self._compute_ncc(ref_crop, insp_crop, warp_matrix, motion_type)

        # --- mm výstupy ---
        scale = px_per_mm or (
            self._scale.px_per_mm if self._scale and self._scale.px_per_mm else None
        )

        shift_px = (
            centroid_insp_px[0] - centroid_ref[0],
            centroid_insp_px[1] - centroid_ref[1],
        )

        if scale is not None:
            centroid_ref_mm = (centroid_ref[0] / scale, centroid_ref[1] / scale)
            centroid_insp_mm = (centroid_insp_px[0] / scale, centroid_insp_px[1] / scale)
            shift_mm: Optional[tuple[float, float]] = (shift_px[0] / scale, shift_px[1] / scale)
        else:
            centroid_ref_mm = None
            centroid_insp_mm = None
            shift_mm = None

        reliability = self._classify_reliability(ncc_score)

        return InspectionResult(
            centroid_ref_px=centroid_ref,
            centroid_insp_px=centroid_insp_px,
            centroid_ref_mm=centroid_ref_mm,
            centroid_insp_mm=centroid_insp_mm,
            shift_px=shift_px,
            shift_mm=shift_mm,
            rotation_deg=rotation_deg,
            ncc_score=ncc_score,
            duration_ms=duration_ms,
            reliability=reliability,
        )

    # ------------------------------------------------------------------
    # Interné pomocné metódy
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_gray(
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        clamp_shape: Optional[tuple[int, int]] = None,
    ) -> np.ndarray:
        """Orezanie obrázka na zadanú oblasť, konverzia na uint8 grayscale."""
        H, W = image.shape[:2]
        if clamp_shape:
            H, W = clamp_shape

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)

        crop = image[y0:y1, x0:x1]
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return crop

    @staticmethod
    def _init_warp_matrix(motion_type: int) -> np.ndarray:
        """Inicializuje warp maticu ako identitu pre daný typ pohybu."""
        if motion_type == cv2.MOTION_HOMOGRAPHY:
            return np.eye(3, 3, dtype=np.float32)
        return np.eye(2, 3, dtype=np.float32)

    @staticmethod
    def _transform_point(
        point: tuple[float, float],
        warp_matrix: np.ndarray,
        motion_type: int,
    ) -> tuple[float, float]:
        """Aplikuje warp maticu na bod (px, py)."""
        px, py = point
        if motion_type == cv2.MOTION_HOMOGRAPHY:
            src = np.array([[[px, py]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, warp_matrix)
            return (float(dst[0, 0, 0]), float(dst[0, 0, 1]))
        else:
            # Afinná / euklidovská / translačná: 2×3 matica
            M = warp_matrix
            new_x = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            new_y = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            return (float(new_x), float(new_y))

    @staticmethod
    def _extract_rotation(warp_matrix: np.ndarray, motion_type: int) -> float:
        """
        Extrahuje rotáciu v stupňoch z warp matice.
        Platné pre MOTION_EUCLIDEAN a MOTION_AFFINE.
        Pre MOTION_TRANSLATION je rotácia vždy 0.
        """
        if motion_type == cv2.MOTION_TRANSLATION:
            return 0.0
        # R = [[cos θ, -sin θ], [sin θ, cos θ]]
        return math.degrees(math.atan2(float(warp_matrix[1, 0]), float(warp_matrix[0, 0])))

    @staticmethod
    def _compute_ncc(
        template: np.ndarray,
        query: np.ndarray,
        warp_matrix: np.ndarray,
        motion_type: int,
    ) -> float:
        """
        Vypočíta NCC skóre medzi template a zarovnaným query obrázkom.
        Vracia hodnotu v rozsahu [-1, 1].
        """
        try:
            # Zarovnaj query podľa nájdenej warp matice
            h, w = template.shape[:2]
            if motion_type == cv2.MOTION_HOMOGRAPHY:
                aligned = cv2.warpPerspective(
                    query.astype(np.float32), warp_matrix, (w, h)
                )
            else:
                aligned = cv2.warpAffine(
                    query.astype(np.float32), warp_matrix, (w, h)
                )

            tmpl_f = template.astype(np.float32)
            result = cv2.matchTemplate(aligned, tmpl_f, cv2.TM_CCOEFF_NORMED)
            return float(result.max())
        except cv2.error:
            return 0.0

    @staticmethod
    def _classify_reliability(ncc_score: float) -> str:
        if ncc_score >= _RELIABILITY_HIGH:
            return "HIGH"
        if ncc_score >= _RELIABILITY_MEDIUM:
            return "MEDIUM"
        return "LOW"
