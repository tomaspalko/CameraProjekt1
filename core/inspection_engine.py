"""
InspectionEngine — vyhľadanie segmentov v inšpekčnom obrázku.

Podporované stratégie zarovnania (AlignmentStrategy):
  ECC_ONLY          – pôvodné správanie: cv2.findTransformECC
  TEMPLATE_ONLY     – cv2.matchTemplate, len translácia (rýchly, veľký rozsah)
  TEMPLATE_THEN_ECC – hrubé zarovnanie cez matchTemplate, jemné cez ECC

Algoritmus (ECC_ONLY):
  1. Orezanie referenčného obrázka a inšpekčného obrázka na ROI (+ offset).
  2. Konverzia na float32 grayscale.
  3. cv2.findTransformECC(template=segment_map_crop, input=inspection_crop).
  4. Aplikácia warp matice na referenčné tažisko → inšpekčné tažisko.
  5. Extrakcia rotácie z warp matice (pre MOTION_EUCLIDEAN).
  6. Výpočet NCC skóre cez cv2.matchTemplate po zarovnaní.
  7. Klasifikácia spoľahlivosti: HIGH ≥ 0.8, MEDIUM ≥ 0.5, LOW < 0.5.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.edge_detection import EdgeDetector
from core.scale_calculator import ScaleCalculator

# Podporované typy pohybu pre ECC
MOTION_TYPES: dict[str, int] = {
    "MOTION_TRANSLATION": cv2.MOTION_TRANSLATION,
    "MOTION_EUCLIDEAN": cv2.MOTION_EUCLIDEAN,
    "MOTION_AFFINE": cv2.MOTION_AFFINE,
}

# Podporované metódy pre template matching
_TM_METHODS: dict[str, int] = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
}

_RELIABILITY_HIGH = 0.8
_RELIABILITY_MEDIUM = 0.5


class AlignmentStrategy(enum.Enum):
    """Stratégia zarovnania inšpekčného obrázka na referenčný."""
    ECC_ONLY          = "ecc_only"
    TEMPLATE_ONLY     = "template_only"
    TEMPLATE_THEN_ECC = "template_then_ecc"


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
    Vyhľadávanie segmentov v inšpekčnom obrázku.

    Podporuje tri stratégie zarovnania: ECC, template matching, alebo ich kombináciu.

    Args:
        scale_calculator: Voliteľný kalibrovaný ScaleCalculator pre mm výstupy.
    """

    def __init__(self, scale_calculator: Optional[ScaleCalculator] = None) -> None:
        self._scale = scale_calculator
        self._edge_detector = EdgeDetector()

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
        edge_method: str = "canny",
        edge_params: Optional[dict] = None,
        alignment_strategy: AlignmentStrategy = AlignmentStrategy.ECC_ONLY,
        template_params: Optional[dict] = None,
    ) -> InspectionResult:
        """
        Spustí inšpekciu.

        Args:
            reference_image:    Referenčný obrázok (uint8, grayscale alebo BGR).
            segment_map:        BGR mapa segmentov rovnakej veľkosti ako referenčný obrázok.
            inspection_image:   Inšpekčný obrázok — musí mať rovnaké rozmery ako reference_image.
            centroid_ref:       (cx, cy) tažisko v referenčnom obrázku (full-image px).
            roi:                (x, y, w, h) oblasť záujmu.
            roi_offset:         (dx, dy) posun ROI pre inšpekčný obrázok.
            ecc_params:         Slovník: motion_type (str), max_iter (int), epsilon (float).
            px_per_mm:          Ak zadané, vypočítajú sa aj mm výstupy.
            edge_method:        "canny" alebo "dexined".
            edge_params:        Parametre detekcie hrán.
            alignment_strategy: Jedna z AlignmentStrategy hodnôt.
            template_params:    Slovník: search_expansion (float), method (str).
                                search_expansion určuje, o koľkonásobok rozmeru ROI sa rozšíri
                                oblasť hľadania (default 0.5 = ±50 % ROI).

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

        # ECC parametre
        ecc = ecc_params or {}
        motion_type_str = ecc.get("motion_type", "MOTION_EUCLIDEAN")
        motion_type = MOTION_TYPES.get(motion_type_str, cv2.MOTION_EUCLIDEAN)
        max_iter = int(ecc.get("max_iter", 200))
        epsilon = float(ecc.get("epsilon", 1e-5))

        # Template matching parametre
        tm = template_params or {}
        search_expansion = float(tm.get("search_expansion", 0.5))
        tm_method = _TM_METHODS.get(tm.get("method", "TM_CCOEFF_NORMED"), cv2.TM_CCOEFF_NORMED)

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

        # --- Edge detection na inšpekčnom výreze (konzistentné s template) ---
        insp_ecc = self._prepare_ecc_input(insp_crop, edge_method, edge_params or {})

        # --- Zarovnanie podľa zvolenej stratégie ---
        if alignment_strategy == AlignmentStrategy.TEMPLATE_ONLY:
            # Template matching pracuje na reálnych fotografických dátach (ref vs insp),
            # nie na segment mape — obe sú z rovnakej domény, čo dáva správnu koreláciu.
            dx, dy = self._align_template(ref_crop, insp_crop, search_expansion, tm_method)
            warp_matrix = self._build_translation_warp(dx, dy)
            motion_type = cv2.MOTION_TRANSLATION  # rotácia = 0 pre čistú transláciu

        elif alignment_strategy == AlignmentStrategy.TEMPLATE_THEN_ECC:
            # Hrubé zarovnanie: template matching na reálnych obrázkoch (ref vs insp)
            # Jemné zarovnanie: ECC na segment mape (hranové štruktúry)
            dx, dy = self._align_template(ref_crop, insp_crop, search_expansion, tm_method)
            warp_init = self._build_translation_warp(dx, dy)
            warp_matrix = self._align_ecc(
                seg_crop, insp_ecc, motion_type, max_iter, epsilon, warp_init
            )

        else:  # ECC_ONLY (default)
            warp_matrix = self._align_ecc(
                seg_crop, insp_ecc, motion_type, max_iter, epsilon
            )

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
    # Interné zarovnávacie metódy
    # ------------------------------------------------------------------

    def _align_ecc(
        self,
        template: np.ndarray,
        query: np.ndarray,
        motion_type: int,
        max_iter: int,
        epsilon: float,
        warp_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Spustí cv2.findTransformECC.

        Args:
            warp_init: Počiatočná warp matica. Ak None, použije sa identita.
                       Pre TEMPLATE_THEN_ECC sa odovzdá translačný výsledok
                       z template matchingu ako seed.
        """
        warp_matrix = warp_init.copy() if warp_init is not None else self._init_warp_matrix(motion_type)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iter,
            epsilon,
        )
        try:
            _, warp_matrix = cv2.findTransformECC(
                templateImage=template.astype(np.float32),
                inputImage=query.astype(np.float32),
                warpMatrix=warp_matrix,
                motionType=motion_type,
                criteria=criteria,
            )
        except cv2.error:
            # ECC nedokonvergoval — zachovajme seed (warp_init) ak existuje,
            # inak vrátime identitu. Pre TEMPLATE_THEN_ECC tak zachováme hrubé
            # translačné zarovnanie z template matchingu.
            warp_matrix = warp_init.copy() if warp_init is not None else self._init_warp_matrix(motion_type)
        return warp_matrix

    def _align_template(
        self,
        template: np.ndarray,
        query: np.ndarray,
        search_expansion: float = 0.5,
        method: int = cv2.TM_CCOEFF_NORMED,
    ) -> tuple[float, float]:
        """
        Nájde translačný posun pomocou cv2.matchTemplate.

        Oblasť hľadania sa rozšíri paddingom o search_expansion * rozmery šablóny
        na každú stranu. Vracia (dx, dy) — posun inšpekčného obrázka voči referenčnému.

        Args:
            template: Výrez segment mapy (grayscale).
            query:    Edge-detected výrez inšpekčného obrázka (grayscale).
            search_expansion: Rozšírenie oblasti hľadania (0.5 = ±50 % rozmeru ROI).
            method:   Metóda matchTemplate (TM_CCOEFF_NORMED alebo TM_CCORR_NORMED).

        Returns:
            (dx, dy) posun v pixeloch.
        """
        pad_x = max(1, int(template.shape[1] * search_expansion))
        pad_y = max(1, int(template.shape[0] * search_expansion))

        padded = cv2.copyMakeBorder(
            query.astype(np.float32),
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_REPLICATE,
        )
        tmpl_f = template.astype(np.float32)

        if padded.shape[0] < tmpl_f.shape[0] or padded.shape[1] < tmpl_f.shape[1]:
            raise ValueError(
                f"Search area ({padded.shape}) je menší ako template ({tmpl_f.shape}). "
                "Zvýšte search_expansion."
            )

        result = cv2.matchTemplate(padded, tmpl_f, method)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        dx = float(max_loc[0] - pad_x)
        dy = float(max_loc[1] - pad_y)
        return dx, dy

    @staticmethod
    def _build_translation_warp(dx: float, dy: float) -> np.ndarray:
        """Vráti 2×3 warp maticu pre čistú transláciu o (dx, dy)."""
        M = np.eye(2, 3, dtype=np.float32)
        M[0, 2] = dx
        M[1, 2] = dy
        return M

    # ------------------------------------------------------------------
    # Pomocné statické metódy
    # ------------------------------------------------------------------

    def _prepare_ecc_input(
        self,
        gray_crop: np.ndarray,
        edge_method: str,
        edge_params: dict,
    ) -> np.ndarray:
        """Aplikuje detekciu hrán na výrez pred ECC — konzistentné s template (segment_map)."""
        if edge_method == "dexined":
            confidence = edge_params.get("confidence", 0.5)
            return self._edge_detector.run_dexined(gray_crop, confidence=confidence)
        # default: canny
        t1 = edge_params.get("threshold1", 50.0)
        t2 = edge_params.get("threshold2", 150.0)
        return self._edge_detector.run_canny(gray_crop, threshold1=t1, threshold2=t2)

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
