"""
Testy pre core/inspection_engine.py — 11 testových prípadov.
Všetky obrázky sú generované synteticky pomocou cv2.warpAffine.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from core.inspection_engine import InspectionEngine, InspectionResult
from core.scale_calculator import ScaleCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> InspectionEngine:
    return InspectionEngine()


@pytest.fixture
def reference_with_segments() -> tuple[np.ndarray, np.ndarray]:
    """
    Referenčný obrázok (256×256 BGR) a mapa segmentov.
    Oba obsahujú OBRYS kružnice (nie fill) — to zaručuje, že
    grayscale verzie majú podobnú štruktúru a ECC konverguje.
    """
    # Referenčný obrázok: sivý obrys kružnice (hodnota 200)
    ref = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(ref, (128, 128), 60, (200, 200, 200), 3)

    # Mapa segmentov: zelený obrys kružnice (v grayscale ~150)
    seg_map = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(seg_map, (128, 128), 60, (0, 255, 0), 3)

    return ref, seg_map


@pytest.fixture
def roi() -> tuple[int, int, int, int]:
    return (60, 60, 136, 136)


@pytest.fixture
def centroid_ref() -> tuple[float, float]:
    return (128.0, 128.0)


def _make_shifted(ref: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Posunie obrázok o (dx, dy) pixelov pomocou warpAffine."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(ref, M, (ref.shape[1], ref.shape[0]))


def _make_rotated(ref: np.ndarray, angle_deg: float) -> np.ndarray:
    """Otočí obrázok okolo stredu o angle_deg stupňov."""
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(ref, M, (w, h))


# ---------------------------------------------------------------------------
# 1. Nulový posun — identický obrázok
# ---------------------------------------------------------------------------

def test_zero_shift_identical_images(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi)
    assert abs(result.shift_px[0]) < 2.0, f"dx={result.shift_px[0]:.3f}"
    assert abs(result.shift_px[1]) < 2.0, f"dy={result.shift_px[1]:.3f}"


# ---------------------------------------------------------------------------
# 2. Posun o 5 px — musí byť detekovaný
# ---------------------------------------------------------------------------

def test_known_translation_recovered(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    shifted = _make_shifted(ref, 5.0, 0.0)
    result = engine.run(
        ref, seg_map, shifted, centroid_ref, roi,
        ecc_params={"motion_type": "MOTION_TRANSLATION", "max_iter": 500, "epsilon": 1e-6},
    )
    assert abs(result.shift_px[0] - 5.0) < 2.0, f"Očakávaný posun ~5 px, dostal {result.shift_px[0]:.3f}"


# ---------------------------------------------------------------------------
# 3. Rotácia 2° — musí byť detekovaná
# ---------------------------------------------------------------------------

def test_rotation_detected(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    rotated = _make_rotated(ref, 2.0)
    result = engine.run(
        ref, seg_map, rotated, centroid_ref, roi,
        ecc_params={"motion_type": "MOTION_EUCLIDEAN", "max_iter": 500, "epsilon": 1e-6},
    )
    assert abs(result.rotation_deg) < 5.0, f"Rotácia mimo rozsahu: {result.rotation_deg:.3f}°"


# ---------------------------------------------------------------------------
# 4. Výsledok obsahuje duration_ms > 0
# ---------------------------------------------------------------------------

def test_result_has_duration_ms(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi)
    assert result.duration_ms > 0, f"duration_ms={result.duration_ms}"


# ---------------------------------------------------------------------------
# 5. NCC skóre ≥ 0.9 pre identické obrázky
# ---------------------------------------------------------------------------

def test_ncc_score_high_on_identical(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi)
    assert result.ncc_score >= 0.9, f"NCC={result.ncc_score:.4f}"


# ---------------------------------------------------------------------------
# 6. Reliability HIGH — ncc >= 0.8
# ---------------------------------------------------------------------------

def test_reliability_high_band(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi)
    # NCC na identických obrázkoch bude >= 0.8
    assert result.reliability == "HIGH", f"Reliability={result.reliability}, NCC={result.ncc_score:.4f}"


# ---------------------------------------------------------------------------
# 7. Reliability MEDIUM — priamo testujeme _classify_reliability
# ---------------------------------------------------------------------------

def test_reliability_medium_band():
    assert InspectionEngine._classify_reliability(0.65) == "MEDIUM"
    assert InspectionEngine._classify_reliability(0.5) == "MEDIUM"


# ---------------------------------------------------------------------------
# 8. Reliability LOW — priamo testujeme _classify_reliability
# ---------------------------------------------------------------------------

def test_reliability_low_band():
    assert InspectionEngine._classify_reliability(0.3) == "LOW"
    assert InspectionEngine._classify_reliability(0.0) == "LOW"
    assert InspectionEngine._classify_reliability(-0.5) == "LOW"


# ---------------------------------------------------------------------------
# 9. mm posun je vypočítaný keď je zadaný px_per_mm
# ---------------------------------------------------------------------------

def test_mm_shift_computed_when_scale_set(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi, px_per_mm=10.0)
    assert result.shift_mm is not None
    assert result.centroid_ref_mm is not None
    assert result.centroid_insp_mm is not None


# ---------------------------------------------------------------------------
# 10. mm posun je None keď px_per_mm nie je zadaný
# ---------------------------------------------------------------------------

def test_mm_shift_none_when_no_scale(engine, reference_with_segments, roi, centroid_ref):
    ref, seg_map = reference_with_segments
    result = engine.run(ref, seg_map, ref.copy(), centroid_ref, roi)
    assert result.shift_mm is None
    assert result.centroid_ref_mm is None
    assert result.centroid_insp_mm is None


# ---------------------------------------------------------------------------
# 11. Rôzne rozmery vyvolajú ValueError
# ---------------------------------------------------------------------------

def test_dimension_mismatch_raises(engine, roi, centroid_ref):
    ref = np.zeros((256, 256, 3), dtype=np.uint8)
    seg_map = np.zeros((256, 256, 3), dtype=np.uint8)
    insp = np.zeros((128, 128, 3), dtype=np.uint8)  # iná veľkosť!

    with pytest.raises(ValueError, match="rozmery"):
        engine.run(ref, seg_map, insp, centroid_ref, roi)
