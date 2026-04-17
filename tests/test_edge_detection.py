"""
Testy pre core/edge_detection.py — 10 testových prípadov.

Testy Canny: bežia vždy (žiadne externé závislosti).
Testy DexiNed: označené @pytest.mark.slow, preskočené pokiaľ DEXINED_TESTS=1.
Test sťahovania váh: mockuje urllib — nevyžaduje internet ani torch.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from core.edge_detection import EdgeDetector

# Preskočiť DexiNed testy pokiaľ nie je explicitne povolené
DEXINED_ENABLED = os.environ.get("DEXINED_TESTS", "0") == "1"
skip_dexined = pytest.mark.skipif(
    not DEXINED_ENABLED,
    reason="Preskočené — nastav DEXINED_TESTS=1 pre spustenie DexiNed testov",
)


# ---------------------------------------------------------------------------
# Pomocná fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> EdgeDetector:
    return EdgeDetector()


@pytest.fixture
def circle_gray() -> np.ndarray:
    """256×256 grayscale obrázok s kružnicou — Canny ho ľahko detekuje."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, 255, 2)
    return img


@pytest.fixture
def circle_color() -> np.ndarray:
    """256×256 BGR obrázok s kružnicou."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, (255, 255, 255), 2)
    return img


# ---------------------------------------------------------------------------
# Canny testy (1–6)
# ---------------------------------------------------------------------------

def test_canny_output_shape_matches_input(detector, circle_gray):
    result = detector.run_canny(circle_gray)
    assert result.shape == circle_gray.shape[:2]


def test_canny_output_dtype_uint8(detector, circle_gray):
    result = detector.run_canny(circle_gray)
    assert result.dtype == np.uint8


def test_canny_binary_values(detector, circle_gray):
    result = detector.run_canny(circle_gray)
    unique = set(np.unique(result))
    assert unique.issubset({0, 255}), f"Nečakané hodnoty pixelov: {unique}"


def test_canny_detects_synthetic_circle_edge(detector, circle_gray):
    result = detector.run_canny(circle_gray, threshold1=30, threshold2=100)
    edge_pixels = np.count_nonzero(result)
    assert edge_pixels > 50, f"Príliš málo hrán detekovaných: {edge_pixels}"


def test_canny_accepts_color_input(detector, circle_color):
    """3-kanálový vstup musí fungovať bez výnimky."""
    result = detector.run_canny(circle_color)
    assert result.shape == circle_color.shape[:2]
    assert result.dtype == np.uint8


def test_canny_high_threshold_fewer_edges(detector, circle_gray):
    """Vyšší prah → menej detekovaných hrán."""
    low = detector.run_canny(circle_gray, threshold1=10, threshold2=50)
    high = detector.run_canny(circle_gray, threshold1=200, threshold2=250)
    assert np.count_nonzero(low) >= np.count_nonzero(high), (
        "Nízky prah by mal dávať aspoň toľko hrán ako vysoký prah"
    )


# ---------------------------------------------------------------------------
# Gaussian blur testy (7–10)
# ---------------------------------------------------------------------------

def test_canny_blur_disabled_matches_default(detector, circle_gray):
    """blur_kernel=0 musí dávať rovnaký výsledok ako volanie bez parametra."""
    default = detector.run_canny(circle_gray)
    explicit_off = detector.run_canny(circle_gray, blur_kernel=0)
    np.testing.assert_array_equal(default, explicit_off)


def test_canny_blur_reduces_noise_edges(detector):
    """blur_kernel=5 musí dávať menej hrán na zašumenom obrázku ako blur=0."""
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    without_blur = detector.run_canny(noisy, blur_kernel=0)
    with_blur = detector.run_canny(noisy, blur_kernel=5)
    assert np.count_nonzero(with_blur) < np.count_nonzero(without_blur), (
        "Gaussian blur mal redukovať počet falošných hrán na zašumenom obrázku"
    )


@pytest.mark.parametrize("k", [3, 5, 7])
def test_canny_blur_kernels_valid(detector, circle_gray, k):
    """Každý povolený kernel (3, 5, 7) musí vrátiť správny tvar a dtype."""
    result = detector.run_canny(circle_gray, blur_kernel=k)
    assert result.shape == circle_gray.shape[:2]
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_canny_blur_even_kernel_clamped(detector, circle_gray):
    """Párny kernel (4) nesmie vyvolať výnimku — interný clamp ho zaokrúhli na 5."""
    result = detector.run_canny(circle_gray, blur_kernel=4)
    assert result.shape == circle_gray.shape[:2]
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# DexiNed testy — vyžadujú stiahnuté ONNX váhy
# ---------------------------------------------------------------------------

@skip_dexined
@pytest.mark.slow
def test_dexined_output_shape_matches_input(detector, circle_gray):
    result = detector.run_dexined(circle_gray)
    assert result.shape == circle_gray.shape[:2]


@skip_dexined
@pytest.mark.slow
def test_dexined_binary_values(detector, circle_gray):
    result = detector.run_dexined(circle_gray)
    unique = set(np.unique(result))
    assert unique.issubset({0, 255}), f"Nečakané hodnoty pixelov: {unique}"


@skip_dexined
@pytest.mark.slow
def test_dexined_confidence_threshold_effect(detector, circle_gray):
    """Nižší confidence prah → viac detekovaných hrán."""
    low_conf = detector.run_dexined(circle_gray, confidence=0.1)
    high_conf = detector.run_dexined(circle_gray, confidence=0.9)
    assert np.count_nonzero(low_conf) >= np.count_nonzero(high_conf), (
        "confidence=0.1 by malo dávať aspoň toľko hrán ako confidence=0.9"
    )


# ---------------------------------------------------------------------------
# Test sťahovania váh (10) — nevyžaduje internet
# ---------------------------------------------------------------------------

def test_dexined_raises_when_weights_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keď ONNX model neexistuje, run_dexined musí vyhodiť FileNotFoundError.
    GUI zabezpečuje stiahnutie pred zavolaním run_dexined.
    """
    detector = EdgeDetector()
    fake_weights = tmp_path / "dexined.onnx"
    monkeypatch.setattr(EdgeDetector, "WEIGHTS_PATH", fake_weights)
    assert not fake_weights.exists()

    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="DexiNed"):
        detector.run_dexined(img)


def test_dexined_weights_path_is_onnx() -> None:
    """WEIGHTS_PATH musí ukazovať na .onnx súbor."""
    assert EdgeDetector.WEIGHTS_PATH.suffix == ".onnx"


def test_dexined_weights_url_points_to_huggingface() -> None:
    """WEIGHTS_URL musí ukazovať na HuggingFace."""
    assert "huggingface.co" in EdgeDetector.WEIGHTS_URL
