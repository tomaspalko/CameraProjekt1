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
# DexiNed testy (7–9) — vyžadujú stiahnuté váhy a torch
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
# Test sťahovania váh (10) — mockuje urllib, nevyžaduje internet
# ---------------------------------------------------------------------------

def test_weight_download_called_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keď váhy neexistujú, _ensure_weights musí zavolať urlretrieve.
    Skutočné sťahovanie je mockované — test nepotrebuje internet.
    """
    detector = EdgeDetector()

    # Presmeruj WEIGHTS_PATH na dočasný adresár
    fake_weights = tmp_path / "dexined.pth"
    monkeypatch.setattr(EdgeDetector, "WEIGHTS_PATH", fake_weights)
    assert not fake_weights.exists()

    calls: list[str] = []

    def fake_urlretrieve(url: str, dest: str) -> None:
        calls.append(url)
        # Vytvor falošný súbor > 1 MB aby prešla kontrola veľkosti
        Path(dest).write_bytes(b"X" * 1_100_000)

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        detector._ensure_weights()

    assert len(calls) == 1, "urlretrieve malo byť zavolané presne raz"
    assert calls[0] == EdgeDetector.WEIGHTS_URL
    assert fake_weights.exists(), "Súbor váh nebol vytvorený"
