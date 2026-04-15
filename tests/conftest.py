"""
Zdieľané pytest fixtures pre celý testovací súbor.
Všetky testové obrázky sú generované synteticky — žiadne externé súbory nie sú potrebné.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.profile_manager import ProfileManager


@pytest.fixture
def tmp_profiles_dir(tmp_path: Path) -> Path:
    """Izolovaný dočasný adresár pre profily (každý test dostane vlastný)."""
    return tmp_path / "profiles"


@pytest.fixture
def profile_manager(tmp_profiles_dir: Path) -> ProfileManager:
    """ProfileManager pracujúci s izolovaným dočasným adresárom."""
    return ProfileManager(tmp_profiles_dir)


@pytest.fixture
def synthetic_gray_image() -> np.ndarray:
    """
    Syntetický 256×256 grayscale obrázok s dvoma sústredými kružnicami.
    Vhodný na testovanie detekcie hrán a segmentov.
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, 255, 2)
    cv2.circle(img, (128, 128), 40, 255, 2)
    return img


@pytest.fixture
def synthetic_color_image() -> np.ndarray:
    """
    Syntetický 256×256 BGR obrázok s kružnicou a obdĺžnikom.
    Vhodný na testovanie vstupu s 3 kanálmi.
    """
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(img, (128, 128), 80, (255, 255, 255), 2)
    cv2.rectangle(img, (40, 40), (215, 215), (200, 200, 200), 2)
    return img


@pytest.fixture
def synthetic_edge_map() -> np.ndarray:
    """
    Syntetická binárna mapa hrán (0/255) s jednou kružnicou.
    Vhodná na priame testovanie SegmentProcessor bez EdgeDetector.
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 255, 1)
    return img
