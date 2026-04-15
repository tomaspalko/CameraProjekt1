"""
Integracne testy — 8 testovacich pripadov (headless, synteticke obrazky).

Testuju spolupracu viacerych komponentov bez GUI a bez DexiNed vahy.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.edge_detection import EdgeDetector
from core.inspection_engine import InspectionEngine
from core.profile_manager import ProfileManager
from core.scale_calculator import ScaleCalculator
from core.segment_processor import SegmentProcessor


# ---------------------------------------------------------------------------
# Pomocne funkcie
# ---------------------------------------------------------------------------

def _circle_image(h: int = 256, w: int = 256, radius: int = 60) -> np.ndarray:
    """BGR obrazok s obrysom kruznice (hrubka 3)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), radius, (200, 200, 200), 3)
    return img


def _shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def _make_profile(pm: ProfileManager, img: np.ndarray) -> dict:
    """Vytvori a ulozi jednoduchy profil s Canny detekciou na celu plochu."""
    profile = pm.create_profile()
    pid = profile["id"]

    detector = EdgeDetector()
    processor = SegmentProcessor()

    edge_map = detector.run_canny(img, threshold1=30, threshold2=100)
    roi = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
    segments = processor.extract_segments(edge_map, min_length=10, roi=roi)

    selected = [s.index for s in segments[:1]] if segments else []
    centroid = (
        processor.compute_combined_centroid(segments, selected)
        if selected
        else (img.shape[1] / 2.0, img.shape[0] / 2.0)
    )

    profile_dir = Path(pm.base_dir) / str(pid)
    profile_dir.mkdir(parents=True, exist_ok=True)
    ref_path = str(profile_dir / "reference.png")
    seg_path = str(profile_dir / "segment_map.png")

    cv2.imwrite(ref_path, img)
    seg_map = processor.render_segment_map(img.shape[:2], segments, selected)
    cv2.imwrite(seg_path, seg_map)

    profile.update({
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "edge_method": "canny",
        "canny_params": {"threshold1": 30, "threshold2": 100},
        "dexined_params": {"confidence": 0.5},
        "min_segment_length": 10,
        "scale_px_per_mm": None,
        "centroid_ref": {"x": centroid[0], "y": centroid[1]},
        "segment_indices": selected,
        "ecc_params": {
            "motion_type": "MOTION_EUCLIDEAN",
            "max_iter": 200,
            "epsilon": 1e-5,
        },
        "roi_inspection_offset": {"dx": 0, "dy": 0},
        "paths": {
            "reference_image": ref_path,
            "segment_map": seg_path,
        },
    })
    pm.save_profile(profile)
    return profile


# ---------------------------------------------------------------------------
# 1. Profil create + save -> reload verify
# ---------------------------------------------------------------------------

def test_create_save_reload_profile(tmp_path):
    """Ulozeny profil sa zhoduje s opat nacitanym."""
    pm = ProfileManager(tmp_path / "profiles")
    img = _circle_image()
    profile = _make_profile(pm, img)
    pid = profile["id"]

    reloaded = pm.load_profile(pid)

    assert reloaded["id"] == pid
    assert reloaded["edge_method"] == "canny"
    assert reloaded["roi"]["w"] > 0
    assert Path(reloaded["paths"]["reference_image"]).exists()
    assert Path(reloaded["paths"]["segment_map"]).exists()


# ---------------------------------------------------------------------------
# 2. Plny inspekcia workflow: identicky obrazok -> posun ~0 px
# ---------------------------------------------------------------------------

def test_inspection_workflow_zero_shift(tmp_path):
    """Identicky inspekcia == referencny: posun musi byt < 1.5 px."""
    pm = ProfileManager(tmp_path / "profiles")
    img = _circle_image()
    profile = _make_profile(pm, img)

    ref_img = cv2.imread(profile["paths"]["reference_image"])
    seg_map = cv2.imread(profile["paths"]["segment_map"])
    c = profile["centroid_ref"]
    roi_d = profile["roi"]
    roi = (roi_d["x"], roi_d["y"], roi_d["w"], roi_d["h"])
    centroid_ref = (c["x"], c["y"])

    engine = InspectionEngine()
    result = engine.run(
        ref_img, seg_map, img.copy(), centroid_ref, roi,
        ecc_params={"motion_type": "MOTION_TRANSLATION", "max_iter": 500, "epsilon": 1e-6},
    )

    assert abs(result.shift_px[0]) < 1.5, f"dx={result.shift_px[0]:.3f}"
    assert abs(result.shift_px[1]) < 1.5, f"dy={result.shift_px[1]:.3f}"


# ---------------------------------------------------------------------------
# 3. Duplikacia profilu -> inspekcia na duplikate prebehne bez chyby
# ---------------------------------------------------------------------------

def test_duplicate_profile_inspect(tmp_path):
    """Duplikovany profil ma novy ID a je pouzitelny pre inspekciu."""
    pm = ProfileManager(tmp_path / "profiles")
    img = _circle_image()
    original = _make_profile(pm, img)

    dup = pm.duplicate_profile(original["id"])
    assert dup["id"] != original["id"]

    # Duplikat musi mat vlastne subory
    dup_reloaded = pm.load_profile(dup["id"])
    assert Path(dup_reloaded["paths"]["reference_image"]).exists()
    assert Path(dup_reloaded["paths"]["segment_map"]).exists()

    # Inspekcia na duplikate
    ref_img = cv2.imread(dup_reloaded["paths"]["reference_image"])
    seg_map = cv2.imread(dup_reloaded["paths"]["segment_map"])
    c = dup_reloaded.get("centroid_ref", {"x": 128.0, "y": 128.0})
    roi_d = dup_reloaded["roi"]
    roi = (roi_d["x"], roi_d["y"], roi_d["w"], roi_d["h"])

    engine = InspectionEngine()
    result = engine.run(
        ref_img, seg_map, ref_img.copy(), (c["x"], c["y"]), roi,
        ecc_params={"motion_type": "MOTION_TRANSLATION", "max_iter": 200, "epsilon": 1e-5},
    )
    assert result.duration_ms > 0


# ---------------------------------------------------------------------------
# 4. Canny pipeline na kruznicy: centroid blizko stredu
# ---------------------------------------------------------------------------

def test_canny_pipeline_centroid_near_center(tmp_path):
    """Canny + SegmentProcessor na kruznici: centroid do 5 px od stredu (128,128)."""
    img = _circle_image()
    detector = EdgeDetector()
    processor = SegmentProcessor()

    edge_map = detector.run_canny(img, threshold1=30, threshold2=100)
    roi = (10, 10, 236, 236)
    segments = processor.extract_segments(edge_map, min_length=10, roi=roi)

    assert len(segments) >= 1, "Aspon jeden segment musi byt najdeny"

    centroid = processor.compute_combined_centroid(segments, [segments[0].index])
    assert abs(centroid[0] - 128) < 5, f"cx={centroid[0]:.1f}"
    assert abs(centroid[1] - 128) < 5, f"cy={centroid[1]:.1f}"


# ---------------------------------------------------------------------------
# 5. Poradie profilov po delete + create: ID recycling
# ---------------------------------------------------------------------------

def test_profile_list_order_id_recycling(tmp_path):
    """Po zmazani profilu id=1 a vytvoreni noveho: novy dostane id=1."""
    pm = ProfileManager(tmp_path / "profiles")

    p1 = pm.create_profile()
    p2 = pm.create_profile()
    assert p1["id"] == 1
    assert p2["id"] == 2

    pm.delete_profile(1)

    p3 = pm.create_profile()
    assert p3["id"] == 1, f"Ocakaval id=1, dostal {p3['id']}"

    profiles = pm.list_profiles()
    ids = [p["id"] for p in profiles]
    assert ids == sorted(ids), "Profily musia byt zoradene podla id"


# ---------------------------------------------------------------------------
# 6. Kalibracia mierky: mm vysledky spravne
# ---------------------------------------------------------------------------

def test_scale_calibration_mm_results(tmp_path):
    """px_per_mm = 10.0 -> shift 5 px = 0.5 mm."""
    pm = ProfileManager(tmp_path / "profiles")
    img = _circle_image()
    profile = _make_profile(pm, img)

    # Nastav mierku
    profile["scale_px_per_mm"] = 10.0
    pm.save_profile(profile)
    reloaded = pm.load_profile(profile["id"])
    assert abs(reloaded["scale_px_per_mm"] - 10.0) < 1e-9

    # Inspekcia s posunutym obrazkom
    ref_img = cv2.imread(profile["paths"]["reference_image"])
    seg_map = cv2.imread(profile["paths"]["segment_map"])
    shifted = _shifted(img, 5.0, 0.0)

    c = profile["centroid_ref"]
    roi_d = profile["roi"]
    roi = (roi_d["x"], roi_d["y"], roi_d["w"], roi_d["h"])

    engine = InspectionEngine()
    result = engine.run(
        ref_img, seg_map, shifted, (c["x"], c["y"]), roi,
        ecc_params={"motion_type": "MOTION_TRANSLATION", "max_iter": 500, "epsilon": 1e-6},
        px_per_mm=10.0,
    )

    assert result.shift_mm is not None
    # posun ~5 px / 10 px_per_mm = ~0.5 mm, tolerancia 0.3 mm
    assert abs(result.shift_mm[0] - 0.5) < 0.3, f"shift_mm[0]={result.shift_mm[0]:.3f}"


# ---------------------------------------------------------------------------
# 7. Cisty posun: NCC >= 0.8, reliability == "HIGH"
# ---------------------------------------------------------------------------

def test_clean_shift_high_reliability(tmp_path):
    """Identicky obrazok -> NCC >= 0.8 a reliability == HIGH."""
    pm = ProfileManager(tmp_path / "profiles")
    img = _circle_image()
    profile = _make_profile(pm, img)

    ref_img = cv2.imread(profile["paths"]["reference_image"])
    seg_map = cv2.imread(profile["paths"]["segment_map"])
    c = profile["centroid_ref"]
    roi_d = profile["roi"]
    roi = (roi_d["x"], roi_d["y"], roi_d["w"], roi_d["h"])

    engine = InspectionEngine()
    result = engine.run(
        ref_img, seg_map, img.copy(), (c["x"], c["y"]), roi,
        ecc_params={"motion_type": "MOTION_TRANSLATION", "max_iter": 500, "epsilon": 1e-6},
    )

    assert result.ncc_score >= 0.8, f"NCC={result.ncc_score:.4f}"
    assert result.reliability == "HIGH", f"Reliability={result.reliability}"


# ---------------------------------------------------------------------------
# 8. Canny vs. DexiNed: rozlicne edge mapy (bez DexiNed vahy -> skip)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_canny_vs_dexined_different_maps():
    """Canny a DexiNed produkuju rozne hrany (potrebuje DexiNed vahy)."""
    import os
    if not os.environ.get("DEXINED_TESTS"):
        pytest.skip("DEXINED_TESTS nie je nastaveny")

    img = _circle_image()
    detector = EdgeDetector()

    canny_map = detector.run_canny(img, threshold1=50, threshold2=150)
    dexined_map = detector.run_dexined(img, confidence=0.5)

    # Musia mat rovnaky tvar
    assert canny_map.shape == dexined_map.shape

    # Ale rozlicny obsah (nie identicky)
    diff = np.abs(canny_map.astype(int) - dexined_map.astype(int))
    assert diff.sum() > 0, "Canny a DexiNed nesmú byť identické"
