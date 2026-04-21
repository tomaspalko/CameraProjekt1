"""
Testy pre core/segment_processor.py — 12 testových prípadov.
Všetky obrázky sú generované synteticky.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.segment_processor import Segment, SegmentProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def proc() -> SegmentProcessor:
    return SegmentProcessor()


@pytest.fixture
def circle_edge_map() -> np.ndarray:
    """256×256 binárna mapa s jednou kružnicou (polomer 60 px, stred 128,128)."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (128, 128), 60, 255, 1)
    return img


@pytest.fixture
def two_circles_edge_map() -> np.ndarray:
    """256×256 binárna mapa s dvoma kružnicami rôznych veľkostí."""
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(img, (64, 64), 40, 255, 1)    # menšia
    cv2.circle(img, (192, 192), 55, 255, 1)  # väčšia
    return img


@pytest.fixture
def tiny_dots_edge_map() -> np.ndarray:
    """256×256 mapa s niekoľkými veľmi malými kontúrami (dĺžka < 10 px)."""
    img = np.zeros((256, 256), dtype=np.uint8)
    # Jednotlivé body — kontúry dĺžky ~0
    for x in range(10, 50, 10):
        img[x, x] = 255
    return img


# ---------------------------------------------------------------------------
# 1. extract_segments nájde kružnicu
# ---------------------------------------------------------------------------

def test_extract_finds_circle_contour(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    assert len(segments) >= 1, "Aspoň jedna kontúra musí byť nájdená"


# ---------------------------------------------------------------------------
# 2. min_length filter odstráni krátke kontúry
# ---------------------------------------------------------------------------

def test_min_length_filter_removes_short(proc, tiny_dots_edge_map):
    segments = proc.extract_segments(tiny_dots_edge_map, min_length=50)
    assert len(segments) == 0, f"Krátke kontúry mali byť filtrované, zostalo: {len(segments)}"


# ---------------------------------------------------------------------------
# 3. Segmenty sú zoradené zostupne podľa arc_length
# ---------------------------------------------------------------------------

def test_segments_sorted_by_arc_length_desc(proc, two_circles_edge_map):
    segments = proc.extract_segments(two_circles_edge_map, min_length=10)
    assert len(segments) >= 2
    lengths = [s.arc_length for s in segments]
    assert lengths == sorted(lengths, reverse=True), f"Nie sú zoradené: {lengths}"


# ---------------------------------------------------------------------------
# 4. ROI offset je aplikovaný — súradnice sú v full-image priestore
# ---------------------------------------------------------------------------

def test_contour_coords_in_full_image_space(proc):
    """Kružnica nakreslená v hornej časti, ROI posunuté — súradnice musia byť odsadené."""
    img = np.zeros((256, 256), dtype=np.uint8)
    # Kružnica v ROI oblasti (50,50)+(100,100), stred kružnice pri (50,50)+(30,30)=(80,80)
    cv2.circle(img, (80, 80), 20, 255, 1)

    roi = (50, 50, 100, 100)  # x=50, y=50
    segments = proc.extract_segments(img, min_length=10, roi=roi)

    assert len(segments) >= 1
    # Všetky x súradnice kontúry musia byť >= roi_x
    for seg in segments:
        min_x = int(seg.contour[:, 0, 0].min())
        assert min_x >= roi[0], f"x súradnica {min_x} je menšia ako roi_x={roi[0]}"


# ---------------------------------------------------------------------------
# 5. Tažisko kružnice je blízko jej geometrického stredu
# ---------------------------------------------------------------------------

def test_centroid_on_circle_near_center(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    assert len(segments) >= 1

    cx, cy = segments[0].centroid
    # Stred kružnice je (128, 128) — tolerancia 5 px
    assert abs(cx - 128) < 5, f"cx={cx:.1f} ďaleko od 128"
    assert abs(cy - 128) < 5, f"cy={cy:.1f} ďaleko od 128"


# ---------------------------------------------------------------------------
# 6. compute_combined_centroid — jeden segment = jeho vlastné tažisko
# ---------------------------------------------------------------------------

def test_combined_centroid_single_segment(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    assert len(segments) >= 1

    seg = segments[0]
    combined = proc.compute_combined_centroid(segments, [seg.index])
    assert abs(combined[0] - seg.centroid[0]) < 1e-6
    assert abs(combined[1] - seg.centroid[1]) < 1e-6


# ---------------------------------------------------------------------------
# 7. compute_combined_centroid — vážený priemer dvoch segmentov
# ---------------------------------------------------------------------------

def test_combined_centroid_weighted(proc, two_circles_edge_map):
    segments = proc.extract_segments(two_circles_edge_map, min_length=10)
    assert len(segments) >= 2

    s0, s1 = segments[0], segments[1]
    total = s0.arc_length + s1.arc_length
    expected_cx = (s0.centroid[0] * s0.arc_length + s1.centroid[0] * s1.arc_length) / total
    expected_cy = (s0.centroid[1] * s0.arc_length + s1.centroid[1] * s1.arc_length) / total

    combined = proc.compute_combined_centroid(segments, [s0.index, s1.index])
    assert abs(combined[0] - expected_cx) < 1e-6, f"cx nesedí: {combined[0]:.4f} != {expected_cx:.4f}"
    assert abs(combined[1] - expected_cy) < 1e-6, f"cy nesedí: {combined[1]:.4f} != {expected_cy:.4f}"


# ---------------------------------------------------------------------------
# 8. render_segment_map — tvar výstupu zodpovedá image_shape
# ---------------------------------------------------------------------------

def test_render_segment_map_shape(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    result = proc.render_segment_map((256, 256), segments, [segments[0].index])
    assert result.shape == (256, 256, 3), f"Nesprávny tvar: {result.shape}"
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# 9. render_segment_map — vybraný segment je zelený
# ---------------------------------------------------------------------------

def test_render_selected_segment_is_green(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    assert len(segments) >= 1

    result = proc.render_segment_map((256, 256), segments, [segments[0].index])

    # Nájdi pixely kde zelená > 200 a modrá + červená < 50
    green_mask = (result[:, :, 1] > 200) & (result[:, :, 0] < 50) & (result[:, :, 2] < 50)
    green_count = int(np.sum(green_mask))
    assert green_count > 10, f"Príliš málo zelených pixelov: {green_count}"


# ---------------------------------------------------------------------------
# 10. hit_test_segment — klik ďaleko od všetkých segmentov vráti None
# ---------------------------------------------------------------------------

def test_hit_test_returns_none_when_far(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    # Kružnica je okolo (128,128), klikneme na (5, 5) — ďaleko
    result = proc.hit_test_segment(segments, (5, 5), tolerance_px=3)
    assert result is None


# ---------------------------------------------------------------------------
# 11. hit_test_segment — klik blízko kontúry vráti správny index
# ---------------------------------------------------------------------------

def test_hit_test_returns_correct_index(proc, circle_edge_map):
    segments = proc.extract_segments(circle_edge_map, min_length=10)
    assert len(segments) >= 1

    # Kružnica má polomer 60, stred 128 — bod (128, 68) je na obvode (128+0, 128-60)
    result = proc.hit_test_segment(segments, (128, 68), tolerance_px=5)
    assert result == segments[0].index


# ---------------------------------------------------------------------------
# 12. hit_test_area — rect pokrývajúci bounding_rect segmentu vráti jeho index
# ---------------------------------------------------------------------------

def test_hit_test_area_returns_overlapping(proc, two_circles_edge_map):
    segments = proc.extract_segments(two_circles_edge_map, min_length=10)
    assert len(segments) >= 2

    # Celý obrázok ako rect — musia byť nájdené všetky segmenty
    all_indices = proc.hit_test_area(segments, (0, 0, 256, 256))
    segment_indices = {s.index for s in segments}
    assert set(all_indices) == segment_indices


# ---------------------------------------------------------------------------
# fit_contour_to_line
# ---------------------------------------------------------------------------

def test_fit_contour_to_line_shape_and_dtype():
    rng = np.random.default_rng(42)
    xs = np.arange(10, 60, dtype=np.int32)
    ys = (100 + rng.integers(-2, 3, size=50)).astype(np.int32)
    contour = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    result = SegmentProcessor.fit_contour_to_line(contour)
    assert result is not None
    assert result.ndim == 3 and result.shape[1] == 1 and result.shape[2] == 2
    assert result.dtype == np.int32


def test_fit_contour_to_line_is_straight():
    rng = np.random.default_rng(7)
    xs = np.arange(20, 80, dtype=np.int32)
    ys = (50 + rng.integers(-3, 4, size=60)).astype(np.int32)
    contour = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    result = SegmentProcessor.fit_contour_to_line(contour)
    assert result is not None
    pts = result[:, 0, :].astype(float)
    fit2 = cv2.fitLine(result.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = fit2[0], fit2[1], fit2[2], fit2[3]
    dx, dy = pts[:, 0] - x0, pts[:, 1] - y0
    residuals = np.abs(dx * vy - dy * vx)
    assert residuals.max() < 0.6


def test_fit_contour_to_line_single_point_returns_none():
    single = np.array([[[10, 20]]], dtype=np.int32)
    assert SegmentProcessor.fit_contour_to_line(single) is None


def test_fit_contour_to_line_extent_coverage():
    xs = np.arange(5, 55, dtype=np.int32)
    ys = np.full(50, 30, dtype=np.int32)
    contour = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    result = SegmentProcessor.fit_contour_to_line(contour)
    assert result is not None
    result_span = int(result[:, 0, 0].max()) - int(result[:, 0, 0].min())
    assert result_span >= 0.8 * (55 - 5)
