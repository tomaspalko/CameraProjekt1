"""
Testy pre core/scale_calculator.py — 9 testových prípadov.
"""

from __future__ import annotations

import math
import pytest

from core.scale_calculator import ScaleCalculator


@pytest.fixture
def calc() -> ScaleCalculator:
    return ScaleCalculator()


# ---------------------------------------------------------------------------
# 1. Horizontálna vzdialenosť 100 px / 10 mm = 10.0 px/mm
# ---------------------------------------------------------------------------

def test_compute_horizontal_100px_10mm(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 100.0, 0.0)
    result = calc.compute(10.0)
    assert abs(result - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Diagonálna vzdialenosť — musí použiť euklidovskú vzdialenosť
# ---------------------------------------------------------------------------

def test_compute_diagonal_uses_euclidean(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 30.0, 40.0)   # hypotenúza = 50 px
    result = calc.compute(5.0)       # 50 px / 5 mm = 10.0
    assert abs(result - 10.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. distance_mm = 0 vyvolá ValueError
# ---------------------------------------------------------------------------

def test_compute_raises_on_zero_distance_mm(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 100.0, 0.0)
    with pytest.raises(ValueError, match="0"):
        calc.compute(0.0)


# ---------------------------------------------------------------------------
# 4. Chýbajúci point_b vyvolá ValueError
# ---------------------------------------------------------------------------

def test_compute_raises_when_point_b_missing(calc):
    calc.set_point(0, 0.0, 0.0)
    with pytest.raises(ValueError, match="point_b"):
        calc.compute(10.0)


# ---------------------------------------------------------------------------
# 5. px → mm → px roundtrip
# ---------------------------------------------------------------------------

def test_pixel_to_mm_roundtrip(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 100.0, 0.0)
    calc.compute(10.0)  # px_per_mm = 10.0

    original_px = 37.5
    mm = calc.pixel_to_mm(original_px)
    back_px = calc.mm_to_px(mm)
    assert abs(back_px - original_px) < 1e-9


# ---------------------------------------------------------------------------
# 6. pixel_to_mm pred compute() vyvolá RuntimeError
# ---------------------------------------------------------------------------

def test_pixel_to_mm_raises_before_compute(calc):
    with pytest.raises(RuntimeError, match="kalibrovaná"):
        calc.pixel_to_mm(50.0)


# ---------------------------------------------------------------------------
# 7. move_point aktualizuje súradnice
# ---------------------------------------------------------------------------

def test_move_point_updates_coordinates(calc):
    calc.set_point(0, 10.0, 20.0)
    calc.move_point(0, 5.0, -3.0)
    assert abs(calc.point_a.x - 15.0) < 1e-9
    assert abs(calc.point_a.y - 17.0) < 1e-9


# ---------------------------------------------------------------------------
# 8. reset() vymaže všetko
# ---------------------------------------------------------------------------

def test_reset_clears_all(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 100.0, 0.0)
    calc.compute(10.0)
    calc.reset()
    assert calc.point_a is None
    assert calc.point_b is None
    assert calc.px_per_mm is None


# ---------------------------------------------------------------------------
# 9. pixels_to_mm_point s origin offsetom
# ---------------------------------------------------------------------------

def test_pixels_to_mm_point_with_origin(calc):
    calc.set_point(0, 0.0, 0.0)
    calc.set_point(1, 100.0, 0.0)
    calc.compute(10.0)  # px_per_mm = 10.0

    # Bod (120, 30), origin (20, 10) → delta px = (100, 20) → delta mm = (10.0, 2.0)
    mm_x, mm_y = calc.pixels_to_mm_point(120.0, 30.0, origin_px=(20.0, 10.0))
    assert abs(mm_x - 10.0) < 1e-9
    assert abs(mm_y - 2.0) < 1e-9
