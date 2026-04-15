"""
ScaleCalculator — kalibrácia mierky pomocou dvoch bodov.

Pracovný postup:
  1. Užívateľ klikne na dva body v obrázku.
  2. Zadá reálnu vzdialenosť medzi nimi v mm.
  3. compute() vypočíta px_per_mm = euklidovská_vzdialenosť_px / vzdialenosť_mm.
  4. Následne sú dostupné konverzie px ↔ mm.

Body sú dragovateľné (move_point). Zobrazujú sa ako krížiky (riešenie vo vrstve UI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScalePoint:
    """Jeden kalibračný bod v súradniciach obrázka."""
    x: float
    y: float


class ScaleCalculator:
    """
    Výpočet mierky (px/mm) z dvoch bodov a zadanej vzdialenosti.

    Atribúty:
        point_a:    Prvý kalibračný bod (index 0).
        point_b:    Druhý kalibračný bod (index 1).
        px_per_mm:  Vypočítaný koeficient mierky (None kým sa nezavolá compute()).
    """

    def __init__(self) -> None:
        self.point_a: Optional[ScalePoint] = None
        self.point_b: Optional[ScalePoint] = None
        self.px_per_mm: Optional[float] = None

    # ------------------------------------------------------------------
    # Nastavenie bodov
    # ------------------------------------------------------------------

    def set_point(self, index: int, x: float, y: float) -> None:
        """
        Nastav kalibračný bod.

        Args:
            index: 0 = point_a, 1 = point_b.
            x, y:  Súradnice v pixeloch (full-image priestor).
        Raises:
            ValueError: ak index nie je 0 alebo 1.
        """
        if index == 0:
            self.point_a = ScalePoint(x=x, y=y)
        elif index == 1:
            self.point_b = ScalePoint(x=x, y=y)
        else:
            raise ValueError(f"Index bodu musí byť 0 alebo 1, dostal: {index}")

    def move_point(self, index: int, dx: float, dy: float) -> None:
        """
        Posunie existujúci bod o (dx, dy) — pre drag podporu v UI.

        Raises:
            ValueError: ak bod ešte nebol nastavený alebo index je neplatný.
        """
        if index == 0:
            if self.point_a is None:
                raise ValueError("point_a ešte nebol nastavený.")
            self.point_a.x += dx
            self.point_a.y += dy
        elif index == 1:
            if self.point_b is None:
                raise ValueError("point_b ešte nebol nastavený.")
            self.point_b.x += dx
            self.point_b.y += dy
        else:
            raise ValueError(f"Index bodu musí byť 0 alebo 1, dostal: {index}")

    # ------------------------------------------------------------------
    # Výpočet mierky
    # ------------------------------------------------------------------

    def compute(self, distance_mm: float) -> float:
        """
        Vypočíta px_per_mm z aktuálnych dvoch bodov a zadanej reálnej vzdialenosti.

        Args:
            distance_mm: Reálna vzdialenosť medzi bodmi v milimetroch (> 0).

        Returns:
            Vypočítaný koeficient px_per_mm.

        Raises:
            ValueError: ak point_a alebo point_b nie sú nastavené,
                        alebo ak distance_mm <= 0,
                        alebo ak sú body na rovnakom mieste.
        """
        if self.point_a is None:
            raise ValueError("point_a nie je nastavený.")
        if self.point_b is None:
            raise ValueError("point_b nie je nastavený.")
        if distance_mm <= 0:
            raise ValueError(f"Vzdialenosť musí byť > 0, dostal: {distance_mm}")

        dx = self.point_b.x - self.point_a.x
        dy = self.point_b.y - self.point_a.y
        dist_px = math.sqrt(dx * dx + dy * dy)

        if dist_px == 0:
            raise ValueError("Oba body sú na rovnakom mieste — vzdialenosť v px je 0.")

        self.px_per_mm = dist_px / distance_mm
        return self.px_per_mm

    # ------------------------------------------------------------------
    # Konverzné metódy
    # ------------------------------------------------------------------

    def pixel_to_mm(self, px: float) -> float:
        """
        Konvertuje vzdialenosť v pixeloch na milimetre.

        Raises:
            RuntimeError: ak px_per_mm ešte nie je nastavené (compute() nebol zavolaný).
        """
        self._require_calibrated()
        return px / self.px_per_mm  # type: ignore[operator]

    def mm_to_px(self, mm: float) -> float:
        """
        Konvertuje vzdialenosť v milimetroch na pixely.

        Raises:
            RuntimeError: ak px_per_mm ešte nie je nastavené.
        """
        self._require_calibrated()
        return mm * self.px_per_mm  # type: ignore[operator]

    def pixels_to_mm_point(
        self,
        px: float,
        py: float,
        origin_px: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """
        Konvertuje bod (px, py) na mm súradnice relatívne k origin_px.

        Args:
            px, py:     Bod v pixeloch (full-image priestor).
            origin_px:  Referenčný bod v pixeloch (napr. tažisko referenčného obrázka).

        Returns:
            (mm_x, mm_y) relatívne k origin.

        Raises:
            RuntimeError: ak px_per_mm ešte nie je nastavené.
        """
        self._require_calibrated()
        mm_x = (px - origin_px[0]) / self.px_per_mm  # type: ignore[operator]
        mm_y = (py - origin_px[1]) / self.px_per_mm  # type: ignore[operator]
        return (mm_x, mm_y)

    def reset(self) -> None:
        """Vymaže oba body a kalibrovanú mierku."""
        self.point_a = None
        self.point_b = None
        self.px_per_mm = None

    # ------------------------------------------------------------------
    # Interné
    # ------------------------------------------------------------------

    def _require_calibrated(self) -> None:
        if self.px_per_mm is None:
            raise RuntimeError(
                "Mierka ešte nie je kalibrovaná — najprv zavolaj compute()."
            )
