"""
ProfileManager — CRUD operácie pre profily uložené ako JSON + súbory.

Každý profil má vlastný podadresár:  profiles/<id>/
  - profile.json    (konfigurácia)
  - reference.png   (referenčná fotka, kopírovaná pri uložení)
  - segment_map.png (mapa segmentov, generovaná pri uložení)

Zápis je atomický: najprv .tmp súbor, potom os.replace().
ID = najnižšie voľné kladné celé číslo.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

# Povinné kľúče v každom profile
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "name",
        "roi",
        "edge_method",
        "canny_params",
        "dexined_params",
        "min_segment_length",
        "scale_px_per_mm",
        "centroid_ref",
        "segment_indices",
        "ecc_params",
        "roi_inspection_offset",
        "paths",
    }
)


def _default_profile(profile_id: int) -> dict:
    """Vráti predvolený slovník profilu pre dané ID."""
    return {
        "id": profile_id,
        "name": f"Profile{profile_id}",
        "roi": {"x": 0, "y": 0, "w": 0, "h": 0},
        "edge_method": "canny",
        "canny_params": {"threshold1": 50, "threshold2": 150},
        "dexined_params": {"confidence": 0.5},
        "min_segment_length": 20,
        "scale_px_per_mm": None,
        "centroid_ref": {"x": 0.0, "y": 0.0},
        "segment_indices": [],
        "ecc_params": {
            "motion_type": "MOTION_EUCLIDEAN",
            "max_iter": 200,
            "epsilon": 1e-5,
        },
        "roi_inspection_offset": {"dx": 0, "dy": 0},
        "paths": {
            "reference_image": f"profiles/{profile_id}/reference.png",
            "segment_map": f"profiles/{profile_id}/segment_map.png",
        },
    }


class ProfileManager:
    """Správa profilov (vytvorenie, načítanie, uloženie, zmazanie, duplikácia)."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Interné pomocné metódy
    # ------------------------------------------------------------------

    def _next_free_id(self) -> int:
        """Vráti najnižšie voľné kladné celé číslo pre nový profil."""
        existing: set[int] = set()
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                existing.add(int(entry.name))
        n = 1
        while n in existing:
            n += 1
        return n

    def _profile_dir(self, profile_id: int) -> Path:
        return self.base_dir / str(profile_id)

    def _json_path(self, profile_id: int) -> Path:
        return self._profile_dir(profile_id) / "profile.json"

    # ------------------------------------------------------------------
    # Verejné CRUD metódy
    # ------------------------------------------------------------------

    def create_profile(self, name: Optional[str] = None) -> dict:
        """
        Vytvorí nový profil s najnižším voľným ID.
        Vráti nový slovník profilu.
        """
        profile_id = self._next_free_id()
        profile_dir = self._profile_dir(profile_id)
        profile_dir.mkdir(parents=True, exist_ok=True)
        data = _default_profile(profile_id)
        if name is not None:
            data["name"] = name
        self.save_profile(data)
        return data

    def load_profile(self, profile_id: int) -> dict:
        """
        Načíta profil podľa ID zo súboru profile.json.
        Raises FileNotFoundError ak profil neexistuje.
        Raises ValueError ak schéma nie je platná.
        """
        json_path = self._json_path(profile_id)
        if not json_path.exists():
            raise FileNotFoundError(f"Profil {profile_id} nebol nájdený.")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.validate_schema(data)
        return data

    def save_profile(self, data: dict) -> None:
        """
        Uloží profil atomicky (.tmp → os.replace).
        Raises ValueError ak schéma nie je platná.
        """
        self.validate_schema(data)
        profile_id = data["id"]
        profile_dir = self._profile_dir(profile_id)
        profile_dir.mkdir(parents=True, exist_ok=True)
        json_path = self._json_path(profile_id)
        tmp_path = json_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, json_path)
        except Exception:
            # Upratíme .tmp súbor pri chybe
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def delete_profile(self, profile_id: int) -> None:
        """
        Zmaže celý adresár profilu vrátane všetkých súborov.
        Raises FileNotFoundError ak profil neexistuje.
        """
        profile_dir = self._profile_dir(profile_id)
        if not profile_dir.exists():
            raise FileNotFoundError(f"Profil {profile_id} nebol nájdený.")
        shutil.rmtree(profile_dir)

    def duplicate_profile(self, profile_id: int) -> dict:
        """
        Zduplikuje existujúci profil (vrátane súborov) pod novým ID.
        Vráti nový slovník profilu.
        Raises FileNotFoundError ak zdrojový profil neexistuje.
        """
        source_dir = self._profile_dir(profile_id)
        if not source_dir.exists():
            raise FileNotFoundError(f"Profil {profile_id} nebol nájdený.")

        new_id = self._next_free_id()
        new_dir = self._profile_dir(new_id)
        shutil.copytree(source_dir, new_dir)

        # Aktualizujeme ID, name a paths v skopírovanom JSON
        new_json_path = new_dir / "profile.json"
        with open(new_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["id"] = new_id
        data["name"] = f"Profile{new_id}"
        data["paths"]["reference_image"] = str(new_dir / "reference.png")
        data["paths"]["segment_map"] = str(new_dir / "segment_map.png")

        tmp_path = new_json_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, new_json_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return data

    def list_profiles(self) -> list[dict]:
        """
        Vráti zoznam všetkých profilov zoradených podľa ID (vzostupne).
        Profily s chybnou schémou sú preskočené.
        """
        profiles: list[dict] = []
        for entry in self.base_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                try:
                    profiles.append(self.load_profile(int(entry.name)))
                except (FileNotFoundError, ValueError):
                    pass
        profiles.sort(key=lambda d: d["id"])
        return profiles

    def validate_schema(self, data: dict) -> None:
        """
        Overí, že slovník obsahuje všetky povinné kľúče.
        Raises ValueError pri chýbajúcom kľúči.
        """
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Profil má chýbajúce kľúče: {', '.join(sorted(missing))}"
            )
