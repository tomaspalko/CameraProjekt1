"""
Testy pre core/profile_manager.py — 10 testových prípadov.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.profile_manager import ProfileManager, REQUIRED_KEYS, _default_profile


# ---------------------------------------------------------------------------
# 1. Prvý profil dostane ID = 1
# ---------------------------------------------------------------------------
def test_create_profile_assigns_id_1_first(profile_manager: ProfileManager) -> None:
    data = profile_manager.create_profile()
    assert data["id"] == 1


# ---------------------------------------------------------------------------
# 2. Druhý profil dostane ID = 2
# ---------------------------------------------------------------------------
def test_create_profile_ids_increment(profile_manager: ProfileManager) -> None:
    profile_manager.create_profile()
    data2 = profile_manager.create_profile()
    assert data2["id"] == 2


# ---------------------------------------------------------------------------
# 3. Po zmazaní ID=1 dostane nový profil opäť ID=1
# ---------------------------------------------------------------------------
def test_id_reuse_after_delete(profile_manager: ProfileManager) -> None:
    p1 = profile_manager.create_profile()
    profile_manager.create_profile()           # ID=2
    profile_manager.delete_profile(p1["id"])   # zmaž ID=1
    p_new = profile_manager.create_profile()
    assert p_new["id"] == 1


# ---------------------------------------------------------------------------
# 4. Uložený profil sa načíta identicky
# ---------------------------------------------------------------------------
def test_load_roundtrip(profile_manager: ProfileManager) -> None:
    data = profile_manager.create_profile(name="TestProfile")
    data["canny_params"]["threshold1"] = 99
    profile_manager.save_profile(data)
    loaded = profile_manager.load_profile(data["id"])
    assert loaded == data


# ---------------------------------------------------------------------------
# 5. Po zápise nesmie zostať .tmp súbor
# ---------------------------------------------------------------------------
def test_atomic_write_no_tmp_remaining(profile_manager: ProfileManager) -> None:
    data = profile_manager.create_profile()
    profile_dir = profile_manager._profile_dir(data["id"])
    tmp_files = list(profile_dir.glob("*.tmp"))
    assert tmp_files == [], f"Nečakané .tmp súbory: {tmp_files}"


# ---------------------------------------------------------------------------
# 6. Duplikát dostane nové ID
# ---------------------------------------------------------------------------
def test_duplicate_gets_new_id(profile_manager: ProfileManager) -> None:
    original = profile_manager.create_profile()
    duplicate = profile_manager.duplicate_profile(original["id"])
    assert duplicate["id"] != original["id"]
    assert duplicate["id"] == 2


# ---------------------------------------------------------------------------
# 7. Duplikát má skopírovaný adresár (ak existujú súbory)
# ---------------------------------------------------------------------------
def test_duplicate_copies_directory(
    profile_manager: ProfileManager, tmp_profiles_dir: Path
) -> None:
    original = profile_manager.create_profile()
    # Simulujeme reference.png v adresári profilu
    ref_path = profile_manager._profile_dir(original["id"]) / "reference.png"
    ref_path.write_bytes(b"PNG_PLACEHOLDER")

    duplicate = profile_manager.duplicate_profile(original["id"])
    dup_ref = profile_manager._profile_dir(duplicate["id"]) / "reference.png"
    assert dup_ref.exists(), "reference.png nebol skopírovaný do duplikátu"


# ---------------------------------------------------------------------------
# 8. list_profiles vracia profily zoradené podľa ID
# ---------------------------------------------------------------------------
def test_list_profiles_sorted_by_id(profile_manager: ProfileManager) -> None:
    profile_manager.create_profile()  # ID=1
    profile_manager.create_profile()  # ID=2
    profile_manager.create_profile()  # ID=3
    profile_manager.delete_profile(2)
    profile_manager.create_profile()  # ID=2 (recyklované)

    profiles = profile_manager.list_profiles()
    ids = [p["id"] for p in profiles]
    assert ids == sorted(ids), f"Profily nie sú zoradené: {ids}"


# ---------------------------------------------------------------------------
# 9. Po zmazaní profilu adresár neexistuje
# ---------------------------------------------------------------------------
def test_delete_removes_directory(profile_manager: ProfileManager) -> None:
    data = profile_manager.create_profile()
    profile_dir = profile_manager._profile_dir(data["id"])
    assert profile_dir.exists()
    profile_manager.delete_profile(data["id"])
    assert not profile_dir.exists()


# ---------------------------------------------------------------------------
# 10. validate_schema vyhodí ValueError pri chýbajúcom kľúči
# ---------------------------------------------------------------------------
def test_validate_schema_raises_on_missing_key(
    profile_manager: ProfileManager,
) -> None:
    data = _default_profile(1)
    del data["roi"]  # odstránime povinný kľúč
    with pytest.raises(ValueError, match="roi"):
        profile_manager.validate_schema(data)
