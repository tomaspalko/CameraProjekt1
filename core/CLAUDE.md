# core/ — Architecture & Data Contracts

## Profile JSON Schema

```json
{
  "id": 1,
  "name": "Profile1",
  "roi": {"x": 10, "y": 20, "w": 200, "h": 150},
  "edge_method": "canny",
  "canny_params": {"threshold1": 50, "threshold2": 150},
  "dexined_params": {"confidence": 0.5},
  "min_segment_length": 20,
  "scale_px_per_mm": 12.5,
  "centroid_ref": {"x": 145.3, "y": 210.7},
  "segment_indices": [0, 2, 5],
  "ecc_params": {
    "motion_type": "MOTION_EUCLIDEAN",
    "max_iter": 200,
    "epsilon": 1e-5
  },
  "roi_inspection_offset": {"dx": 0, "dy": 0},
  "roi_search_expansion": 0,
  "paths": {
    "reference_image": "profiles/1/reference.png",
    "segment_map": "profiles/1/segment_map.png"
  }
}
```

---

## Key Architectural Decisions

| Decision | Rationale |
|---|---|
| Segment coords always in full-image space | Single coordinate system across UI, engine, JSON |
| Segment map as ECC template (not raw image) | Robust to lighting; only structure participates |
| Atomic JSON write (`.tmp` → `os.replace`) | No corruption on crash |
| ID = lowest free positive integer | Scan `profiles/` dir, find `min(ℕ \ existing_ids)` |
| Undo = state snapshot list | Simpler than command pattern for small state |
| QThread for ECC inspection | `findTransformECC` can take 100–500 ms; keeps GUI responsive |
| `warp_matrix` + `motion_type` in `InspectionResult` | Umožňuje UI transformovať segmenty do inšpekčného priestoru bez opätovného behu ECC |

---

## Application Flow

### Profile Configuration (Tab 1)

```
Load image
  └─ Draw ROI
       └─ Toggle edge display (Canny or DexiNed)
            └─ Select segments (click / drag-deselect)
                 └─ [Optional] Set scale (2-point → distance in mm)
                      └─ Save profile
                           ├─ reference.png  (copy of loaded image)
                           ├─ segment_map.png (rendered contours)
                           └─ profile.json   (all parameters)
```

### Inspection (Tab 2)

```
Select profile → reference image + segment map shown
  └─ Load inspection image (must match reference dimensions)
       └─ Configure ECC params + ROI offset
            └─ Run Inspection
                 └─ Output: centroid shift (px + mm), rotation °,
                            NCC score, reliability, duration ms
                 └─ Overlay: vybrané segmenty (zelené) zobrazené na novej
                             pozícii v inšpekčnom obrázku (warp cez ECC maticu)
```

**Aligned segment overlay** (`ui/inspection_tab._build_aligned_overlay`):
1. Z `segment_map.png` extrahuje masku zelených pixelov (= vybrané segmenty).
2. Oreže na ROI oblasť referenčného priestoru.
3. Aplikuje `warp_matrix` z `InspectionResult` (`warpAffine` alebo `warpPerspective`).
4. Vloží výsledok na správnu pozíciu v inšpekčnom obrázku (ROI + offset).
5. Ak `warp_matrix` chýba alebo warp zlyhá — fallback na statickú `segment_map.png`.
