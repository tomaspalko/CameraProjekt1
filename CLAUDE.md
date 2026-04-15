# CLAUDE.md — CameraProjekt1

Living design document. Updated after each completed phase.

---

## Project Purpose

Desktop application for recognising the **position of a predefined object** in an image.
A reference image is configured with a ROI, detected edge segments, and an optional scale
calibration. During inspection, an incoming image is aligned to the reference using ECC and
the centroid shift (x, y) + rotation of the selected segments is reported.

---

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| GUI | PyQt6 |
| Image processing | OpenCV (`cv2`) |
| Deep learning | PyTorch — DexiNed edge detector |
| Profile storage | JSON + filesystem |
| Code formatting | ruff / standard PEP8 |
| UI theme | Dark mode (čierne pozadie, svetlý text) |
| Tests | pytest |

---

## Repository Layout

```
CameraProjekt1/
├── main.py                        # QApplication entry point
├── pyproject.toml                 # deps + black + pytest config
├── requirements.txt
├── .gitignore
├── CLAUDE.md                      # this file
├── core/
│   ├── profile_manager.py         # JSON CRUD, ID recycling, atomic writes
│   ├── edge_detection.py          # Canny + DexiNed wrappers
│   ├── segment_processor.py       # contour extraction, centroids, hit-test
│   ├── scale_calculator.py        # two-point px ↔ mm calibration
│   └── inspection_engine.py       # ECC matching, shift/rotation output
├── models/
│   └── dexined/
│       ├── model.py               # DexiNed PyTorch architecture
│       └── weights/               # auto-downloaded (~50 MB), git-ignored
├── ui/
│   ├── main_window.py             # QMainWindow, profile list, tab switching
│   ├── profile_tab.py             # Tab 1: configuration UI + state machine
│   ├── inspection_tab.py          # Tab 2: dual viewer + result panel
│   └── widgets/
│       ├── image_viewer.py        # zoom/pan, ROI rubber-band, overlays
│       ├── profile_list_widget.py # list + CRUD buttons
│       └── download_progress_dialog.py
├── profiles/                      # runtime data, git-ignored
│   └── <id>/
│       ├── profile.json
│       ├── reference.png
│       └── segment_map.png
└── tests/
    ├── conftest.py                # synthetic image fixtures
    ├── test_profile_manager.py
    ├── test_edge_detection.py
    ├── test_segment_processor.py
    ├── test_scale_calculator.py
    ├── test_inspection_engine.py
    └── test_integration.py
```

---

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
```

---

## UI Layout Overview

### Main Window

- Left pane: profile list (name = `Profile{id}`)
- Actions: New · Edit · Delete · Duplicate · Inspect
- Right pane: stacked — Profile Tab or Inspection Tab

### Tab 1 — Profile Configuration

| Left panel (~280 px) | Right panel |
|---|---|
| Image: Load / Delete | `ImageViewer` (zoom, pan, ROI, overlays) |
| ROI: Draw / Clear | |
| Edges: Show/Hide, Canny/DexiNed radio | |
| Canny T1/T2 sliders + spinboxes | |
| DexiNed confidence slider | |
| Min Segment Length slider | |
| Scale: Set / Clear / display value | |
| Undo | |
| Save Profile | |

### Tab 2 — Inspection

- Horizontal splitter: reference viewer (left) + inspection viewer (right)
- Coordinate label under each viewer (px and mm)
- Right panel: ECC params, ROI offset, Show/Hide segments, Run, results

---

## UI Theme — Dark Mode

Celá aplikácia používa **tmavý (čierny) vzhľad**. Implementované cez PyQt6 QSS (Qt Style Sheets):

```python
# main.py — aplikované pri štarte
app.setStyle("Fusion")
palette = QPalette()
palette.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
palette.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 20))
palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(40, 40, 40))
palette.setColor(QPalette.ColorRole.Button,          QColor(50, 50, 50))
palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
palette.setColor(QPalette.ColorRole.Highlight,       QColor(0, 120, 215))
palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
app.setPalette(palette)
```

Farebná paleta UI:

| Prvok | Farba |
|---|---|
| Pozadie okna | `#1e1e1e` |
| Pozadie widgetov | `#141414` |
| Text | `#dcdcdc` |
| Tlačidlá | `#323232` |
| Zvýraznenie / akcia | `#0078d7` |
| Reliability HIGH | `#00c853` (zelená) |
| Reliability MEDIUM | `#ffd600` (žltá) |
| Reliability LOW | `#d50000` (červená) |
| ROI obdĺžnik | `#0078d7` (modrý) |
| Vybraný segment | `#00e676` (zelená) |
| Nevybraný segment | `#757575` (sivá) |

---

## Development Commands

```bash
# Install dependencies
pip install -e .

# Run fast tests (no DexiNed weights needed)
pytest -m "not slow and not gui" -v

# Run DexiNed tests (downloads ~50 MB weights on first run)
DEXINED_TESTS=1 pytest -m "slow" -v

# Launch the application
python main.py
```

---

## GitHub Workflow

Repository: **tomaspalko/CameraProjekt1**

One commit per phase. PR groupings:
- **PR 1** — Core (Phases 1–5)
- **PR 2** — UI (Phases 6–8)
- **PR 3** — Polish (Phases 9–10)

---

## Phase Completion Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project scaffold + ProfileManager | ✅ done |
| 2 | Edge detection (Canny + DexiNed) | ✅ done |
| 3 | Segment processor | ✅ done |
| 4 | Scale calculator | ✅ done |
| 5 | Inspection engine (ECC) | ✅ done |
| 6 | Image viewer widget + Main window | ✅ done |
| 7 | Profile configuration tab (Tab 1) | ✅ done |
| 8 | Inspection tab (Tab 2) | ✅ done |
| 9 | Integration tests + end-to-end validation | ✅ done |
| 10 | DexiNed finalisation + download dialog + QA | ✅ done |

---

## Manual QA Checklist (Phase 10)

- [ ] Load real photo → draw ROI → run Canny → select 3 segments → save profile
- [ ] Load same photo as inspection → run ECC → verify shift ≈ 0 ± 0.5 px
- [ ] Shift photo 10 px → verify ~10 px shift reported
- [ ] First DexiNed use triggers download progress dialog
- [ ] Duplicate a profile, edit it — verify original unchanged
- [ ] Delete profile id=1, create new → verify id=1 is reused

---

*Last updated: Phase 1 — projekt scaffold, ProfileManager, GitHub repo vytvorený*
