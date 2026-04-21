# ui/ — Layout & Theme

## UI Layout

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

## Dark Mode Theme

Aplikované pri štarte cez `app.setStyle("Fusion")` + `QPalette`:

```python
palette.setColor(QPalette.ColorRole.Window,          QColor(30, 30, 30))
palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
palette.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 20))
palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(40, 40, 40))
palette.setColor(QPalette.ColorRole.Button,          QColor(50, 50, 50))
palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
palette.setColor(QPalette.ColorRole.Highlight,       QColor(0, 120, 215))
palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
```

### Color Reference

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
