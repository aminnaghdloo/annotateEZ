# AnnotateEZ

A lightweight GUI tool for visualizing and annotating immunofluorescence (IF) microscopy cell images stored in HDF5 format.

## Features

- Tile-based grid view of single-cell images with configurable layout
- Multi-class annotation with customizable label names and colors
- Configurable RGB channel mapping with per-channel gain control
- Optional mask boundary overlay (binary or multi-cell instance masks)
- Click-and-drag annotation for fast bulk labeling
- Per-page lazy image loading with background prefetch — handles large datasets without loading everything into memory
- Sort events by any DataFrame column without modifying the underlying data
- Undo / redo with configurable depth
- Dark and light themes
- Full keyboard shortcut support
- Annotations saved back to the source HDF5 file and exported to TSV

## Input Format

The tool expects an HDF5 file with the following datasets:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `images` | `(N, H, W, C)` | `uint16` | Multi-channel cell images |
| `features` | — | pandas DataFrame | Per-cell features (stored via `df.to_hdf`) |
| `channels` | `(C,)` | string | Channel names |
| `masks` *(optional)* | `(N, H, W)` or `(N, H, W, 1)` | int | Binary or instance-segmentation masks |

## Installation

```bash
pip install annotate-ez
```

## Usage

```bash
annotate-ez
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `→` / `←` | Next / previous page |
| `Alt+0` – `Alt+6` | Switch active label |
| `0` | Switch to RGB composite view |
| `1` – `9` | Switch to single-channel grayscale view (by channel index) |
| `Ctrl+S` | Save annotations |
| `Ctrl+Z` | Undo last annotation change |
| `Ctrl+Y` | Redo |

## Annotation Tips

**Single-click annotation**
Left-click any tile to assign the currently selected label. Right-click to reset it to the background class (label 0).

**Click-and-drag annotation**
Hold the mouse button and drag across tiles to annotate multiple cells in one motion. The entire drag gesture counts as a single undo step, so you can reverse it with one `Ctrl+Z`.

**Batch labeling**
Use the **All** button to assign the active label to every tile on the current page, or **None** to reset the entire page to background. Both actions are undoable.

**Sorting for faster review**
Use the sort panel to reorder cells by any feature column (e.g., a confidence score or size metric) before annotating. Sort order is display-only — the saved DataFrame is never reordered.

**Undo depth**
The default undo history keeps the last 3 steps. Increase `max undo steps` in Settings if you want a deeper history, keeping in mind that each step stores a full copy of all labels.

## Configuration

On first launch, a default configuration file is created at:

```
~/.config/annotate-ez/config.yml
```

All settings are also editable via the **Settings** dialog inside the app. Key options:

| Setting | Description |
|---------|-------------|
| `theme` | `dark` or `light` |
| `tile_size` | Pixel size of each image tile (always kept odd) |
| `x_size` / `y_size` | Number of columns / rows in the tile grid |
| `max_undo_steps` | Maximum number of undo steps stored in memory |
| `image_key` / `data_key` | HDF5 dataset keys for images and features |
| `output_dir` | Directory where the TSV export is written |

## Requirements

- Python 3.8 – 3.9 (tested)
- PyQt5 ≥ 5.15
- h5py ≥ 3.1
- pandas ≥ 1.3
- numpy ≥ 1.21
- tables ≥ 3.8
- PyYAML ≥ 6.0

## License

MIT © Amin Naghdloo
