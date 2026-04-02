# AnnotateEZ

A lightweight GUI tool for visualizing and annotating immunofluorescence (IF) microscopy cell images stored in HDF5 format.

## Features

- Tile-based grid view of single-cell images with configurable layout
- Multi-class annotation with customizable label names and colors
- Configurable RGB channel mapping (assign any channel to red, green, blue, or gray)
- Optional mask boundary overlay (binary or multi-cell instance masks)
- Annotations saved back to the source HDF5 file and exported to TSV

## Input Format

The tool expects an HDF5 file with the following datasets:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `images` | `(N, H, W, C)` | `uint16` | Multi-channel cell images |
| `features` | — | pandas DataFrame | Per-cell features (stored via `df.to_hdf`) |
| `channels` | `(C,)` | string | Channel names |
| `masks` *(optional)* | `(N, H, W)` or `(N, H, W, 1)` | int | Binary (0/1) or instance-segmentation masks |

## Installation

```bash
pip install annotate-ez
```

## Usage

```bash
annotate-ez
```

Or:

```bash
python -m annotateez
```

## Configuration

On first launch, a default configuration file is created at:

```
~/.config/annotate-ez/config.yml
```

## Requirements

- Python 3.8
- PyQt5 5.12.3
- h5py 3.1.0
- pandas 1.3.1
- numpy 1.21.1
- tables 3.8.0
- PyYAML 6.0.1

## License

MIT © Amin Naghdloo
