# Depth Map Generation and Smoothing Script Documentation

## Author

- **Ethan Ortiz**

## Overview

This script provides functionality to generate depth maps from RGB images using the `LiheYoung/depth-anything-small-hf` model, downscale and upscale images for efficient processing, and apply temporal smoothing between consecutive frames via affine transformations.

## Table of Contents

1. [Features](#features)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Script Structure](#script-structure)
   - [Global Pipeline Initialization](#global-pipeline-initialization)
   - [Helper Functions](#helper-functions)
   - [Main Processing](#main-processing)
6. [Examples](#examples)
7. [Argument Reference](#argument-reference)
8. [Contributing](#contributing)
9. [License](#license)

## Features

- **Depth Estimation**: Generates depth maps using a Hugging Face transformer pipeline.
- **Downscaling/Upscaling**: Improves processing speed by resizing images.
- **Temporal Smoothing**: Applies affine warping and blending to smooth depth maps across frames.
- **CLI Interface**: Run the script via command-line arguments for batch or single-image processing. This is not how it is used in our full application

## Dependencies

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- Transformers (Hugging Face)

## Installation

```bash
# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install opencv-python numpy pillow tqdm transformers
```

## Usage

```bash
python3 depth_script.py \
  --left_img /path/to/image.png \
  --output_dir /path/to/output \
  [--use_smoothing] \
  [--prev_rgb /path/to/prev_rgb.png] \
  [--prev_depth /path/to/prev_depth.png] \
  [--min_depth <float>]  # default: 0 \
  [--max_depth <float>]  # default: 255
```

## Script Structure

### Global Pipeline Initialization

```python
# Load depth-estimation model once at startup
depth_pipeline = pipeline(
    task="depth-estimation",
    model="LiheYoung/depth-anything-small-hf"
)
```

### Helper Functions

- `downscale(img, factor=2)`: Resize input image by integer factor using area interpolation.
- `upscale(img, shape)`: Resize image back to original shape using linear interpolation.
- `depth_anything_generate(image_path)`: Creates a depth map using depth anything
  1. Loads and downsizes the image.
  2. Runs depth estimation pipeline.
  3. Normalizes and converts depth to `uint8`.
  4. Upscales to original resolution.
  5. Returns the depth map.
- `smooth_with_affine(prev_rgb, curr_rgb, prev_depth, curr_depth, alpha=0.7)`:
  1. Detects ORB features and matches between frames.
  2. Estimates an affine transform.
  3. Warps the previous depth map to the current frame.
  4. Blends warped and current depths by weight `alpha`.

### Main Processing

The `main` function:
1. Parses CLI arguments.
2. Ensures output directory exists.
3. Generates current depth map via `depth_anything_generate`.
4. If smoothing is enabled and previous frame data provided:
   - Calls `smooth_with_affine` to refine depth continuity.
5. Optionally remaps depth values to a specified range.
6. Inverts and writes the final depth map to disk.

## Examples

Generate a depth map without smoothing:
```bash
python3 depth_script.py \
  --left_img image01.png \
  --output_dir ./depths
```

Generate with smoothing using previous frame:
```bash
python3 depth_script.py \
  --left_img frame02.png \
  --output_dir ./depths \
  --use_smoothing \
  --prev_rgb frame01.png \
  --prev_depth ./depths/depth01.png
```

## Argument Reference

| Argument         | Type    | Required | Default | Description                                              |
|------------------|---------|----------|---------|----------------------------------------------------------|
| `--left_img`     | string  | Yes      | None    | Path to the input RGB image.                             |
| `--output_dir`   | string  | Yes      | None    | Directory for saving output depth maps.                  |
| `--use_smoothing`| flag    | No       | False   | Enable temporal smoothing between frames.                |
| `--prev_rgb`     | string  | No       | None    | Previous frame RGB image (requires smoothing).           |
| `--prev_depth`   | string  | No       | None    | Previous frame depth map (requires smoothing).           |
| `--min_depth`    | float   | No       | 0       | Minimum depth value after remapping.                     |
| `--max_depth`    | float   | No       | 255     | Maximum depth value after remapping.                     |

