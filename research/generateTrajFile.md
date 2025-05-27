# Trajectory Generation Script

## Author

Ethan Ortiz

## Overview

This version of traj generation uses ORB to track camera motion and orientation.

This Python script processes a sequence of images captured by a single camera to estimate the camera's motion over time. It uses feature-based methods (ORB keypoints and descriptors), essential matrix estimation with RANSAC, and pose recovery to compute the relative transformation between consecutive frames. The resulting trajectory is saved as a text file, where each line represents a flattened 4×4 transformation matrix in SE(3) format.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Script Structure](#script-structure)
  - [`load_images(folder)`](#load_imagesfolder)
  - [`compute_pose(img1, img2, K)`](#computeposeimg1-img2-k)
  - [`generate_trajectory(image_folder, focal_length, dt)`](#generate_trajectoryimage_folder-focal_length-dt)
  - [`main(args)`](#mainargs)
- [Command-line Arguments](#command-line-arguments)
- [Example](#example)
- [License](#license)

## Requirements

- Python 3.6 or higher
- [OpenCV (cv2)](https://opencv.org/) (tested with OpenCV 4.x)
- [NumPy](https://numpy.org/)

## Installation

1. Clone the repository or copy the script into your project directory.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

```bash
python3 trajectory_generator.py \
  --image_folder /path/to/images \
  --output_file trajectory.txt \
  --focal_length 718.856 \
  [--dt 0.1]
```

- `--image_folder`: Path to the folder containing sequential images (`frame0001.png`, `frame0002.png`, etc.).
- `--output_file`: Path to the output text file where the trajectory will be saved.
- `--focal_length`: Camera focal length in pixels (used to construct the intrinsic matrix).
- `--dt`: (Optional) Time interval between frames in seconds (default: `0.1`).

## Script Structure

### `load_images(folder)`

```python
files = sorted(glob.glob(os.path.join(folder, 'frame*.png')))
if not files:
    files = sorted(glob.glob(os.path.join(folder, 'frame*.jpg')))
return files
```

- Scans the given folder for files starting with `frame` and ending in `.png` or `.jpg`.
- Returns a sorted list of file paths.

### `compute_pose(img1, img2, K)`

- Detects and computes ORB features and descriptors for two grayscale images.
- Matches features using a brute-force matcher with Hamming distance.
- Estimates the essential matrix via RANSAC and recovers the relative rotation (`R`) and translation (`t`).
- Returns `R` (3×3) and `t` (3×1) or `(None, None)` if estimation fails.

### `generate_trajectory(image_folder, focal_length, dt)`

- Loads all images and initializes the camera intrinsic matrix `K`.
- Iterates over consecutive image pairs to compute relative poses.
- Accumulates a global transformation matrix `T_total` (4×4) for each frame.
- Returns a list of 4×4 transformation matrices (one per frame).

### `main(args)`

- Parses command-line arguments.
- Calls `generate_trajectory` and writes the flattened trajectory matrices to the specified output file.

## Command-line Arguments

| Argument         | Type    | Description                                                             | Required | Default |
|------------------|---------|-------------------------------------------------------------------------|----------|---------|
| `--image_folder` | `str`   | Path to folder containing sequential images.                            | Yes      | N/A     |
| `--output_file`  | `str`   | Output file for the trajectory (e.g., `traj.txt`).                     | Yes      | N/A     |
| `--focal_length` | `float` | Camera focal length in pixels.                                          | Yes      | N/A     |
| `--dt`           | `float` | Time interval between frames in seconds (for timestamp simulation).     | No       | `0.1`   |

## Example

```bash
python3 trajectory_generator.py \
  --image_folder ./data/frames \
  --output_file ./results/trajectory.txt \
  --focal_length 700.0 \
  --dt 0.05
```