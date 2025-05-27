# Two-Pass Affine Depth Map Scaling Script

This script performs a two-pass affine scaling (scale + offset) on a sequence of grayscale depth map images to minimize frame-to-frame differences while preventing darkening drift. It applies a forward pass (raw → forward-scaled) and a backward pass (forward-scaled → final) with mean anchoring and logs all parameters.

---

## Author

Ethan Ortiz

## Requirements

- Python 3
- [OpenCV](https://pypi.org/project/opencv-python/)
- NumPy

## Usage

```bash
python depth_smoothing.py <input_dir> <output_dir> [--pattern PATTERN] [--log_csv LOG_CSV]
```

- `<input_dir>`: Directory containing your raw depth map frames (e.g., `frames/raw`).
- `<output_dir>`: Directory where the final, two-pass scaled frames will be saved.
- `--pattern`: Glob pattern to match filenames (default: `*.png`).
- `--log_csv`: Name of the CSV file (saved in `<output_dir>`) to record forward/backward scales and offsets (default: `scale_factors.csv`).


### Example

```bash
python depth_smoothing.py data/2DVideo/results data/test/results --pattern "depth*" --log_csv depth_scales.csv
```

This reads all `depth*` files in `data/2DVideo/results`, processes them, and writes results to `data/test/results`. It logs parameters in `data/test/results/depth_scales.csv`.

---

## Script Behavior

1. **Forward Pass**
   - Anchors the first frame (scale=1.0, offset=0).
   - For each subsequent frame:
     - Computes optimal scale `s_f` and offset `b_f` to minimize
       \(\|\text{prev} - (s_f \cdot \text{curr} + b_f)\|^2\).
     - Clamps `s_f` to [0.9, 1.1] to prevent collapse.
     - Applies `curr * s_f + b_f` and saves to a temporary folder.

2. **Backward Pass**
   - Anchors the last forward-scaled frame, then mean-anchors it to match the very first frame’s mean.
   - Walks backward through the forward-scaled frames:
     - Computes backward scale `s_b` and offset `b_b` against the next (in video time) processed frame.
     - Applies affine transform, then shifts each frame’s mean to the reference mean.
     - Saves final frames to `<output_dir>`.

3. **Mean Anchoring**
   - After backward scaling each frame, the script shifts its pixel values so its global mean equals the first raw frame’s mean, ensuring no drift.

4. **Logging**
   - A CSV file (`--log_csv`) is written in `<output_dir>` with columns:
     - `filename`
     - `forward_scale`, `forward_offset`
     - `backward_scale`, `backward_offset`

---

## Tips & Troubleshooting

- **Adjust clamp range**: Modify `MIN_SCALE` and `MAX_SCALE` in the script to tighten or relax scaling limits.
- **Pattern matching**: Ensure your `--pattern` matches your file extensions (e.g., `*.png`, `frame_*.exr`).
