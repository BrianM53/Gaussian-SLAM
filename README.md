# Running the SLAM Pipeline (`streaming_to_SLAM.py`)

This script is used to start the SLAM pipeline.
The pipeline watches a target folder of `.jpg` frames from a video, starts the original Guassian-SLAM, and open a live viewer in the brower as the ouput generates.

---

# Setup

Before running the script, make sure your environment is correctly set up.  
Follow the instructions in [`setup.md`](./setup.md) for installing required packages and preparing your project.
Ensure that streaming is correctly configured to send frames to a target folder.

---

# Usage

Run the script from the command line:

```bash
python streaming_to_SLAM.py [options]
```

### Main Arguments

| Argument             | Type  | Default   | Description |
|:---------------------|:------|:----------|:------------|
| `--footage`           | str   | "Footage/Frames" | Path to the folder containing `.jpg` frame images. |
| `--space`             | int   | 5         | Number of digits in the filenames (e.g., `frame00001.jpg`). |
| `--scene_name`        | str   | None      | Optional label for the scene (useful for organizing output). |

---

### (Optional) Depth Map Parameters

| Argument             | Type  | Default   | Description |
|:---------------------|:------|:----------|:------------|
| `--depth_smoothing`   | bool  | True      | If `True`, apply temporal smoothing across frames for depth maps. |
| `--depth_scale`       | int   | 50        | Scalar to adjust brightness of depth maps. |
| `--depth_max`         | int   | 150       | Maximum value for linear scaling of depth. (Higher is closer to camera, set the limit less than 255 to avoid error points) |
| `--depth_min`         | int   | 0         | Minimum value for linear scaling of depth. (This will likely not need to be set more than 0, but it could be useful if depth generation is struggling with certain data.)|

You usually **do not need to modify** these unless you are tuning depth outputs for specific tests.

---

# 📋 Example Commands

1. **Basic run with default settings:**
```bash
python streaming_to_SLAM.py
```

2. **Process frames from a specific folder and assign a scene name:**
```bash
python streaming_to_SLAM.py --footage /path/to/frames --scene_name "Campus_Walkthrough"
```

3. **Disable depth smoothing and adjust depth scaling (if needed):**
```bash
python streaming_to_SLAM.py --depth_smoothing False --depth_scale 70 --depth_max 200 --depth_min 10
```

4. **Handle filenames with six digits (e.g., `frame000001.jpg`):**
```bash
python streaming_to_SLAM.py --space 6
```

---

# Need More Help?

- **Setup instructions:** See [`setup.md`](./setup.md)
- **Footage notes:** Make sure your `.jpg` files are numbered consistently with leading zeros based on the `--space` argument.
- **Conda Activate: ** Make sure that you are using the correct conda environment
```bash
conda activate gslam
```