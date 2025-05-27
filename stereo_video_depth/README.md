# Stereo Depth Estimation from Video

## Author

Ethan Ortiz

## Overview

This script computes disparity and depth maps from a stereo video file using
OpenCV's StereoSGBM algorithm. It takes a stereo video (side-by-side or top-
bottom format) as input and generates depth maps for each frame, saving them
as both PNG images and NumPy arrays.

Optionally, you can input a left and a right image file to create one depth map at a time.

## Requirements

Before running the script, ensure you have the following dependencies
installed:

## Usage

Run the script from the command line with the following arguments:

### Arguments

Argument| Type| Default| Description  
---|---|---|---  
`--video`| string|  Path to the stereo video file (.mov).  
`--left_img`| string|  Path to the left image file with 5 numbers at the end of the name (.png).  
`--right_img`| string| Path to the right image file with 5 numbers at the end of the name (.png).  
`--output_dir`| string| `output`| Directory where depth maps will be saved.  
`--arrangement`| string| `side-by-side`| Stereo image arrangement in the video. Choose between `side-by-side` or `top-bottom`.  
`--focal_length`| float| `35`| Camera focal length in pixels.  
`--baseline`| float| `500`| Distance between the stereo cameras (same unit as depth output).  
`--ndisp`| int| `10`| Disparity range for depth computation.  
`--blockSize`| int| `4`| Block size for SGBM computation.  
`--uniquenessRatio`| int| `1`| Uniqueness ratio for disparity computation.  
`--disp12MaxDiff`| int| `100`| Maximum allowed difference in disparity calculation.  
  
### Example Usage

    python stereo_depth_from_video.py --video input\stereo_footage\test_video_1.mov --output_dir output/mov 

This will:

  1. Process `test_video_1.mov`

  2. Assume it is a side-by-side stereo video

  3. Compute depth maps for each frame

  4. Save depth images as PNGs and NumPy arrays in `output/mov/`

## Output Files

Each frame generates:

  * A depth map image (`depth_00001.png`, `depth_00002.png`, etc.)

  * A NumPy depth map (`depth_00001.npy`, `depth_00002.npy`, etc.)

## Troubleshooting

### 1\. Video Not Opening

**Error:** `Cannot open video file`

  * Ensure the file path is correct.

  * Try converting the video to a supported format using FFmpeg:
    ```
    ffmpeg -i input.mov -vcodec libx264 output.mp4
    ```


### 2\. OpenCV Error: Invalid Number of Channels

**Error:** `Invalid number of channels in input image`

  * Ensure the video is in a supported format (BGR or grayscale).

  * Try modifying the frame extraction to check the number of channels before processing:
    ```
    print(left_img.shape, right_img.shape)
    ```


## Notes

  * The script assumes stereo videos are either **side-by-side** or **top-bottom**.

  * The depth computation uses OpenCV's WLS filter for better results.

  * The baseline and focal length should be adjusted based on the camera setup.

# **Stereo Video Frame Extractor**

This script extracts left and right frames from a stereo `.mov` video file. It
assumes the video is arranged either **side-by-side** or **top-bottom** and
splits each frame accordingly, saving the images as PNG files.

## **Usage**
    
    python extract_frames.py input.mov left_frames/ right_frames/ --arrangement side-by-side
        

## **Arguments**

  * `input_file` (str) - Path to the input `.mov` file.

  * `left_folder` (str) - Folder where left frames will be saved.

  * `right_folder` (str) - Folder where right frames will be saved.

  * `--arrangement` (optional) - Specifies the stereo layout in the video.

    * Options: `side-by-side` (default) or `top-bottom`.

## **How It Works**

  1. The script opens the input `.mov` video.

  2. It reads frames one by one and checks the **arrangement** :

     * **Side-by-side** : The frame is split vertically at the midpoint.

     * **Top-bottom** : The frame is split horizontally at the midpoint.

  3. The left and right images are saved as `left_XXXXX.png` and `right_XXXXX.png`, where `XXXXX` is the frame number.

  4. It prints progress updates every 100 frames.

  5. The process continues until all frames are extracted.

## **Example**

Extract frames from a stereo video where the left and right images are stacked
**top-bottom** :

    
    
    python extract_frames.py stereo_video.mov output/left output/right --arrangement top-bottom
        

## **Output Structure**

If your video has 300 frames and you extract with:
    
    python extract_frames.py input.mov left_frames/ right_frames/
        

Your output folders will contain:
    
    left_frames/
          ├── left_00000.png
          ├── left_00001.png
          ├── ...
          ├── left_00299.png
        
    right_frames/
          ├── right_00000.png
          ├── right_00001.png
          ├── ...
          ├── right_00299.png
        

## **Dependencies**

  * Python 3

  * OpenCV (`cv2`)

  * argparse (`pip install opencv-python numpy` if needed)



## Author

Ethan Ortiz

  

