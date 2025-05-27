#!/usr/bin/env python3
# Author : Ethan Ortiz
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import pipeline

# Load model once globally
depth_pipeline = pipeline(
    task="depth-estimation",
    model="LiheYoung/depth-anything-small-hf"
)

def downscale(img, factor=2):
    return cv2.resize(img, (img.shape[1] // factor, img.shape[0] // factor), interpolation=cv2.INTER_AREA)

def upscale(img, shape):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

def depth_anything_generate(image_path: str) -> np.ndarray:
    """
    Generates and optionally saves/displays a depth map from a single image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Normalized depth map as a uint8 array.
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image)
    
    # Downscale
    downscaled_np = downscale(original_np, factor=2)
    downscaled_img = Image.fromarray(downscaled_np)

    # Get depth map
    result = depth_pipeline(downscaled_img)
    depth = np.array(result["depth"])

    # Normalize and convert to uint8
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Upscale back to original size
    depth_upscaled = upscale(depth_norm, original_np.shape[:2])


    return depth_upscaled
def smooth_with_affine(prev_rgb, curr_rgb, prev_depth, curr_depth, alpha=0.7):
    """
    Estimate affine from prev_rgb→curr_rgb, warp prev_depth,
    then blend: alpha*curr + (1-alpha)*warped_prev.
    """
    # 1) detect & match features (you can swap in ORB, SIFT, etc.)
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(prev_rgb, None)
    kp2, des2 = orb.detectAndCompute(curr_rgb, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # 2) pull out point‐pairs
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    # 3) estimate affine
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    if M is None:
        # fallback: no smoothing if estimation failed
        return None
    # 4) warp prev_depth
    h, w = curr_rgb.shape[:2]
    warped = cv2.warpAffine(prev_depth, M, (w,h), flags=cv2.INTER_LINEAR)
    # 5) blend
    return cv2.addWeighted(curr_depth, alpha, warped, 1-alpha, 0)

def main(output_dir, frame=None, use_smoothing=False, prev_rgb=None, prev_depth=None, min_depth=None, max_depth=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if frame:
        depth_path = output_dir / f"depth{frame[-9:-4]}.png"
        curr_pil = Image.open(frame).convert("RGB")
        curr_np  = np.array(curr_pil)
        
        curr_depth = depth_anything_generate(frame)
        if use_smoothing and prev_rgb is not None and prev_depth is not None:
            # try smoothing
            smoothed = None
            try:
                smoothed = smooth_with_affine(prev_rgb, curr_np, prev_depth, curr_depth, alpha=0.8)
            except Exception as e:
                print(f"[Depth] Error during smoothing: {e}")
            if smoothed is not None:
                curr_depth = smoothed
        if min_depth is not None and max_depth is not None:
            # all depth values are adjusted to linearly fit inside of min_depth and the compute max_depth
            min_depth = min_depth
            max_depth = max_depth
            
            scale = (max_depth - min_depth) / 255
            curr_depth = (curr_depth * scale) + min_depth
            
            # print(f"Min depth: {curr_depth.min()}, Scale: {scale}")
            # print(f"Max depth: {curr_depth.max()}")
            
            # fit to closest integer 0-255
            curr_depth = np.clip(curr_depth, 0, 255).astype(np.uint8)
        
        depth_inverted = 255 - curr_depth
        cv2.imwrite(depth_path, depth_inverted)
        return curr_depth, curr_np
    else:
        print("Error: Please provide a left image.")
    return (None, None)
        
if __name__ == "__main__":
    # This probably will never be used
    arg_parser = argparse.ArgumentParser(description="Depth Mapping Script")
    arg_parser.add_argument("--frame", type=str, help="Path to the current frame")
    arg_parser.add_argument("--output_dir", type=str, help="Output directory for depth maps")
    arg_parser.add_argument("--use_smoothing", action="store_true", help="Use smoothing with affine transformation")
    arg_parser.add_argument("--prev_rgb", type=str, help="Path to the previous RGB image")
    arg_parser.add_argument("--prev_depth", type=str, help="Path to the previous depth map")
    arg_parser.add_argument("--min_depth", type=float, default=0, help="Minimum depth value")
    arg_parser.add_argument("--max_depth", type=float, default=150, help="Maximum depth value")
    args = arg_parser.parse_args()

    # Call the main function
    main(
        output_dir=args.output_dir,
        frame=args.frame,
        use_smoothing=args.use_smoothing,
        prev_rgb=args.prev_rgb,
        prev_depth=args.prev_depth,
        min_depth=args.min_depth,
        max_depth=args.max_depth
    )