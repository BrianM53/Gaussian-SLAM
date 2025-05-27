#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def compute_disparity(left_img, right_img, ndisp, blockSize, disp12MaxDiff, uniquenessRatio):
    blur_amount = 75
    blurred_left_img = cv2.bilateralFilter(left_img, d=9, sigmaColor=blur_amount, sigmaSpace=blur_amount)
    blurred_right_img = cv2.bilateralFilter(right_img, d=9, sigmaColor=blur_amount, sigmaSpace=blur_amount)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity = -16 * ndisp,
        numDisparities = ndisp * 16, 
        blockSize = blockSize,
        uniquenessRatio = uniquenessRatio,
        speckleWindowSize = 200,
        speckleRange = 2,
        disp12MaxDiff = disp12MaxDiff,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        # preFilterCap=40,
        # P1=2*blockSize*blockSize,
        # P2=4*2*blockSize*blockSize
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    lmbda = 18000
    sigma = 1.3
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(left_img, right_img)
    dispr = right_matcher.compute(right_img, left_img)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, blurred_left_img, None, dispr, right_view=blurred_right_img) / 16.0
    # return displ
    return disparity

def process_video(video_path, arrangement):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if arrangement == "side-by-side":
            left_img = gray[:, :w // 2]
            right_img = gray[:, w // 2:]
        elif arrangement == "top-bottom":
            left_img = gray[:h // 2, :]
            right_img = gray[h // 2:, :]
        else:
            raise ValueError("Unknown arrangement. Choose 'side-by-side' or 'top-bottom'.")
        
        yield left_img, right_img
    
    cap.release()

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.video:
        video_frames = process_video(args.video, args.arrangement)
        frame_count = 0
        
        for left_img, right_img in video_frames:
            if left_img is None or right_img is None:
                print("Error: One or both frames are None")
                continue

            disparity = compute_disparity(left_img, right_img, args.ndisp, args.blockSize, args.disp12MaxDiff, args.uniquenessRatio)
            
            epsilon = 1e-6
            depth_map = (args.baseline * args.focal_length) / (disparity + epsilon)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            
            depth_path = output_dir / f"depth_{frame_count:05d}.png"
            npy_path = output_dir / f"depth_{frame_count:05d}.npy"
            cv2.imwrite(str(depth_path), depth_vis)
            np.save(str(npy_path), depth_map)
            
            frame_count += 1
            print(frame_count, "Depth map saved to", depth_path)
    elif args.left_img and args.right_img:
        left_img = cv2.imread(args.left_img, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(args.right_img, cv2.IMREAD_GRAYSCALE)
        disparity = compute_disparity(left_img, right_img, args.ndisp, args.blockSize, args.disp12MaxDiff, args.uniquenessRatio)
        epsilon = 1e-6
        depth_map = (args.baseline * args.focal_length) / (disparity + epsilon)
        
        depth_path = output_dir / f"depth_{args.left_img[-9:-4]}.png"
        npy_path = output_dir / f"depth_{args.left_img[-9:-4]}.npy"
        
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(depth_path), depth_vis)
    
    else:
        print("Error: Please provide either a video file or left and right images.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute disparity and depth maps from stereo video.")
    parser.add_argument("--video", type=str, help="Path to the stereo video file (.mov).")
    
    parser.add_argument("--left_img", type=str, help="Path to the left frame (PNG).")
    parser.add_argument("--right_img", type=str, help="Path to the right frame (PNG).")
    
    parser.add_argument("--arrangement", type=str, choices=["side-by-side", "top-bottom"], default="side-by-side",help="Stereo image arrangement in the video")
    parser.add_argument("--output_dir", type=str, default="output/depth", help="Output file for depth frames.")
    parser.add_argument("--focal_length", type=float, default=35, help="Focal length in pixels.")
    parser.add_argument("--baseline", type=float, default=500, help="Baseline (in same unit as depth desired).")
    parser.add_argument("--ndisp", type=int, default=1, help="Disparity range.")
    parser.add_argument("--uniquenessRatio", type=int, default=10, help="Uniqueness ratio for disparity computation.")
    parser.add_argument("--blockSize", type=int, default=3, help="Block size for single SGBM computation.")
    parser.add_argument("--disp12MaxDiff", type=int, default=100, help="disp12MaxDiff for single SGBM computation.")
    args = parser.parse_args()
    main(args)