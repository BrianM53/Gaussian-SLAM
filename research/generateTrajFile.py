#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os
import math
import argparse
import time
from pathlib import Path

# This version of traj generation uses ORB to track camera motion and orientation.

def load_images(folder):
    """Load and sort image filenames from the given folder that start with 'frame'."""
    files = sorted(glob.glob(os.path.join(folder, 'frame*.png')))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, 'frame*.jpg')))
    return files

def compute_pose(img1, img2, K):
    """
    Computes the relative pose between img1 and img2 using ORB features,
    Essential matrix estimation, and pose recovery.
    
    Returns:
      R: 3x3 rotation matrix
      t: 3x1 translation vector (up-to-scale)
    """
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, None

    # Brute-force matching with Hamming distance.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 8:
        return None, None
    matches = sorted(matches, key=lambda x: x.distance)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Compute Essential Matrix with RANSAC.
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None, None
    
    # Recover pose.
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def generate_trajectory(image_folder, focal_length, dt=0.1):
    """
    Processes images in the specified folder, computes the relative pose between
    consecutive frames, accumulates the global transformation, and returns a list of 4x4
    transformation matrices (one per frame).
    """
    image_files = load_images(image_folder)
    if not image_files:
        print("No images found in the folder.")
        return []
    
    # Load first image to get image size.
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_img.shape
    
    # Construct intrinsic matrix K.
    K = np.array([[focal_length, 0, w/2],
                  [0, focal_length, h/2],
                  [0, 0, 1]])
    
    # The initial pose is identity.
    T_total = np.eye(4)
    trajectory = [T_total.copy()]  # Pose for frame 0.
    
    for i in range(1, len(image_files)):
        img1 = cv2.imread(image_files[i-1], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        R, t = compute_pose(img1, img2, K)
        if R is None or t is None:
            print(f"Warning: Pose could not be recovered for frame {i}")
            # Append the last known pose if tracking fails.
            trajectory.append(T_total.copy())
            continue
        
        # Build relative transformation matrix.
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Accumulate transformation.
        T_total = np.dot(T_total, T_rel)
        
        trajectory.append(T_total.copy())
        print(f"Processed frame {i}")
    
    return trajectory

def main(args):
    trajectory = generate_trajectory(args.image_folder, args.focal_length, dt=args.dt)
    if not trajectory:
        return
    
    output_file = Path(args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, T in enumerate(trajectory):
            # Flatten the 4x4 matrix in row-major order and format each number in scientific notation.
            line = " ".join("{:.18e}".format(x) for x in T.flatten())
            f.write(line + "\n")
    print(f"Trajectory saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a trajectory file from a folder of sequential images. "
                    "Each line of the output file is a flattened 4x4 transformation matrix in SE(3) format."
    )
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing sequential images (e.g., PNG or JPG).")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for the trajectory (e.g., traj.txt).")
    parser.add_argument("--focal_length", type=float, required=True, help="Camera focal length in pixels.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time interval between frames in seconds (for timestamp simulation).")
    args = parser.parse_args()
    main(args)