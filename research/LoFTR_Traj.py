#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os
import argparse
from pathlib import Path
import torch
import kornia as K
import kornia.feature as KF
from kornia.utils import image_to_tensor

def load_images(folder):
    """Load and sort image filenames from the given folder that start with 'frame'."""
    files = sorted(glob.glob(os.path.join(folder, 'frame*.png')))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, 'frame*.jpg')))
    return files

def prepare_image_for_loftr(img):
    """Convert image to proper format for LoFTR"""
    if img is None:
        return None
    # Ensure image is 2D grayscale (H, W)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert to tensor (C, H, W) where C=1 for grayscale
    img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.
    return img_tensor

def compute_pose_loftr(img1, img2, K_matrix, loftr_model):
    """Computes relative pose using LoFTR"""
    try:
        # Prepare images (should be (1, H, W) tensors)
        img1_tensor = prepare_image_for_loftr(img1)
        img2_tensor = prepare_image_for_loftr(img2)
        
        if img1_tensor is None or img2_tensor is None:
            print("Error: Could not convert images to tensors")
            return None, None
            
        # Add batch dimension (B, C, H, W) where B=1
        img1_tensor = img1_tensor.unsqueeze(0).to('cuda')
        img2_tensor = img2_tensor.unsqueeze(0).to('cuda')
        
        # Match features
        input_dict = {
            "image0": img1_tensor, 
            "image1": img2_tensor
        }
        
        with torch.no_grad():
            correspondences = loftr_model(input_dict)
        
        pts1 = correspondences['keypoints0'].cpu().numpy()
        pts2 = correspondences['keypoints1'].cpu().numpy()
        
        if len(pts1) < 20:
            print(f"Warning: Only {len(pts1)} matches found")
            return None, None
            
        E, mask = cv2.findEssentialMat(pts1, pts2, K_matrix, 
                                     method=cv2.RANSAC,
                                     prob=0.999,
                                     threshold=0.5)
        if E is None:
            print("Warning: Essential matrix could not be computed")
            return None, None
            
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_matrix, mask=mask)
        return R, t
        
    except Exception as e:
        print(f"LoFTR processing error: {str(e)}")
        return None, None

def generate_trajectory(image_folder, focal_length, dt=0.1):
    """Generate trajectory using only LoFTR"""
    image_files = load_images(image_folder)
    if not image_files:
        print("No images found in the folder.")
        return []
    
    # Load first image to get dimensions
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        print("Could not read first image")
        return []
    
    h, w = first_img.shape
    
    # Camera intrinsic matrix
    K_matrix = np.array([[focal_length, 0, w/2],
                        [0, focal_length, h/2],
                        [0, 0, 1]])
    
    # Initialize LoFTR
    print("Initializing LoFTR...")
    loftr_model = KF.LoFTR(pretrained='indoor').eval().to('cuda')
    
    T_total = np.eye(4)
    trajectory = [T_total.copy()]
    
    for i in range(1, len(image_files)):
        img1 = cv2.imread(image_files[i-1], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not load frame {i-1} or {i}")
            trajectory.append(T_total.copy())
            continue
        
        R, t = compute_pose_loftr(img1, img2, K_matrix, loftr_model)
        
        if R is None or t is None:
            print(f"Warning: Pose could not be recovered for frame {i}")
            trajectory.append(T_total.copy())
            continue
        
        # Build relative transformation matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t.flatten()
        
        # Accumulate transformation
        T_total = np.dot(T_total, T_rel)
        trajectory.append(T_total.copy())
        print(f"Processed frame {i}/{len(image_files)}")
    
    return trajectory

def main(args):
    trajectory = generate_trajectory(
        args.image_folder, 
        args.focal_length,
        dt=args.dt
    )
    
    if not trajectory:
        return
    
    output_file = Path(args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for T in trajectory:
            line = " ".join("{:.18e}".format(x) for x in T.flatten())
            f.write(line + "\n")
    print(f"Trajectory saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate trajectory from sequential images using LoFTR"
    )
    parser.add_argument("--image_folder", type=str, required=True,
                       help="Path to folder containing sequential images")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path for the trajectory")
    parser.add_argument("--focal_length", type=float, required=True,
                       help="Camera focal length in pixels")
    parser.add_argument("--dt", type=float, default=1/60,
                       help="Time interval between frames in seconds")
    args = parser.parse_args()
    
    main(args)