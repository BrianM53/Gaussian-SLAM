#!/usr/bin/env python3
# Author : Ethan Ortiz
import os
import argparse
import glob
import tempfile
import shutil
import numpy as np
import cv2
import csv

# clamp range for scale to avoid collapse
MIN_SCALE = 0.9
MAX_SCALE = 1.1

def parse_args():
    p = argparse.ArgumentParser(
        description="Two-pass affine scaling of depth maps with mean anchoring"
    )
    p.add_argument("input_dir", help="Directory of raw depth-map frames")
    p.add_argument("output_dir", help="Directory to write final, two-pass scaled frames")
    p.add_argument("--pattern", default="*.png",
                   help="Glob pattern to match your frames (e.g. '*.tif')")
    p.add_argument("--log_csv", default="scale_factors.csv",
                   help="CSV filename (in output_dir) to record scales & offsets")
    return p.parse_args()

def get_sorted_files(d, pattern):
    files = glob.glob(os.path.join(d, pattern))
    files.sort()
    return files

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not load '{path}'")
    return img

def save_img(path, img, dtype):
    # clip to original dtype range then convert
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        img = np.clip(img, info.min, info.max)
    img = img.astype(dtype)
    cv2.imwrite(path, img)

def compute_affine(prev, curr):
    """
    Solve for s, b minimizing ||prev - (s*curr + b)||^2
    """
    mean_p = prev.mean()
    mean_c = curr.mean()
    cov_pc = np.sum((prev-mean_p)*(curr-mean_c))
    var_c  = np.sum((curr-mean_c)**2)
    s = cov_pc/var_c if var_c!=0 else 1.0
    # clamp to avoid extreme darkening
    s = float(np.clip(s, MIN_SCALE, MAX_SCALE))
    b = mean_p - s*mean_c
    return s, b

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) gather raw frames
    in_files = get_sorted_files(args.input_dir, args.pattern)
    if not in_files:
        print("No input files found. Check your --pattern and input_dir.")
        return

    # load first raw frame to get dtype & reference mean
    first_raw = load_img(in_files[0])
    dtype = first_raw.dtype
    ref_mean = float(first_raw.mean())

    # temp dir for forward pass
    with tempfile.TemporaryDirectory() as tmp_fwd:
        forward_params = {}  # basename -> (s_f, b_f)

        # --- Forward pass ---
        # anchor first frame
        base0 = os.path.basename(in_files[0])
        shutil.copy(in_files[0], os.path.join(tmp_fwd, base0))
        forward_params[base0] = (1.0, 0.0)
        prev = first_raw.astype(np.float32)

        for path in in_files[1:]:
            curr = load_img(path).astype(np.float32)
            s_f, b_f = compute_affine(prev, curr)
            scaled = curr * s_f + b_f
            out_fwd = os.path.join(tmp_fwd, os.path.basename(path))
            save_img(out_fwd, scaled, dtype)
            forward_params[os.path.basename(path)] = (s_f, b_f)
            prev = scaled

        # --- Backward pass ---
        fwd_files = get_sorted_files(tmp_fwd, args.pattern)
        backward_params = {}  # basename -> (s_b, b_b_total)

        # anchor last frame (backward scale=1, but we still mean-anchor)
        last = fwd_files[-1]
        img_last = load_img(last).astype(np.float32)
        mean_offset = ref_mean - img_last.mean()
        final_last = img_last + mean_offset
        out_last = os.path.join(args.output_dir, os.path.basename(last))
        save_img(out_last, final_last, dtype)
        backward_params[os.path.basename(last)] = (1.0, mean_offset)
        prev = final_last

        # walk backwards
        for path in reversed(fwd_files[:-1]):
            curr = load_img(path).astype(np.float32)
            s_b, b_b = compute_affine(prev, curr)
            scaled_b = curr * s_b + b_b
            # mean-anchor to first frame
            mean_offset = ref_mean - scaled_b.mean()
            final_b = scaled_b + mean_offset

            out_path = os.path.join(args.output_dir, os.path.basename(path))
            save_img(out_path, final_b, dtype)
            # record total backward offset = b_b + mean_offset
            backward_params[os.path.basename(path)] = (s_b, b_b + mean_offset)
            prev = final_b

    # --- Write CSV log ---
    csv_path = os.path.join(args.output_dir, args.log_csv)
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['filename',
                    'forward_scale','forward_offset',
                    'backward_scale','backward_offset'])
        for path in in_files:
            fn = os.path.basename(path)
            s_f, b_f = forward_params.get(fn, (None,None))
            s_b, b_b = backward_params.get(fn, (None,None))
            w.writerow([fn, s_f, b_f, s_b, b_b])

    print("✅ Two-pass affine scaling complete.")
    print(f"Final frames in: {args.output_dir}")
    print(f"Scale & offset log: {csv_path}")

if __name__ == "__main__":
    main()
