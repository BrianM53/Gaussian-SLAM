import cv2
import os
import argparse
from pathlib import Path

def extract_frames(input_file, left_folder, right_folder, arrangement='side-by-side'):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if arrangement == 'side-by-side':
            # Split the frame horizontally
            h, w, _ = frame.shape
            mid = w // 2
            left_frame = frame[:, :mid]
            right_frame = frame[:, mid:]
        elif arrangement == 'top-bottom':
            # Split the frame vertically
            h, w, _ = frame.shape
            mid = h // 2
            left_frame = frame[:mid, :]
            right_frame = frame[mid:, :]
        else:
            print("Unknown arrangement specified. Exiting.")
            break

        left_path = os.path.join(left_folder, f"left_{frame_index:05d}.png")
        right_path = os.path.join(right_folder, f"right_{frame_index:05d}.png")
        cv2.imwrite(left_path, left_frame)
        cv2.imwrite(right_path, right_frame)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{total_frames} frames.")

    cap.release()
    print("Frame extraction complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract left and right frames from stereo .mov footage."
    )
    parser.add_argument("input_file", type=str, help="Path to input .mov file.")
    parser.add_argument("left_folder", type=str, help="Folder to save left frames (PNG).")
    parser.add_argument("right_folder", type=str, help="Folder to save right frames (PNG).")
    parser.add_argument(
        "--arrangement",
        type=str,
        choices=['side-by-side', 'top-bottom'],
        default='side-by-side',
        help="Arrangement of the stereo footage (default: side-by-side)."
    )
    args = parser.parse_args()

    # Ensure output directories exist
    Path(args.left_folder).mkdir(parents=True, exist_ok=True)
    Path(args.right_folder).mkdir(parents=True, exist_ok=True)

    extract_frames(args.input_file, args.left_folder, args.right_folder, args.arrangement)
