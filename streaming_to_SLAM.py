import os
import time
import subprocess
from PIL import Image
from datetime import datetime
import cv2
import numpy as np
import shutil
import threading
import queue
import yaml
import signal
import depth_mapping as depth_mapping
import argparse

NEW_SUBMAP_EVERY = 10

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

def generate_config(scene_name, H, W, depth_scale):
    """ Generate the config file for SLAM """
    config = {
        'dataset_name': 'replica',
        'inherit_from': 'configs/Replica/replica.yaml',
        'data': {
            'scene_name': scene_name,
            'input_path': f'data/stream_connection/{scene_name}/',
            'output_path': f'output/stream_connection/{scene_name}/'
        },
        'cam': {
            'H': H,
            'W': W,
            'fx': (.5 * max(H, W)),
            'fy': (.5 * max(H, W)),
            'cx': (W / 2) - .5,
            'cy': (H / 2) - .5,
            'depth_scale': depth_scale
        },
        'project_name': "Gaussian_SLAM_replica",
        'checkpoint_path': None,
        'use_wandb': False,
        'seed': 0,
        'mapping': {
            'new_submap_every': NEW_SUBMAP_EVERY,
            'map_every': 4,
            'iterations': 50,
            'new_submap_iterations': 50,
            'new_submap_points_num': 100000,
            'new_submap_gradient_points_num': 25000,
            'new_frame_sample_size': -1,
            'new_points_radius': 0.000001,
            'current_view_opt_iterations': 0.4,
            'alpha_thre': 0.6,
            'pruning_thre': 0.1,
            'submap_using_motion_heuristic': True
        },
        'tracking': {
            'gt_camera': False,
            'w_color_loss': 0.95,
            'iterations': 60,
            'cam_rot_lr': 0.0002,
            'cam_trans_lr': 0.002,
            'odometry_type': "odometer",
            'help_camera_initialization': False,
            'init_err_ratio': 5,
            'odometer_method': "point_to_plane",
            'filter_alpha': False,
            'filter_outlier_depth': True,
            'alpha_thre': 0.98,
            'soft_alpha': True,
            'mask_invalid_depth': False
        }
    }
    
    config_path = f'configs/stream_connection/{scene_name}_config.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    os.makedirs(f"data/stream_connection/{scene_name}/", exist_ok=True)
    os.makedirs(f"data/stream_connection/{scene_name}/results/", exist_ok=True)
    os.makedirs(f"output/stream_connection/{scene_name}/", exist_ok=True)

    print("[System] Config file written: " + f'configs/stream_connection/{scene_name}_config.yaml')

def get_image_resolution(file_path):
    """ One time get image resolution """
    initial_size = os.path.getsize(file_path)
    time.sleep(0.1)
    if os.path.getsize(file_path) != initial_size:
        print(f"Warning: {file_path} might still be updating. Retrying...")
        time.sleep(0.2)
    with Image.open(file_path) as img:
        print("[System] Image Resolution: ", img.size)
        return img.size

def run_depth_mapping(img, op, prev_rgb=None, prev_depth=None, min_depth=None, max_depth=None, use_smoothing=True):
    """ Run depth mapping on the image """
    return depth_mapping.main(
        frame=img,
        output_dir=op,
        use_smoothing=use_smoothing,
        prev_rgb=prev_rgb,
        prev_depth=prev_depth,
        min_depth=min_depth,
        max_depth=max_depth)

def start_slam(scene_name, q):
    """ Subprocess to start the SLAM """
    slam_config_path = f"configs/stream_connection/{scene_name}_config.yaml"
    try:
        proc = subprocess.Popen(
            ["python3", "-u", "run_slam.py", slam_config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        print(f"[System] SLAM subprocess started with PID: {proc.pid}")
    except Exception as e:
        print(f"[System] Failed to start SLAM subprocess: {e}")

    def stream_output(process):
        for line in process.stdout:
            decoded_line = line.decode("utf-8").strip()
            if decoded_line:
                print(decoded_line)
        process.stdout.close()

    output_thread = threading.Thread(target=stream_output, args=(proc,))
    output_thread.start()
    time.sleep(15)
    os.kill(proc.pid, signal.SIGUSR2)
    q.put(proc)

def lv(maps, op, sizeinuint=5):
    """ Subprocess to start the live viewer """
    command = [
        "python", "live_ckpt_merger.py",
        "--submap_dir", maps,
        "--output_dir", op,
        "--batch_ckpts", str(sizeinuint),
    ]
    live_viewer_proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return live_viewer_proc

def main(args):
    watch_frame_folder = args.footage
    FRAME_SPACE_SIZE = args.space

    first_frame = f"/frame{'0' * FRAME_SPACE_SIZE}.jpg"
    resolution = get_image_resolution(watch_frame_folder + first_frame)
    print(f"[System] Resolution of {first_frame}: {resolution[0]}x{resolution[1]}")

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    scene_name = f"stream_scene_{current_time}"
    if args.scene_name:
        scene_name = f"{args.scene_name}"
    H, W = resolution[1], resolution[0]
    generate_config(scene_name, H, W, args.depth_scale)
    
    def run_main_loop():
        """ Each iteration is a frame processed; Concurrent with SLAM """
        i = 0
        curr_frame = watch_frame_folder + f"/frame{str(i).zfill(FRAME_SPACE_SIZE)}.jpg"
        next_frame = watch_frame_folder + f"/frame{str(i + 1).zfill(FRAME_SPACE_SIZE)}.jpg"
        prev_depth = None
        prev_rgb = None
        while os.path.exists(next_frame) and i < 100000:
            if i % 10 == 0:
                print(f"[Depth] Processing frame {i}")
            prev_depth, prev_rgb = run_depth_mapping(
                curr_frame,
                f"data/stream_connection/{scene_name}/results/",
                prev_rgb, prev_depth, args.depth_min, args.depth_max, args.depth_smoothing)
            shutil.copy2(curr_frame, f"data/stream_connection/{scene_name}/results/")
            i += 1
            curr_frame = next_frame
            next_frame = watch_frame_folder + f"/frame{str(i + 1).zfill(FRAME_SPACE_SIZE)}.jpg"

    main_loop_thread = threading.Thread(target=run_main_loop)
    main_loop_thread.start()
    
    slam_proc_queue = queue.Queue()
    slam_thread = threading.Thread(target=start_slam, args=(scene_name, slam_proc_queue))
    slam_thread.start()
    slam_proc = slam_proc_queue.get()
    
    viewer_proc = lv(f"output/stream_connection/{scene_name}/submaps/", f"output/stream_connection/{scene_name}/")
    print("[System] Live viewer started with PID:", viewer_proc.pid)

    main_loop_thread.join()
    time.sleep(15)
    if slam_proc and slam_proc.poll() is None:
        print("[System] Sending stop signal (SIGUSR1) to SLAM subprocess.")
        os.kill(slam_proc.pid, signal.SIGUSR1)
        slam_proc.wait()
        print("[System] SLAM process terminated.")

    slam_thread.join()
    print("[System] All threads finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SLAM pipeline .")
    parser.add_argument("--footage", type=str, default="Footage/Frames", help="Path to folder with frames")
    parser.add_argument("--space", type=int, default=5, help="Length of number in file name")
    parser.add_argument("--scene_name", type=str, help="(Optional) Name of the scene")
    
    # These arguments will change parameters for the depth calculation 
    # The default values are set to work well, but adjusting them based on the data may yield better results
    parser.add_argument("--depth_smoothing", type=bool, default=True, help="If true, temporal smoothing will be used")
    parser.add_argument("--depth_scale", type=int, default=50, help="Scalar for depth map")
    parser.add_argument("--depth_max", type=int, default=150, help="Used for adjusting the depths space linearly")
    parser.add_argument("--depth_min", type=int, default=0, help="Used for adjusting the depths space linearly")
    args = parser.parse_args()
    main(args)
