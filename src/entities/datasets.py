import math
import os
import time
import signal
import threading
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import json
import imageio

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from kornia.utils import image_to_tensor


class BaseDataset(Dataset):
    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]
        self.depth_scale = dataset_config["depth_scale"]

        self.intrinsics = np.array([[self.fx, 0, self.cx],
                                    [0, self.fy, self.cy],
                                    [0, 0, 1]])
        self.color_paths = []
        self.depth_paths = []
        self.poses = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)

""" Original Replica dataset loading for static run.
    Uncomment for using run_slam.py """ 
# class Replica(BaseDataset):
#     def __init__(self, dataset_config: dict):
#         super().__init__(dataset_config)
#         self.color_paths = sorted(
#             list((self.dataset_path / "results").glob("frame*.jpg")))
#         self.depth_paths = sorted(
#             list((self.dataset_path / "results").glob("depth*.png")))
#         self.load_poses(self.dataset_path / "traj.txt")
#         print(f"Loaded {len(self.color_paths)} frames")

#     def load_poses(self, path):
#         self.poses = []
#         with open(path, "r") as f:
#             lines = f.readlines()
#         for line in lines:
#             c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
#             self.poses.append(c2w.astype(np.float32))

#     def __getitem__(self, index):
#         color_data = cv2.imread(str(self.color_paths[index]))
#         color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
#         depth_data = cv2.imread(
#             str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
#         depth_data = depth_data.astype(np.float32) / self.depth_scale
#         return index, color_data, depth_data, self.poses[index]


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "color").glob("*.jpg")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(self.dataset_path / "pose")

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(path.glob('*.txt'),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                ls.append(list(map(float, line.split(' '))))
            c2w = np.array(ls).reshape(4, 4).astype(np.float32)
            self.poses.append(c2w)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]

def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")


class Replica(BaseDataset):
    """ Edited Replica dataset for running with streaming """
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)

        self.frame_queue = Queue()
        self.zero_padding = 5  # default

        # Initial file list load.
        self._refresh_file_lists(initial=True)

        print(f"[Pose] Queued {len(self.color_paths)} frames to pose computation thread.")
        self.pose_file = self.dataset_path / "traj.txt"

        self.streaming_active = False
        self.pose_lock = threading.Lock()

        # Start the pose computation thread only once.
        self.pose_thread = threading.Thread(target=self.compute_poses_from_orb, daemon=True)
        self.pose_thread.start()

        # Set signal handlers (they will trigger start/stop of streaming threads).
        signal.signal(signal.SIGUSR2, self.start_streaming)
        signal.signal(signal.SIGUSR1, self.stop_streaming)

        print("[SLAM] Initialized Replica Streaming Dataset")

    def _refresh_file_lists(self, initial=False):
        """Internal method to update file lists from the results directory."""
        results_path = self.dataset_path / "results"
        new_color_paths = sorted(results_path.glob("frame*.jpg"))
        new_depth_paths = sorted(results_path.glob("depth*.png"))
        if initial:
            self.color_paths = new_color_paths
            self.depth_paths = new_depth_paths
            if self.color_paths:
                prev_file = self.color_paths[0]
            else:
                prev_file = None
            if len(self.color_paths) > 1:
                # Queue each frame pair from the initial set.
                for curr_file in self.color_paths[1:]:
                    self.frame_queue.put((prev_file, curr_file))
                    prev_file = curr_file
                # Determine zero_padding from the filename.
                test_name = self.color_paths[0].stem
                digits = ''.join(ch for ch in reversed(test_name) if ch.isdigit())[::-1]
                self.zero_padding = len(digits)
        else:
            # For refresh, only append new files that are not already in our list.
            for f in new_color_paths:
                if f not in self.color_paths:
                    self.color_paths.append(f)
                    if len(self.color_paths) > 1:
                        # Use the last two files as a new pair.
                        self.frame_queue.put((self.color_paths[-2], self.color_paths[-1]))
            for f in new_depth_paths:
                if f not in self.depth_paths:
                    self.depth_paths.append(f)

    def refresh(self):
        """Public method to refresh file lists without restarting threads."""
        self._refresh_file_lists(initial=False)

    def start_streaming(self, signum=None, frame=None):
        print("[SLAM] Starting frame streaming...")
        self.streaming_active = True
        self.stream_thread = threading.Thread(target=self._watch_for_new_frames, daemon=True)
        self.stream_thread.start()

    def stop_streaming(self, signum, frame):
        print("[SLAM] Stop signal received")
        self.streaming_active = False
        if hasattr(self, 'stream_thread') and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1)
            print("[SLAM] Streaming thread has stopped.")
        if self.pose_thread.is_alive():
            self.pose_thread.join(timeout=1)
            print("[SLAM] Pose thread has stopped.")
        self.write_poses_to_file()

    def compute_poses_from_orb(self):
        print("[Pose] Starting pose computation thread...")
        K_mat = self.intrinsics
        T_total = np.eye(4)
        with self.pose_lock:
            if self.pose_file.exists():
                self.poses = self.load_existing_poses()
                T_total = self.poses[-1] if self.poses else T_total

        while True:
            try:
                curr_color_file, prev_color_file = self.frame_queue.get(timeout=0.2)
            except Empty:
                if not self.streaming_active or self.frame_queue.empty():
                    print("[Pose] No more frames to process and streaming is stopped. Exiting pose thread.")
                    self.write_poses_to_file()
                    break
                continue

            img1 = cv2.imread(str(prev_color_file), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(curr_color_file), cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"[Pose] Warning: Missing or invalid image for {curr_color_file}")
                time.sleep(0.2)
                continue

            R, t = self.compute_relative_pose(img1, img2, K_mat)
            if R is None or t is None:
                print(f"[Pose] Pose could not be recovered for {curr_color_file}")
                with self.pose_lock:
                    self.poses.append(T_total.copy())
                continue

            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.flatten()
            T_total = T_total @ T_rel

            with self.pose_lock:
                idx = len(self.poses)
                self.poses.append(T_total.copy())
                if idx == 1:
                    self.poses[0] = T_total.copy()
                if (idx + 1) % 10 == 0:
                    print(f"[Pose] Frame {idx + 1} processed")

    def write_poses_to_file(self, path=None):
        if path is None:
            path = self.dataset_path / "traj.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {len(self.poses)} poses to {path}")
        with open(path, "w") as f:
            for pose in self.poses:
                flat_pose = pose.flatten()
                line = " ".join(f"{x:.18e}" for x in flat_pose)
                f.write(line + "\n")

    def load_existing_poses(self):
        poses = []
        with open(self.pose_file, "r") as f:
            for line in f:
                T = np.array(list(map(float, line.strip().split()))).reshape(4, 4)
                poses.append(T.astype(np.float32))
        return poses

    def compute_relative_pose(self, img1, img2, K):
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return None, None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 8:
            return None, None

        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return None, None

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        return R, t

    def _watch_for_new_frames(self):
        print("[SLAM] Watching for new frames...")
        idx = len(self.color_paths)
        prev_file = self.color_paths[-1] if self.color_paths else None
        sleeps = 0

        while self.streaming_active:
            curr_file = self.dataset_path / "results" / f"frame{str(idx).zfill(self.zero_padding)}.jpg"
            depth_file = self.dataset_path / "results" / f"depth{str(idx).zfill(self.zero_padding)}.png"
            if curr_file.exists() and depth_file.exists():
                if prev_file is not None:
                    self.frame_queue.put((prev_file, curr_file))
                    print(f"[SLAM] Appended new frame: {idx}")
                prev_file = curr_file
                self.color_paths.append(curr_file)
                self.depth_paths.append(depth_file)
                idx += 1
                sleeps = 0
            else:
                time.sleep(0.1)
                sleeps += 1
                if sleeps % 100 == 0:
                    print("[SLAM] Waiting for new frames...")
                if sleeps > 500:
                    break
        print("[SLAM] Exiting _watch_for_new_frames thread.")

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale

        with self.pose_lock:
            pose = self.poses[index] if index < len(self.poses) else np.eye(4, dtype=np.float32)

        return index, color_data, depth_data, pose
