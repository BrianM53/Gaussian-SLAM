import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def load_trajectory(file_path):
    """Load trajectory from file (each line = flattened 4x4 transformation matrix)."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                T = np.array(values).reshape(4, 4)
                poses.append(T)
    return poses

def plot_trajectory(poses, mode="3D"):
    """Plot trajectory from list of 4x4 poses."""
    positions = np.array([T[:3, 3] for T in poses])

    if mode == "2D":
        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 2], 'b-', marker='o')  # x vs z
        plt.title("Camera Trajectory (Top-down View)")
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.axis('equal')
        plt.grid(True)
        plt.savefig("trajectory_plot.png", dpi=300)
        print("Trajectory plot saved as trajectory_plot.png")


    elif mode == "3D":
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', marker='o')
        ax.set_title("3D Camera Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=30, azim=45)
        plt.savefig("trajectory_plot.png", dpi=300)
        print("Trajectory plot saved as trajectory_plot.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize trajectory from 4x4 pose file.")
    parser.add_argument("trajectory_file", type=str, help="Path to trajectory file (e.g., traj.txt)")
    parser.add_argument("--mode", type=str, choices=["2D", "3D"], default="3D", help="Plot in 2D or 3D")
    args = parser.parse_args()

    poses = load_trajectory(args.trajectory_file)
    plot_trajectory(poses, mode=args.mode)
