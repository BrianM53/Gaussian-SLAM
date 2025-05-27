#!/usr/bin/env python3

"""
This is the third file for our “live viewer” family of scripts which all do the same job but I'm uploading all of them for R&D purposes cause they have differences in performance.
 This one achieves achieves the crispest reconstruction so far, with minimal blobbiness. The trade-off
is that it’s quite slow, and there may still be some small holes left. However,
overall fidelity and sharpness surpass the other two files, at the expense
of computation time.
"""

import argparse
import glob
import os
import time
from pathlib import Path

import meshcat
import meshcat.geometry as g
import numpy as np
import open3d as o3d
import torch

from meshcat.geometry import TriangularMeshGeometry, Mesh, MeshPhongMaterial


def transfer_pointcloud_color_to_mesh(o3d_pcd, o3d_mesh):
    """Nearest-neighbor color assignment from point cloud to alpha-shape mesh."""
    if not o3d_pcd.has_colors():
        print("⚠️ No color in accum point cloud, skipping color assignment.")
        return

    pcd_tree = o3d.geometry.KDTreeFlann(o3d_pcd)
    pcd_colors = np.asarray(o3d_pcd.colors)

    new_colors = []
    for v in o3d_mesh.vertices:
        # 1 nearest neighbor
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
        nearest = idx[0]
        new_colors.append(pcd_colors[nearest])

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_colors))


def load_gaussian_params(ckpt_file):
    data = torch.load(ckpt_file, map_location="cpu")
    if isinstance(data, dict) and "gaussian_params" in data:
        return data["gaussian_params"]
    return data


def extract_points_and_colors(gparams):
    """
    Return (xyz, color), with color in [0..1], or None if no features_dc in submap.
    """
    xyz = gparams["xyz"]
    if torch.is_tensor(xyz):
        xyz = xyz.cpu().numpy()
    else:
        xyz = np.asarray(xyz)

    color = None
    if "features_dc" in gparams:
        f_dc = gparams["features_dc"]
        if torch.is_tensor(f_dc):
            f_dc = f_dc.cpu().numpy()
        # handle shape (N,3,1) or (N,1,3)
        if f_dc.ndim == 3 and f_dc.shape[2] == 1:
            f_dc = f_dc[:, :, 0]
        elif f_dc.ndim == 3 and f_dc.shape[1] == 1:
            f_dc = f_dc[:, 0, :]
        color = np.clip(f_dc, 0.0, 1.0)

    return xyz, color


def merge_ckpt_batch_into_cloud(ckpt_files):
    """
    Merge a small batch of .ckpt submaps => single open3d PointCloud with color
    if all submaps have color, else no color.
    """
    all_xyz = []
    all_col = []

    for ckpt in ckpt_files:
        try:
            gparams = load_gaussian_params(ckpt)
            xyz, col = extract_points_and_colors(gparams)
            if xyz is None or xyz.shape[0] == 0:
                print(f"⚠️ {os.path.basename(ckpt)} has no points.")
                continue
            all_xyz.append(xyz)

            if col is not None and col.shape[0] == xyz.shape[0]:
                all_col.append(col)
            else:
                all_col.append(None)

            print(f"✅ Merged {xyz.shape[0]} pts from {os.path.basename(ckpt)}")
        except Exception as e:
            print(f"❌ Skipping {os.path.basename(ckpt)}: {e}")

    if not all_xyz:
        return None

    merged_xyz = np.concatenate(all_xyz, axis=0)

    # only use color if *all* submaps had color
    use_color = all(c is not None for c in all_col)
    if use_color:
        merged_colors = np.concatenate([c for c in all_col if c is not None], axis=0)
        merged_colors = np.clip(merged_colors, 0.0, 1.0)
    else:
        merged_colors = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_xyz)
    if merged_colors is not None and merged_colors.shape[0] == merged_xyz.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return pcd


def live_merge_ckpts_alpha(submap_dir,
                           output_dir,
                           batch_ckpts=5,
                           refresh_rate=2.0,
                           alpha=0.05):
    """
    Batch merges submaps => accum_pcd => alpha-shape mesh => color => display.
    alpha ~ (0.02...0.1). Larger alpha => more bridging => fewer holes but 
    can create big artificial surfaces. Tweak to taste.
    """
    print(f"🌐 Opening Meshcat viewer. Watching for .ckpt in: {submap_dir}")
    vis = meshcat.Visualizer().open()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = set()
    accum_pcd = o3d.geometry.PointCloud()

    while True:
        all_ckpts = set(sorted(glob.glob(os.path.join(submap_dir, "*.ckpt"))))
        new_files = list(all_ckpts - processed)

        if not new_files:
            time.sleep(refresh_rate)
            continue

        batch = sorted(new_files)[:batch_ckpts]
        print(f"\n📊 Merging batch of {len(batch)} ckpt files...")

        pcd_batch = merge_ckpt_batch_into_cloud(batch)
        if pcd_batch is None or len(pcd_batch.points)==0:
            print("⚠️ This batch is empty. Skipping.")
            processed.update(batch)
            continue

        # accumulate
        if len(accum_pcd.points)==0:
            accum_pcd = pcd_batch
        else:
            old_pts = np.asarray(accum_pcd.points)
            new_pts = np.asarray(pcd_batch.points)
            combined_xyz = np.concatenate((old_pts, new_pts), axis=0)
            if accum_pcd.has_colors() and pcd_batch.has_colors():
                old_col = np.asarray(accum_pcd.colors)
                new_col = np.asarray(pcd_batch.colors)
                combined_col = np.concatenate((old_col, new_col), axis=0)
                updated = o3d.geometry.PointCloud()
                updated.points = o3d.utility.Vector3dVector(combined_xyz)
                updated.colors = o3d.utility.Vector3dVector(combined_col)
                accum_pcd = updated
            else:
                updated = o3d.geometry.PointCloud()
                updated.points = o3d.utility.Vector3dVector(combined_xyz)
                accum_pcd = updated

        # remove outliers
        if len(accum_pcd.points)>2000:
            accum_pcd, _ = accum_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

        # (Optional) downsample
        # accum_pcd = accum_pcd.voxel_down_sample(0.01)

        # Save debug point cloud
        pcd_filename = f"accum_{int(time.time())}.ply"
        pcd_path = out_dir / pcd_filename
        o3d.io.write_point_cloud(str(pcd_path), accum_pcd)
        print(f"✅ Saved accum pcd => {pcd_path} with {len(accum_pcd.points)} points")

        # alpha-shape to fill holes more aggressively
        print(f"🌀 Creating mesh from alpha shape with alpha={alpha}")
        # must convert to triangle mesh
        # alpha shapes can be slow for large # of points
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(accum_pcd, alpha)

        # geometry cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()

        # color
        if accum_pcd.has_colors():
            transfer_pointcloud_color_to_mesh(accum_pcd, mesh)
        else:
            print("⚠️ accum_pcd has no color, skipping color xfer")

        mesh_filename = f"mesh_{int(time.time())}.ply"
        mesh_path = out_dir / mesh_filename
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"✅ Saved alpha-shape mesh => {mesh_path}, #verts={len(mesh.vertices)}")

        # Display in Meshcat
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if mesh.has_vertex_colors():
            vcols = np.asarray(mesh.vertex_colors)
        else:
            vcols = np.ones((verts.shape[0],3), dtype=np.float32)

        geom = TriangularMeshGeometry(verts, faces, color=vcols)
        mat = MeshPhongMaterial(vertexColors=True)
        obj = g.Mesh(geom, mat)

        vis["accum_alpha_mesh"].set_object(obj)
        print(f"🎨 Displayed alpha-shape mesh with {len(verts)} vertices and {len(faces)} faces.")

        processed.update(batch)
        time.sleep(refresh_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Accumulate .ckpt submaps, then form an alpha-shape mesh to fill holes more aggressively."
    )
    parser.add_argument("--submap_dir", required=True, help="Directory containing .ckpt submaps.")
    parser.add_argument("--output_dir", required=True, help="Output folder for .ply files.")
    parser.add_argument("--batch_ckpts", type=int, default=5, help="How many new ckpts per iteration.")
    parser.add_argument("--refresh_rate", type=float, default=2.0, help="Seconds between scans.")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Alpha shape radius. Larger => more bridging => fewer holes, but more artificial surfaces.")
    args = parser.parse_args()

    live_merge_ckpts_alpha(
        submap_dir=args.submap_dir,
        output_dir=args.output_dir,
        batch_ckpts=args.batch_ckpts,
        refresh_rate=args.refresh_rate,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()
