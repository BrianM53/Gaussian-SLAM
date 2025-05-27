#!/usr/bin/env python3

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

# REPLACE with your existing color/mesh code if you like
def transfer_pointcloud_color_to_mesh(o3d_pcd, o3d_mesh):
    pcd_tree = o3d.geometry.KDTreeFlann(o3d_pcd)
    pcd_points = np.asarray(o3d_pcd.points)
    pcd_colors = np.asarray(o3d_pcd.colors)

    mesh_vertex_colors = []
    for v in o3d_mesh.vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
        nearest_idx = idx[0]
        mesh_vertex_colors.append(pcd_colors[nearest_idx])

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_vertex_colors))

def load_gaussian_params(ckpt_file):
    data = torch.load(ckpt_file, map_location="cpu")
    if isinstance(data, dict) and "gaussian_params" in data:
        return data["gaussian_params"]
    return data

def extract_points_and_colors(gparams):
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
        if f_dc.ndim == 3 and f_dc.shape[2] == 1:
            f_dc = f_dc[:, :, 0]
        elif f_dc.ndim == 3 and f_dc.shape[1] == 1:
            f_dc = f_dc[:, 0, :]
        color = np.clip(f_dc, 0.0, 1.0)
    return xyz, color

def merge_ckpt_batch_into_cloud(ckpt_files):
    all_xyz = []
    all_col = []
    for f in ckpt_files:
        try:
            gparams = load_gaussian_params(f)
            xyz, col = extract_points_and_colors(gparams)
            if xyz is None or xyz.shape[0] == 0:
                print(f"⚠️ {os.path.basename(f)} has no points.")
                continue
            all_xyz.append(xyz)
            if col is not None and col.shape[0] == xyz.shape[0]:
                all_col.append(col)
            else:
                all_col.append(None)
            print(f"✅ Merged {xyz.shape[0]} pts from {os.path.basename(f)}")
        except Exception as e:
            print(f"❌ Skipping {os.path.basename(f)}: {e}")

    if not all_xyz:
        return None

    xyz_merged = np.concatenate(all_xyz, axis=0)
    use_color = all(c is not None for c in all_col)
    if use_color:
        merged_colors = np.concatenate([c for c in all_col if c is not None], axis=0)
        merged_colors = np.clip(merged_colors, 0.0, 1.0)
    else:
        merged_colors = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_merged)
    if merged_colors is not None and merged_colors.shape[0] == xyz_merged.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return pcd

def live_merge_ckpts(submap_dir, output_dir, batch_ckpts=5, refresh_rate=2.0):
    print("🌐 Opening Meshcat viewer. Watching for .ckpt in:", submap_dir)
    vis = meshcat.Visualizer().open()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = set()
    accum_pcd = o3d.geometry.PointCloud()

    while True:
        ckpt_files = set(sorted(glob.glob(os.path.join(submap_dir, "*.ckpt"))))
        new_files = list(ckpt_files - processed)
        if not new_files:
            time.sleep(refresh_rate)
            continue

        batch = sorted(new_files)[:batch_ckpts]
        print(f"📊 Merging batch of {len(batch)} ckpt files...")

        pcd_merged = merge_ckpt_batch_into_cloud(batch)
        if pcd_merged is None or len(pcd_merged.points) == 0:
            print("⚠️ This batch is empty. Skipping.")
        else:
            # Accumulate
            if len(accum_pcd.points) == 0:
                accum_pcd = pcd_merged
            else:
                combined_xyz = np.concatenate((np.asarray(accum_pcd.points),
                                               np.asarray(pcd_merged.points)), axis=0)
                if accum_pcd.has_colors() and pcd_merged.has_colors():
                    combined_col = np.concatenate((np.asarray(accum_pcd.colors),
                                                   np.asarray(pcd_merged.colors)), axis=0)
                    new_accum = o3d.geometry.PointCloud()
                    new_accum.points = o3d.utility.Vector3dVector(combined_xyz)
                    new_accum.colors = o3d.utility.Vector3dVector(combined_col)
                    accum_pcd = new_accum
                else:
                    new_accum = o3d.geometry.PointCloud()
                    new_accum.points = o3d.utility.Vector3dVector(combined_xyz)
                    accum_pcd = new_accum

            # --- NEW: Clean up accum_pcd before Poisson
            # 1) Remove outliers
            if len(accum_pcd.points) > 1000:
                accum_pcd, _ = accum_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
            # 2) (Optional) Voxel downsample
            # accum_pcd = accum_pcd.voxel_down_sample(voxel_size=0.01)

            # Save .ply for debug
            ply_name = f"accum_{int(time.time())}.ply"
            out_path = output_dir / ply_name
            o3d.io.write_point_cloud(str(out_path), accum_pcd)
            print(f"✅ Saved {len(accum_pcd.points)} pts => {out_path}")

            # Poisson
            print("🌀 Poisson surface reconstruction with depth=10")
            accum_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                accum_pcd, depth=10, n_threads=-1
            )

            # Remove very low density => removes floating pieces
            low_thresh = np.quantile(densities, 0.02)  # remove bottom 2%
            print(f"Removing mesh vertices with density < {low_thresh:.3f}")
            vertices_to_remove = densities < low_thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.compute_vertex_normals()

            # Transfer color from pcd => mesh
            if accum_pcd.has_colors():
                transfer_pointcloud_color_to_mesh(accum_pcd, mesh)
            else:
                print("⚠️ accum_pcd has no color, skipping color xfer")

            # (Optional) Crop bounding box if needed
            # bbox = accum_pcd.get_axis_aligned_bounding_box()
            # mesh = mesh.crop(bbox)

            # Convert for Meshcat
            v = np.asarray(mesh.vertices)
            f = np.asarray(mesh.triangles)
            vc = np.asarray(mesh.vertex_colors)
            geom = g.TriangularMeshGeometry(v, f, color=vc)
            mat = g.MeshPhongMaterial(vertexColors=True)
            obj = g.Mesh(geom, mat)

            # Show in meshcat
            vis["accum_poisson_mesh"].set_object(obj)
            print(f"🎨 Displayed mesh with {len(v)} vertices, {len(f)} faces in Meshcat.")

        processed.update(batch)
        time.sleep(refresh_rate)

def main():
    parser = argparse.ArgumentParser(description="Poisson mesh from submaps + color, with cleaning steps.")
    parser.add_argument("--submap_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_ckpts", type=int, default=5)
    parser.add_argument("--refresh_rate", type=float, default=2.0)
    args = parser.parse_args()

    live_merge_ckpts(args.submap_dir, args.output_dir, args.batch_ckpts, args.refresh_rate)

if __name__ == "__main__":
    main()
