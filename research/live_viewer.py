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

from src.entities.gaussian_model import GaussianModel
from src.entities.arguments import OptimizationParams
from src.utils.utils import torch2np
from meshcat.geometry import TriangularMeshGeometry, Mesh, MeshPhongMaterial


def transfer_pointcloud_color_to_mesh(o3d_pcd, o3d_mesh):
    """
    For each vertex in 'o3d_mesh', find the nearest neighbor in 'o3d_pcd'
    and copy its color over. This ensures the mesh has vertex_colors that
    match the input point cloud's color distribution.
    """
    # Build a KD-tree for the pcd
    pcd_tree = o3d.geometry.KDTreeFlann(o3d_pcd)
    mesh_vertex_colors = []

    pcd_points = np.asarray(o3d_pcd.points)
    pcd_colors = np.asarray(o3d_pcd.colors)

    # For each vertex in the mesh, sample the nearest color
    for v in o3d_mesh.vertices:
        # k = 1 => find one nearest neighbor
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 1)
        nearest_idx = idx[0]
        mesh_vertex_colors.append(pcd_colors[nearest_idx])

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_vertex_colors))


###############################################################################
# Main logic for reading .ckpt submaps, combining them, and building a
# Poisson-reconstructed mesh with color, then showing in Meshcat.
###############################################################################
def maprange(a, b, s):
    """ Maps values in s from range a to range b. """
    (a1, a2), (b1, b2) = a, b
    return b1 + ((s - a1) * (b2 - b1) / (a2 - a1))


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

        # handle shape (N,3,1) or (N,1,3)
        if f_dc.ndim == 3 and f_dc.shape[2] == 1:
            f_dc = f_dc[:, :, 0]
        elif f_dc.ndim == 3 and f_dc.shape[1] == 1:
            f_dc = f_dc[:, 0, :]

        # Option A: simple clamp if values are [0,1] or [-1,+1]
        color = np.clip(f_dc, 0.0, 1.0)

        # Option B: if you know it ranges [-3,+3], do this:
        # color = maprange((-3,3), (0,1), f_dc)
        # color = np.clip(color, 0.0, 1.0)

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

    # only use color if all submaps had color
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
    """
    Accumulate points from every batch, Poisson-reconstruct a mesh, then
    copy the original colors to the mesh's vertices, and display in Meshcat
    with lighting (Phong shading + per-vertex color).
    """
    print("🌐 Opening Meshcat viewer. Watching for .ckpt in:", submap_dir)
    vis = meshcat.Visualizer().open()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = set()
    accum_pcd = o3d.geometry.PointCloud()

    while True:
        all_ckpts = set(sorted(glob.glob(os.path.join(submap_dir, "*.ckpt"))))
        new_ckpts = list(all_ckpts - processed)
        if not new_ckpts:
            time.sleep(refresh_rate)
            continue

        # Take 'batch_ckpts' from new_ckpts
        batch = sorted(new_ckpts)[:batch_ckpts]
        print(f"📊 Merging batch of {len(batch)} ckpt files...")

        pcd_merged = merge_ckpt_batch_into_cloud(batch)
        if pcd_merged is None or len(pcd_merged.points) == 0:
            print("⚠️ Merged batch is empty. Skipping accumulate/display.")
        else:
            # Merge new pcd into the global accum
            if len(accum_pcd.points) == 0:
                accum_pcd = pcd_merged
            else:
                # combine geometry
                accum_points = np.concatenate(
                    (np.asarray(accum_pcd.points), np.asarray(pcd_merged.points)),
                    axis=0
                )
                if accum_pcd.has_colors() and pcd_merged.has_colors():
                    accum_cols = np.asarray(accum_pcd.colors)
                    new_cols = np.asarray(pcd_merged.colors)
                    combined_cols = np.concatenate((accum_cols, new_cols), axis=0)

                    new_accum = o3d.geometry.PointCloud()
                    new_accum.points = o3d.utility.Vector3dVector(accum_points)
                    new_accum.colors = o3d.utility.Vector3dVector(combined_cols)
                    accum_pcd = new_accum
                else:
                    new_accum = o3d.geometry.PointCloud()
                    new_accum.points = o3d.utility.Vector3dVector(accum_points)
                    accum_pcd = new_accum

            # Write the updated global pcd to .ply
            ply_name = f"accum_{int(time.time())}.ply"
            out_file = output_dir / ply_name
            o3d.io.write_point_cloud(str(out_file), accum_pcd)
            print(f"✅ Wrote ACCUMULATED cloud with {len(accum_pcd.points)} total points to {out_file}")

            # Build a mesh from accum_pcd so we can do shading
            accum_pcd.estimate_normals()
            print("Running Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                accum_pcd, depth=8
            )

            # Remove outliers
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            mesh.compute_vertex_normals()

            print(f" Reconstructed mesh has {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")

            # Now copy color from the point cloud to the mesh
            if accum_pcd.has_colors():
                transfer_pointcloud_color_to_mesh(accum_pcd, mesh)
            else:
                print("No colors in accum_pcd, so mesh will be uncolored")

            # Convert O3D mesh -> TriangularMeshGeometry for Meshcat
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vertex_colors = np.asarray(mesh.vertex_colors)  # shape (N,3)

            # Build the geometry with color
            meshcat_geom = g.TriangularMeshGeometry(vertices, faces, color=vertex_colors)

            # Set up a Phong material that uses vertex colors
            material = g.MeshPhongMaterial(vertexColors=True)
            # For extra shading control, you can do e.g.:
            # material.reflectivity = 0.5
            # material.wireframe = False
            # material.opacity = 1.0
            # etc.

            # Build the final Mesh
            meshcat_obj = g.Mesh(meshcat_geom, material)

            # Display in Meshcat
            vis["accum_ckpts_cloud"].set_object(meshcat_obj)
            print(f"Updated Meshcat with colored mesh of {len(mesh.vertices)} vertices.")

        # Mark processed
        processed.update(batch)
        time.sleep(refresh_rate)


def main():
    parser = argparse.ArgumentParser(
        description="Live merges .ckpt submaps in an accumulating point cloud, then shows a Poisson mesh in Meshcat."
    )
    parser.add_argument("--submap_dir", required=True,
                        help="Directory containing your submap .ckpt files.")
    parser.add_argument("--output_base_dir", required=True,
                        help="Base directory under which a subfolder will be created to store outputs.")
    parser.add_argument("--batch_ckpts", type=int, default=5,
                        help="Number of new ckpt files to merge per iteration.")
    parser.add_argument("--refresh_rate", type=float, default=2.0,
                        help="Seconds to wait between scanning for new .ckpt files.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    full_output_dir = Path(args.output_base_dir) / f"accum_{timestamp}"
    full_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Output will be stored in: {full_output_dir}")
    live_merge_ckpts(args.submap_dir, full_output_dir, args.batch_ckpts, args.refresh_rate)


if __name__ == "__main__":
    main()