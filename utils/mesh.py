#!/usr/bin/env python3

from models.sdf_model import SdfModel

import numpy as np
import plyfile
import skimage.measure
import torch
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
import cv2

from typing import List, Tuple, Optional
import os


# grid_resolution: 256 is typically sufficient 
# max batch: as large as GPU memory will allow
# shape_feature is either point cloud, mesh_idx (neuralpull), or generated latent code (deepsdf)
def create_mesh(
    model: SdfModel,
    shape_feature: torch.Tensor,
    grid_resolution: int = 256,
    max_batch: int = 1000000,
    level_set: float = 0.0,
    from_plane_features: bool = False,
) -> Tuple[Optional[plyfile.PlyData], np.ndarray]:

    model.eval()

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (grid_resolution - 1)
    cube = create_cube(grid_resolution)
    cube_points = cube.shape[0]

    head = 0
    while head < cube_points:
        query = cube[head : min(head + max_batch, cube_points), 0:3].unsqueeze(0)
        
        if from_plane_features:
            pred_sdf = model.forward_with_plane_features(shape_feature.cuda(), query.cuda()).detach().cpu()
        else:
            pred_sdf = model(shape_feature.cuda(), query.cuda()).detach().cpu()

        cube[head : min(head + max_batch, cube_points), 3] = pred_sdf.squeeze()

        head += max_batch

    sdf_values = cube[:, 3] 
    sdf_values = sdf_values.reshape(grid_resolution, grid_resolution, grid_resolution) 

    return convert_sdf_samples_to_ply(
        sdf_values.data,
        voxel_origin,
        voxel_size,
        level_set
    )


# create cube from (-1,-1,-1) to (1,1,1) and uniformly sample points for marching cube
def create_cube(N):

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long().float() / N) % N
    samples[:, 0] = ((overall_index.long().float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    return samples



def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor: torch.Tensor,
    voxel_grid_origin: List[float],
    voxel_size: float,
    level_set: float = 0.0,
) -> Tuple[Optional[plyfile.PlyData], np.ndarray]:
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :level_set: the level set to extract the mesh at

    This function adapted from: https://github.com/RobotLocomotion/spartan
    (https://github.com/RobotLocomotion/spartan/blob/854b26e3af75910ef57b874db7853abd4249543e/src/catkin_projects/fusion_server/src/fusion_server/tsdf_fusion.py#L126-L224)
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    sdf_diagnostics = create_sdf_diagnostics_array(numpy_3d_sdf_tensor)

    sdf_min = np.min(numpy_3d_sdf_tensor)
    if sdf_min > 0:
        print(f"[WARN]: SDF values are all positive, minimum SDF is {sdf_min:.6f}")

    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print(f"[ERROR] marching cubes failed: {e}")

        print("DIAGNOSTICS:")
        print(f"\t Data min: {sdf_min:.6f}")
        print(f"\t Data max: {np.max(numpy_3d_sdf_tensor):.6f}")

        values_near_zero = np.sum((numpy_3d_sdf_tensor >= -0.05) & (numpy_3d_sdf_tensor <= 0.05))
        print(f"\t Voxels near zero (-0.05 to 0.05): {values_near_zero}")

        zero_crossings = np.sum((numpy_3d_sdf_tensor[:-1, :, :] * numpy_3d_sdf_tensor[1:, :, :]) < 0) + \
                        np.sum((numpy_3d_sdf_tensor[:, :-1, :] * numpy_3d_sdf_tensor[:, 1:, :]) < 0) + \
                        np.sum((numpy_3d_sdf_tensor[:, :, :-1] * numpy_3d_sdf_tensor[:, :, 1:]) < 0)
        if zero_crossings == 0:
            print("\t No zero-crossings found")

        print(f"\t Contains NaN: {np.any(np.isnan(numpy_3d_sdf_tensor))}")
        print(f"\t Contains inf: {np.any(np.isinf(numpy_3d_sdf_tensor))}")
        print(f"\t Number of finite values: {np.sum(np.isfinite(numpy_3d_sdf_tensor))} (out of a total of {numpy_3d_sdf_tensor.size})")

        return None, sdf_diagnostics

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces], text=True)

    return ply_data, sdf_diagnostics

def save_out_pcd(points: torch.Tensor, dirpath: str) -> bool:
    os.makedirs(dirpath, exist_ok=True)
    pcd_filepath = os.path.join(dirpath, "input_pcd.ply")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy().squeeze())
    return o3d.io.write_point_cloud(pcd_filepath, pcd, write_ascii=True)

def plydata_to_open3d_mesh(ply_data: plyfile.PlyData) -> o3d.geometry.TriangleMesh:
    vertices, faces = get_plydata_vertices_and_faces(ply_data)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh

def plydata_to_trimesh(ply_data: plyfile.PlyData) -> trimesh.Trimesh:
    vertices, faces = get_plydata_vertices_and_faces(ply_data)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh

def get_plydata_vertices_and_faces(ply_data: plyfile.PlyData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract vertex and face information from a PlyData object.

    :param ply_data: The PlyData object to extract data from.
    :return: A tuple containing the vertices and faces as numpy arrays.
    """
    vertex_data = ply_data['vertex']
    vertices = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)

    face_data = ply_data['face']
    faces = np.vstack(face_data['vertex_indices'])

    return vertices, faces


def create_sdf_diagnostics_array(sdf_data: np.ndarray) -> np.ndarray:
    sdf_min = float(np.min(sdf_data))
    iso_surface_lvl: float = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    ax.hist(sdf_data.flatten(), bins=50, alpha=0.7)
    ax.axvline(x=iso_surface_lvl, color='red', linestyle='--', label=f'Iso-surface ({iso_surface_lvl})')
    ax.axvline(x=sdf_min, color='green', linestyle='--', label=f'Min ({sdf_min:.3f})')
    ax.set_xlabel('Voxel values')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_title('SDF Data Distribution')

    # Middle slice
    center_slice = sdf_data[sdf_data.shape[0] // 2, :, :]
    im = axes[1].imshow(center_slice, cmap='viridis')
    cbar = fig.colorbar(im, ax=axes[1], label='Value')
    cbar.ax.axhline(y=iso_surface_lvl, color='red', linewidth=2)
    contours = axes[1].contour(center_slice, levels=[iso_surface_lvl], colors='red', linewidths=2)
    axes[1].clabel(contours, inline=True, fontsize=8, fmt='%g')
    axes[1].set_title('Middle slice of volume')

    fig.tight_layout()
    fig.canvas.draw()

    # Convert RGBA → BGR for OpenCV: https://stackoverflow.com/a/77714706/6010333
    img_rgba = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

    plt.close(fig)  # clean up

    # cv2.imshow('Image', img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_bgr
