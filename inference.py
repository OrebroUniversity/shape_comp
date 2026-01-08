#!/usr/bin/env python3

from models.combined_model import CombinedModel
from utils import mesh

import numpy as np
import open3d as o3d
import torch
from torch.nn import functional as F
from einops import reduce
import cv2

import time
import json
from typing import Tuple
import os

@torch.no_grad()
def predict_sdf(model: CombinedModel, partial_pcd: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts the signed distance function (SDF) and plane features with a trained model
    for a given partial point cloud.

    Parameters
    ----------
    model : CombinedModel
        The model to use for prediction.
    partial_pcd : torch.Tensor
        The partial point cloud to predict on, shape (N, 3).
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - Predicted SDF of shape (num_samples, 1024, 1).
        - Predicted plane features of shape (num_samples, 768, 64, 64).
    """

    denoised_latents = model.diffusion_model.generate_from_pc(partial_pcd, batch=num_samples)
    plane_features = model.vae_model.decode(denoised_latents)
    pred_sdf = model.sdf_model.forward_with_plane_features(plane_features, partial_pcd.repeat(num_samples, 1, 1))
    return pred_sdf, plane_features

def post_process_recon_mesh(m: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    # Keep biggest cluster: https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Connected-components
    triangle_clusters, cluster_n_triangles, cluster_area = (m.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    num_clusters = len(cluster_n_triangles)
    if num_clusters > 1:
        largest_cluster_id = np.argmax(cluster_area)
        triangles_to_remove = triangle_clusters != largest_cluster_id
        m.remove_triangles_by_mask(triangles_to_remove)

    # Mesh filtering: https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-filtering
    m = m.filter_smooth_simple(number_of_iterations=1)

    m.compute_vertex_normals()
    return m

def perform_icp(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform ICP registration between source and target point clouds.
    """
    return o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=0.01,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

def align_recon_with_pcd(recon_mesh: o3d.geometry.TriangleMesh, partial_pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, float]:
    """
    Align the reconstructed mesh with the partial point cloud.

    `recon_mesh` and `partial_pcd` get mutated.
    """
    if len(partial_pcd.points) > 1024:
        partial_pcd = partial_pcd.farthest_point_down_sample(1024)

    recon_mesh = post_process_recon_mesh(recon_mesh)
    recon_pcd = recon_mesh.sample_points_uniformly(number_of_points=len(partial_pcd.points))

    icp_sol = perform_icp(source_pcd=recon_pcd, target_pcd=partial_pcd)

    recon_pcd = recon_pcd.transform(icp_sol.transformation)
    recon_mesh = recon_mesh.transform(icp_sol.transformation)

    return (recon_mesh, recon_pcd, icp_sol.fitness)

if __name__ == "__main__":
    dev_count = torch.cuda.device_count()
    print(f"GPUs found: {dev_count}")
    for i in range(dev_count):
        gpu_name = torch.cuda.get_device_name(i)
        mem_in_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"\t - {gpu_name}: VRAM {mem_in_gb:.2f} GB,")
    
    mesh_create_batch_size = 2**16  # adjust based on your GPU memory

    ckpt_path: str = "shape_comp.ckpt"
    assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} does not exist."
    all_obj_ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint {ckpt_path} has weights for objects:")
    for obj in all_obj_ckpt.keys():
        print(f" - {obj}")

    obj_name: str = "can"

    specs_filepath: str = "config/specs.json"
    specs = json.load(open(specs_filepath))
    print(f"================\n{json.dumps(specs, indent=4)}\n================")

    print(f"Loading model weights for object {obj_name} from checkpoint at {ckpt_path} ..")
    tic = time.time()
    model = CombinedModel(specs=specs).cuda()
    model.load_state_dict(all_obj_ckpt[f"{obj_name}_ckpt"], strict=True)
    toc = time.time()
    duration = toc - tic
    print(f"Model ready in {duration:.2f}s")

    partial_pcd_fp: str = "example/can_partial.ply"
    assert os.path.isfile(partial_pcd_fp), f"Partial point cloud {partial_pcd_fp} does not exist."

    partial_pcd = o3d.io.read_point_cloud(partial_pcd_fp)
    if len(partial_pcd.points) == 0:
        raise ValueError(f"Partial PCD has 0 points: {partial_pcd_fp}")
    elif len(partial_pcd.points) > 1024:
        print(f"Partial PCD has {len(partial_pcd.points)} points, downsampling to 1024 points")
        partial_pcd = partial_pcd.farthest_point_down_sample(num_samples=1024)

    partial_pcd_original_center = partial_pcd.get_center()
    origin = np.array([0.0, 0.0, 0.0])
    partial_pcd = partial_pcd.translate(origin, relative=False)
    partial_pcd_extents = partial_pcd.get_oriented_bounding_box(robust=True).extent
    diagonal_length = np.linalg.norm(partial_pcd_extents)
    partial_pcd_scaling_factor = 1.0 / diagonal_length
    partial_pcd = partial_pcd.scale(scale=partial_pcd_scaling_factor, center=origin)
    partial_pcd_tensor = torch.tensor(np.asarray(partial_pcd.points), dtype=torch.float32)

    pred_sdf, plane_feature = predict_sdf(model, partial_pcd_tensor.cuda(), num_samples=1)

    # refer to section 5.5 Ablation Experiments to know more about consistency loss and why the comparison is against torch.zeros_like()
    sdf_loss = F.l1_loss(pred_sdf, torch.zeros_like(pred_sdf), reduction='none')
    consistency_loss = reduce(sdf_loss, 'b ... -> b', 'mean', b = sdf_loss.shape[0]) # one value per generated sample

    threshold = 0.1
    print(f"Consistency loss = {consistency_loss.item()} (threshold = {threshold})")
    assert consistency_loss.item() <= threshold, f"Consistency loss {consistency_loss.item()} exceeds threshold {threshold}"

    print("Creating mesh ..")
    tic = time.time()
    ply_file, sdf_diagnostics = mesh.create_mesh(
        model.sdf_model,
        plane_feature,
        grid_resolution=128,
        max_batch=mesh_create_batch_size,
        from_plane_features=True,
    )
    toc = time.time()
    duration = toc - tic
    print(f"Mesh created in {duration:.2f}s")

    if cv2.imwrite(f"sdf_diagnostics_{obj_name}.png", (sdf_diagnostics).astype(np.uint8)):
        print("Saved SDF diagnostics")
    else:
        print("Failed to save SDF diagnostics")

    recon_mesh = mesh.plydata_to_open3d_mesh(ply_file)
    recon_mesh = recon_mesh.scale(scale=(1.0 / partial_pcd_scaling_factor), center=recon_mesh.get_center())
    recon_mesh = recon_mesh.translate(partial_pcd_original_center, relative=False)
    partial_pcd = partial_pcd.translate(partial_pcd_original_center, relative=False)
    partial_pcd = partial_pcd.scale(scale=(1.0 / partial_pcd_scaling_factor), center=partial_pcd_original_center)

    aligned_recon_mesh, aligned_recon_pcd, icp_fitness = align_recon_with_pcd(recon_mesh, partial_pcd)

    # visualize in Open3d
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
    scene_pcd_fp = "example/scene_pcd.ply"
    scene_pcd = o3d.io.read_point_cloud(scene_pcd_fp)
    o3d.visualization.draw_geometries([aligned_recon_mesh, scene_pcd, origin_frame])

    save_path = os.path.join(os.path.dirname(partial_pcd_fp), f"{obj_name}_completed.ply")
    if o3d.io.write_triangle_mesh(save_path, aligned_recon_mesh, write_ascii=True):
        print(f"Mesh saved to {save_path}")
    else:
        print(f"Failed to save mesh to {save_path}")

    print("EXIT")
