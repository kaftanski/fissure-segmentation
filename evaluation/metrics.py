from typing import Sequence

import numpy as np
import open3d as o3d
import torch
from numpy.typing import ArrayLike

from utils.general_utils import mask_to_points


def point_surface_distance(query_points: ArrayLike, trg_points: ArrayLike, trg_tris: ArrayLike) -> torch.Tensor:
    """ Parallel unsigned distance computation from N query points to a target triangle mesh using Open3d.

    :param query_points: query points for distance computation. ArrayLike of shape (Nx3)
    :param trg_points: vertices of the target mesh. ArrayLike of shape (Vx3)
    :param trg_tris: shared edge triangle index list of the target mesh. ArrayLike of shape (Tx3)
    :return: euclidean distance from every input point to the closest point on the target mesh. Tensor of shape (N)
    """
    # construct ray casting scene with target mesh in it
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertex_positions=np.array(trg_points, dtype=np.float32), triangle_indices=np.array(trg_tris, dtype=np.uint32))  # we do not need the geometry ID for mesh

    # distance computation
    dist = scene.compute_distance(np.array(query_points, dtype=np.float32))
    return torch.utils.dlpack.from_dlpack(dist.to_dlpack())


def assd(mesh_x: o3d.geometry.TriangleMesh, mesh_y: o3d.geometry.TriangleMesh):
    """ Symmetric surface distance between batches of meshes x and y, averaged over points.

    :param mesh_x: first mesh
    :param mesh_y: second mesh
    :return: Mean distance, standard deviation of distances, Hausdorff and 95th quantile distance
    """
    if len(mesh_x.vertices) == 0 or len(mesh_y.vertices) == 0:
        return (torch.tensor(float('NaN')),) * 4

    dist_xy = point_surface_distance(query_points=mesh_x.vertices, trg_points=mesh_y.vertices, trg_tris=mesh_y.triangles)
    dist_yx = point_surface_distance(query_points=mesh_y.vertices, trg_points=mesh_x.vertices, trg_tris=mesh_x.triangles)

    mean, std, hd, hd95 = _symmetric_point_distances(dist_xy, dist_yx)
    return mean, std, hd, hd95


def label_mesh_assd(labelmap: torch.Tensor, mesh: o3d.geometry.TriangleMesh, spacing: Sequence[float] = (1., 1., 1.)):
    """

    :param labelmap: binary labelmap (all non-zero elements are considered foreground)
    :param mesh: triangle mesh
    :param spacing: the image spacing for the labelmap (used to convert pixel into world coordinates
    :return: Mean distance, standard deviation of distances, Hausdorff and 95th quantile distance and the extracted points
    """
    # compute point cloud from foreground pixels in labelmap
    points = mask_to_points(labelmap, spacing)
    return *pseudo_symmetric_point_to_mesh_distance(points, mesh), points


def pseudo_symmetric_point_to_mesh_distance(points: ArrayLike, mesh: o3d.geometry.TriangleMesh):
    """

    :param points: points to query distance to mesh for
    :param mesh: reference mesh
    :return: Mean distance, standard deviation of distances, Hausdorff and 95th quantile distance
    """
    # distance from labelmap (point cloud) to mesh
    dist_pts_to_mesh = point_surface_distance(query_points=points, trg_points=mesh.vertices, trg_tris=mesh.triangles)

    # chamfer distance from mesh (samples from point cloud) to labelmap point cloud
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points.numpy().astype(np.float32)))
    n_samples = 10**6
    mesh_sampled_pc = mesh.sample_points_uniformly(n_samples)
    dist_mesh_to_points = torch.from_numpy(np.asarray(mesh_sampled_pc.compute_point_cloud_distance(pcd)))
    print(dist_pts_to_mesh.mean(), dist_mesh_to_points.mean())

    mean, std, hd, hd95 = _symmetric_point_distances(dist_pts_to_mesh, dist_mesh_to_points)
    return mean, std, hd, hd95


def label_label_assd(labelmap1, labelmap2, spacing: Sequence[float] = (1., 1., 1.)):
    # compute point clouds from foreground pixels in labelmap
    points1 = mask_to_points(labelmap1, spacing)
    points2 = mask_to_points(labelmap2, spacing)

    pc1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points1.numpy().astype(np.float32)))
    pc2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points2.numpy().astype(np.float32)))

    # compute chamfer distances
    dist_12 = torch.from_numpy(np.asarray(pc1.compute_point_cloud_distance(pc2)))
    dist_21 = torch.from_numpy(np.asarray(pc2.compute_point_cloud_distance(pc1)))
    print(dist_12.mean(), dist_21.mean())

    mean, std, hd, hd95 = _symmetric_point_distances(dist_12, dist_21)
    return mean, std, hd, hd95


def _symmetric_point_distances(dist_points1, dist_points2):
    mean = (dist_points1.mean() + dist_points2.mean()) / 2
    std = (dist_points1.std() + dist_points2.std()) / 2
    hd = (dist_points1.max() + dist_points2.max()) / 2
    hd95 = (torch.quantile(dist_points1, q=0.95) + torch.quantile(dist_points2, q=0.95)) / 2
    return mean, std, hd, hd95


def batch_assd(verts_x: torch.Tensor, faces_x: torch.Tensor, verts_y: torch.Tensor, faces_y: torch.Tensor):
    """ Symmetric surface distance between batches of meshes x and y, averaged over points.

    :param verts_x: vertices of mesh x. Shape: (BxV1x3)
    :param faces_x: triangle list of mesh x. Shape: (BxT1x3)
    :param verts_y: vertices of mesh y. Shape: (BxV2x3)
    :param faces_y: triangle list of mesh y. Shape: (BxT2x3)
    :return: Mean distance, standard deviation per mesh, Hausdorff and 95th quantile distance
    """
    batch = verts_x.shape[0]

    mean = torch.zeros(batch)
    std = torch.zeros_like(mean)
    hd = torch.zeros_like(mean)
    hd95 = torch.zeros_like(mean)
    for i in range(batch):
        dist_xy = point_surface_distance(query_points=verts_x[i], trg_points=verts_y[i], trg_tris=faces_y[i])
        dist_yx = point_surface_distance(query_points=verts_y[i], trg_points=verts_x[i], trg_tris=faces_x[i])
        mean[i] += (dist_xy.mean() + dist_yx.mean()) / 2
        std[i] += (dist_xy.std() + dist_yx.std()) / 2
        hd[i] += (dist_xy.max() + dist_yx.max()) / 2
        hd95[i] += (torch.quantile(dist_xy, q=0.95) + torch.quantile(dist_yx, q=0.95)) / 2

    return mean.mean(), std.mean(), hd.mean(), hd95.mean()


def batch_dice(prediction, target, n_labels):
    labels = torch.arange(n_labels)
    dice = torch.zeros(prediction.shape[0], n_labels).to(prediction.device)

    pred_flat = prediction.flatten(start_dim=1)
    targ_flat = target.flatten(start_dim=1)
    for l in labels:
        label_pred = pred_flat == l
        label_target = targ_flat == l
        dice[:, l] = 2 * (label_pred * label_target).sum(-1) / (label_pred.sum(-1) + label_target.sum(-1) + 1e-8)

    return dice.mean(0).cpu()


def binary_recall(prediction, target):
    binary_pred = (prediction != 0).flatten(start_dim=1)
    binary_targ = (target != 0).flatten(start_dim=1)
    return ((binary_pred * binary_targ).sum(-1) + 1e-8) / (binary_targ.sum(-1) + 1e-8)


def binary_precision(prediction, target):
    binary_pred = (prediction != 0).flatten(start_dim=1)
    binary_targ = (target != 0).flatten(start_dim=1)
    return ((binary_pred * binary_targ).sum(-1) + 1e-8) / (binary_pred.sum(-1) + 1e-8)
