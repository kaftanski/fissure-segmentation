import numpy as np
from typing import Sequence

import open3d as o3d
import torch

from utils import mask_to_points
from data_processing.surface_fitting import point_surface_distance


def assd(mesh_x: o3d.geometry.TriangleMesh, mesh_y: o3d.geometry.TriangleMesh):
    """ Symmetric surface distance between batches of meshes x and y, averaged over points.

        :param mesh_x: first mesh
        :param mesh_y: second mesh
        :return: Mean distance, standard deviation of distances, Hausdorff and 95th quantile distance
        """
    dist_xy = point_surface_distance(query_points=mesh_x.vertices, trg_points=mesh_y.vertices, trg_tris=mesh_y.triangles)
    dist_yx = point_surface_distance(query_points=mesh_y.vertices, trg_points=mesh_x.vertices, trg_tris=mesh_x.triangles)

    mean, std, hd, hd95 = _symmetric_point_distances(dist_xy, dist_yx)
    return mean, std, hd, hd95


def label_mesh_assd(labelmap: torch.Tensor, mesh: o3d.geometry.TriangleMesh, spacing: Sequence[float] = (1., 1., 1.)):
    """

    :param labelmap: binary labelmap (all non-zero elements are considered foreground)
    :param mesh: triangle mesh
    :param spacing: the image spacing for the labelmap (used to convert pixel into world coordinates
    :return: Mean distance, standard deviation of distances, Hausdorff and 95th quantile distance
    """
    # compute point cloud from foreground pixels in labelmap
    points = mask_to_points(labelmap, spacing)

    # distance from labelmap (point cloud) to mesh
    dist_pts_to_mesh = point_surface_distance(query_points=points, trg_points=mesh.vertices, trg_tris=mesh.triangles)

    # chamfer distance from mesh (samples from point cloud) to labelmap point cloud
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points.numpy().astype(np.float32)))
    n_samples = 10**6
    mesh_sampled_pc = mesh.sample_points_uniformly(n_samples)
    dist_mesh_to_points = torch.from_numpy(np.asarray(mesh_sampled_pc.compute_point_cloud_distance(pcd)))
    print(dist_pts_to_mesh.mean(), dist_mesh_to_points.mean())

    mean, std, hd, hd95 = _symmetric_point_distances(dist_pts_to_mesh, dist_mesh_to_points)
    return mean, std, hd, hd95, points


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
    return (binary_pred * binary_targ).sum(-1) / binary_targ.sum(-1)


def binary_precision(prediction, target):
    binary_pred = (prediction != 0).flatten(start_dim=1)
    binary_targ = (target != 0).flatten(start_dim=1)
    return (binary_pred * binary_targ).sum(-1) / binary_pred.sum(-1)


if __name__ == '__main__':
    vx = torch.randn(16, 100, 3)
    tx = torch.randint(100, (16, 200, 3))
    vy = torch.randn(16, 160, 3)
    ty = torch.randint(160, (16, 180, 3))
    print(batch_assd(vx, tx, vy, ty))