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

    mean = (dist_xy.mean() + dist_yx.mean()) / 2
    std = (dist_xy.std() + dist_yx.std()) / 2
    hd = (dist_xy.max() + dist_yx.max()) / 2
    hd95 = (torch.quantile(dist_xy, q=0.95) + torch.quantile(dist_yx, q=0.95)) / 2

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

    mean = (dist_pts_to_mesh.mean() + dist_mesh_to_points.mean()) / 2

    std = (dist_pts_to_mesh.std() + dist_mesh_to_points.std()) / 2
    hd = (dist_pts_to_mesh.max() + dist_mesh_to_points.max()) / 2
    hd95 = (torch.quantile(dist_pts_to_mesh, q=0.95) + torch.quantile(dist_mesh_to_points, q=0.95)) / 2

    return mean, std, hd, hd95, points


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


if __name__ == '__main__':
    vx = torch.randn(16, 100, 3)
    tx = torch.randint(100, (16, 200, 3))
    vy = torch.randn(16, 160, 3)
    ty = torch.randint(160, (16, 180, 3))
    print(batch_assd(vx, tx, vy, ty))
