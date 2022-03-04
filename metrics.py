import open3d as o3d
import torch

from surface_fitting import point_surface_distance
from utils import create_o3d_mesh


def assd(verts_x: torch.Tensor, faces_x: torch.Tensor, verts_y: torch.Tensor, faces_y: torch.Tensor):
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


def ssd(mesh_x: o3d.geometry.TriangleMesh, mesh_y: o3d.geometry.TriangleMesh):
    """ Symmetric surface distance between batches of meshes x and y, averaged over points.

        :param mesh_x: first mesh
        :param mesh_y: second mesh
        :return: Mean distance, standard deviation per mesh, Hausdorff and 95th quantile distance
        """
    dist_xy = point_surface_distance(query_points=mesh_x.vertices, trg_points=mesh_y.vertices, trg_tris=mesh_y.triangles)
    dist_yx = point_surface_distance(query_points=mesh_y.vertices, trg_points=mesh_x.vertices, trg_tris=mesh_x.triangles)

    mean = (dist_xy.mean() + dist_yx.mean()) / 2
    std = (dist_xy.std() + dist_yx.std()) / 2
    hd = (dist_xy.max() + dist_yx.max()) / 2
    hd95 = (torch.quantile(dist_xy, q=0.95) + torch.quantile(dist_yx, q=0.95)) / 2

    return mean, std, hd, hd95


if __name__ == '__main__':
    vx = torch.randn(16, 100, 3)
    tx = torch.randint(100, (16, 200, 3))
    vy = torch.randn(16, 160, 3)
    ty = torch.randint(160, (16, 180, 3))
    print(assd(vx, tx, vy, ty))
