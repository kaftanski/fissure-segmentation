import warnings
from typing import Sequence

import numpy as np
import open3d as o3d
import os

import torch
from numpy.typing import ArrayLike
from torch.nn import functional as F
import SimpleITK as sitk

from image_ops import sitk_image_to_tensor, resample_equal_spacing


def pairwise_dist(x):
    """ distance from each point in x to itself and others

    :param x: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    xTx = torch.bmm(x, x.transpose(2, 1))
    dist = xx - 2.0 * xTx + xx.transpose(2, 1)
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0  # ensure diagonal is 0
    return dist


def pairwise_dist2(x, y):
    """ distance from each point in x to its corresponding point in y

    :param x: point cloud batch of shape (B x N x 3)
    :param y: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    yy = (y ** 2).sum(2, keepdim=True)
    xTy = torch.bmm(x, y.transpose(2, 1))
    dist = xx - 2.0 * xTy + yy.transpose(2, 1)
    return dist


def save_points(points: torch.Tensor, labels: torch.Tensor, path: str, case: str, sequence: str = 'fixed'):
    torch.save(points.cpu(), os.path.join(path, f'{case}_coords_{sequence}.pth'))
    torch.save(labels.cpu(), os.path.join(path, f'{case}_labels_{sequence}.pth'))


def load_points(path: str, case: str, sequence: str = 'fixed', feat: str = None):
    return torch.load(os.path.join(path, f'{case}_coords_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_fissures_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_lobes_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_{feat}_{sequence}.pth'), map_location='cpu') if feat is not None \
               else None


def filter_1d(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6, )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(5, )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
                    weight.view(view)).view(B, C, D, H, W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter_1d(img, weight, 0)
    img = filter_1d(img, weight, 1)
    img = filter_1d(img, weight, 2)

    return img


def create_o3d_mesh(verts: ArrayLike, tris: ArrayLike):
    verts = o3d.utility.Vector3dVector(np.array(verts, dtype=np.float32))
    tris = o3d.utility.Vector3iVector(np.array(tris, dtype=np.uint32))
    return o3d.geometry.TriangleMesh(vertices=verts, triangles=tris)


def kpts_to_grid(kpts_world, shape, align_corners=None):
    """ expects points in xyz format from a tensor with given shape

    :param kpts_world:
    :param shape:
    :param align_corners:
    :return: points in range [-1, 1]
    """
    device = kpts_world.device
    D, H, W = shape

    kpts_pt_ = (kpts_world / (torch.tensor([W, H, D], device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor([W, H, D], device=device)

    return kpts_pt_


def kpts_to_world(kpts_pt, shape, align_corners=None):
    """  expects points in xyz format

    :param kpts_pt:
    :param shape:
    :param align_corners:
    :return: points in xyz format transformed into the shape
    """
    device = kpts_pt.device
    D, H, W = shape

    if not align_corners:
        kpts_pt /= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor([W, H, D], device=device)
    kpts_world_ = (((kpts_pt + 1) / 2) * (torch.tensor([W, H, D], device=device) - 1))

    return kpts_world_


def mask_to_points(mask: torch.Tensor, spacing: Sequence[float] = (1., 1., 1.)):
    points_img = torch.nonzero(mask).flip(-1)
    points_world = points_img * torch.tensor(spacing, device=mask.device)
    return points_world


def mask_out_verts_from_mesh(mesh: o3d.geometry.TriangleMesh, mask: torch.Tensor, spacing: torch.Tensor):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)) / spacing.cpu()
    vertices = vertices.floor().long()

    # prevent index out of bounds by removing vertices out of range
    for d in range(len(mask.shape)):
        vertices[:, d] = torch.clamp(vertices[:, d], max=mask.shape[len(mask.shape)-1-d] - 1)

    # remove vertices outside the lung mask
    remove_verts = torch.ones(vertices.shape[0], dtype=torch.bool)
    remove_verts[mask[vertices[:, 2], vertices[:, 1], vertices[:, 0]]] = 0
    mesh.remove_vertices_by_mask(remove_verts.numpy())


def remove_all_but_biggest_component(mesh: o3d.geometry.TriangleMesh, right: bool = None, center_x: float = None):
    # TODO: if 2 big components are present, choose the "right" one -> right component for right fissure
    # get connected components and select the biggest
    triangle_clusters, _, cluster_area = mesh.cluster_connected_triangles()
    print(f"found {len(cluster_area)} connected components in prediction")
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_area = np.asarray(cluster_area)

    if center_x is not None:
        all_verts = np.asarray(mesh.vertices)
        all_tris = np.asarray(mesh.triangles)

        # check that cluster has its center in the right or left half of the space (depending on the value in "right")
        for c, cluster in enumerate(np.unique(triangle_clusters)):
            # current cluster triangles
            c_tris = all_tris[triangle_clusters == cluster]

            # all vertices belonging to current cluster
            c_verts = all_verts[np.unique(c_tris)]

            # center of cluster
            c_center = np.mean(c_verts, axis=0)

            # if the cluster is in the wrong half of the space remove it from the choice
            if right and c_center[0] > center_x:  # we are searching for right fissure but got cluster on the left
                # set the value to negative, but so that in case we have no cluster in the right half,
                # still the biggest will be chosen by argmax later
                cluster_area[c] = -1 / cluster_area[c]
            elif not right and c_center[0] < center_x:  # we are searching for left fissure but got cluster on the right
                cluster_area[c] = -1 / cluster_area[c]

    triangles_to_remove = np.logical_not(triangle_clusters == cluster_area.argmax())
    mesh.remove_triangles_by_mask(triangles_to_remove)


def binary_lung_mask_to_left_right(lung_mask: sitk.Image, left_label=1, right_label=2):
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    lung_components = connected_components_filter.Execute(lung_mask)
    obj_cnt = connected_components_filter.GetObjectCount()
    if obj_cnt < 2:
        raise ValueError(f'Found only {obj_cnt} connected components in lung mask. '
                         f'Can\'t determine left and right fissures.')
    elif obj_cnt > 2:
        warnings.warn(f'Found {obj_cnt} connected components in lung mask, but expected 2. '
                      f'Assuming the biggest 2 components are the right & left lung.')

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lung_components_sorted = relabel_filter.Execute(lung_components)
    if obj_cnt > 2:
        print(relabel_filter.GetSizeOfObjectsInPhysicalUnits())

    # extract 2 biggest objects (left & right lung)
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({l: 0 for l in range(3, relabel_filter.GetOriginalNumberOfObjects() + 1)})
    right_left_lung = change_label_filter.Execute(lung_components_sorted)

    # figure out which label is right, which is left
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(right_left_lung)
    centroids = np.array([shape_stats.GetCentroid(l) for l in range(1, 3)])
    cur_right_label, cur_left_label = np.argsort(centroids[:, 0]) + 1  # smaller x is right

    # change labels to be 1 for left, 2 for right lung
    change_map = {cur_left_label.item(): left_label, cur_right_label.item(): right_label}
    change_label_filter.SetChangeMap(change_map)
    right_left_lung = change_label_filter.Execute(right_left_lung)

    return right_left_lung


def binary_to_fissure_segmentation(binary_fissure_seg: torch.Tensor, lung_mask: sitk.Image, resample_spacing=None):
    left_label = 1
    right_label = 2

    right_left_lung = binary_lung_mask_to_left_right(lung_mask, left_label=left_label, right_label=right_label)

    if resample_spacing is not None:
        right_left_lung = resample_equal_spacing(right_left_lung, target_spacing=resample_spacing,
                                                 use_nearest_neighbor=True)

    # use lung labels to assign right/left fissure label to binary fissure segmentation
    lung_mask_tensor = sitk_image_to_tensor(right_left_lung).to(binary_fissure_seg.device)
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == left_label)] = left_label
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == right_label)] = right_label

    return binary_fissure_seg
