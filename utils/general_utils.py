import contextlib
import glob
import inspect
import itertools
import os
from typing import Sequence, List, Tuple

import numpy as np
import open3d as o3d
import torch
from numpy.typing import ArrayLike
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from torch.nn import functional as F

from constants import ALIGN_CORNERS


def new_dir(*paths_to_join):
    full_path = os.path.join(*paths_to_join)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def pairwise_dist(x):
    """ squared euclidean distance from each point in x to itself and others

    :param x: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    xTx = torch.bmm(x, x.transpose(2, 1))
    dist = xx - 2.0 * xTx + xx.transpose(2, 1)
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0  # ensure diagonal is 0
    return dist


def load_points(path: str, case: str, sequence: str = 'fixed', feat: str = None):
    return torch.load(os.path.join(path, f'{case}_coords_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_fissures_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_lobes_{sequence}.pth'), map_location='cpu') if os.path.isfile(os.path.join(path, f'{case}_lobes_{sequence}.pth')) else None, \
           torch.load(os.path.join(path, f'{case}_{feat}_{sequence}.pth'), map_location='cpu') if feat is not None \
               else None


def create_o3d_mesh(verts: ArrayLike, tris: ArrayLike):
    verts = o3d.utility.Vector3dVector(np.array(verts, dtype=np.float32))
    tris = o3d.utility.Vector3iVector(np.array(tris, dtype=np.uint32))
    return o3d.geometry.TriangleMesh(vertices=verts, triangles=tris)


def o3d_to_pt3d_meshes(o3d_meshes: List[o3d.geometry.TriangleMesh]):
    verts = []
    faces = []
    vert_normals = []
    for m in o3d_meshes:
        verts.append(torch.from_numpy(np.asarray(m.vertices)).float())
        faces.append(torch.from_numpy(np.asarray(m.triangles)).float())
        vert_normals.append(torch.from_numpy(np.asarray(m.vertex_normals)).float())

    return Meshes(verts, faces, verts_normals=vert_normals)


def pt3d_to_o3d_meshes(pt3d_meshes: Meshes):
    return [create_o3d_mesh(m.verts_padded().squeeze().detach().cpu(), m.faces_padded().squeeze().detach().cpu()) for m in pt3d_meshes]


def kpts_to_grid(kpts_world, shape, align_corners=None, return_transform=False):
    """ expects points in xyz format from a tensor with given shape

    :param kpts_world:
    :param shape:
    :param align_corners:
    :return: points in range [-1, 1]
    """
    device = kpts_world.device
    D, H, W = shape

    scale1 = 1 / (torch.tensor([W, H, D], device=device) - 1)
    kpts_pt_ = (kpts_world * scale1) * 2 - 1
    if not align_corners:
        scale2 = (torch.tensor([W, H, D], device=device) - 1) / torch.tensor([W, H, D], device=device)
        kpts_pt_ *= scale2

    if return_transform:
        transform = Transform3d(device=device)
        transform = transform.scale(scale1[None] * 2).translate(-1, -1, -1)
        if not align_corners:
            transform = transform.scale(scale2[None])
        assert torch.allclose(kpts_pt_, transform.transform_points(kpts_world), atol=1e-6)
        return kpts_pt_, transform

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
        kpts_pt = kpts_pt / ((torch.tensor([W, H, D], device=device) - 1) / torch.tensor([W, H, D], device=device))
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
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("empty mesh")
        return

    # get connected components and select the biggest
    triangle_clusters, _, cluster_area = mesh.cluster_connected_triangles()
    print(f"found {len(cluster_area)} connected components in prediction")
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_area = np.asarray(cluster_area)

    if center_x is not None and right is not None:
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
    # print('Removing triangles')
    mesh.remove_triangles_by_mask(triangles_to_remove)
    # print('Removing unreferenced points')
    mesh.remove_unreferenced_vertices()


def points_to_label_map(pts, labels, out_shape, spacing):
    """

    :param pts_index: in xyz format and in world coordinates (with the given spacing)
    :param labels: label for each point
    :param out_shape: in zyx format
    :param spacing: in xyz dimension
    :return: labelmap tensor
    """
    # transform points into pixel coordinates
    pts_index = pts / torch.tensor([spacing], device=pts.device)
    pts_index = pts_index.flip(-1).round().long()

    # prevent index out of bounds
    for d in range(len(out_shape)):
        pts_index[:, d] = torch.clamp(pts_index[:, d], min=0, max=out_shape[d] - 1)

    # create tensor
    label_map = torch.zeros(out_shape, dtype=labels.dtype, device=labels.device)
    label_map[pts_index[:, 0], pts_index[:, 1], pts_index[:, 2]] = labels.squeeze()

    return label_map, pts_index


def load_meshes(base_dir, case, sequence, obj_name='fissure') -> Tuple[o3d.geometry.TriangleMesh]:
    meshlist = sorted(glob.glob(os.path.join(base_dir, f'{case}_mesh_{sequence}', f'{case}_{obj_name}*_{sequence}.obj')))
    return tuple(o3d.io.read_triangle_mesh(m) for m in meshlist)


def sample_patches_at_kpts(img: torch.Tensor, kpts_grid: torch.Tensor, patch_size: int):
    """

    :param img: tensor to sample from of shape [1, 1, D, H, W]
    :param kpts_grid: pytorch grid points of shape [N, 3]
    :param patch_size: size of the sample patch in each dimension
    :return: sampled patches of shape [1, N, patch_size, patch_size, patch_size]
    """
    # make sure the points are in pytorch grid format [-1,1]
    if not (kpts_grid.min() >= -1. and kpts_grid.max() <= 1.):
        raise ValueError('Keypoints are not given in Pytorch grid coordinates')

    if img.shape[1] != 1 or img.shape[0] != 1:
        raise NotImplementedError('Cannot, for now, sample from batched or multichannel images.')

    # whole image grid with patch-size as dimension (value range [-1, 1])
    patch_grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), size=[1, 1] + [patch_size] * 3, align_corners=ALIGN_CORNERS).to(
        img.device)

    # limit to patch in pytorch grid coordinates
    patch_grid = patch_grid * (patch_size / torch.tensor(img.shape[2:][::-1], device=img.device))

    # broadcast the grid to every keypoint location
    patch_grid_at_kpts = patch_grid + kpts_grid.view(kpts_grid.shape[0], 1, 1, 1, kpts_grid.shape[-1])

    # reshape the grid to have shape: (B, n_keypoints, patch_size^3, 1, 3)
    # that means the batch size is matched to the image and the spatial dims contain number of KPs and the flattened grid
    patch_grid_at_kpts = patch_grid_at_kpts.flatten(start_dim=1, end_dim=-2).view(img.shape[0], kpts_grid.shape[0], patch_size**3, 1, 3)

    # sample the voxels in the patches around keypoints
    interpolation_mode = 'nearest' if patch_size % 2 == 1 else 'bilinear'
    patches = F.grid_sample(img, patch_grid_at_kpts,
                            mode=interpolation_mode, padding_mode='border', align_corners=ALIGN_CORNERS)

    # restore the spatial dimensions and have the different patches in the channel dimension
    patches = patches.view(img.shape[0], kpts_grid.shape[0], patch_size, patch_size, patch_size)
    return patches


def inverse_affine_transform(point_cloud: np.ndarray, scaling: float, rotation_mat: np.ndarray, affine_translation: np.ndarray):
    """

    :param point_cloud: (N, 3)
    :param scaling:
    :param rotation_mat:
    :param affine_translation:
    :return:
    """
    backend = torch if isinstance(point_cloud, torch.Tensor) else np
    rotation_inverse = backend.linalg.inv(rotation_mat)
    pc_not_affine = backend.matmul(rotation_inverse[None, :, :],
                                   1/scaling * (point_cloud - affine_translation)[:, :, None]).squeeze()
    return pc_not_affine


def knn(x, k, self_loop=False):
    # use k+1 and ignore first neighbor to exclude self-loop in graph
    k_modifier = 0 if self_loop else 1

    dist = pairwise_dist(x.transpose(2, 1))
    idx = dist.topk(k=k+k_modifier, dim=-1, largest=False)[1][..., k_modifier:]  # (batch_size, num_points, k)
    return idx


def decompose_similarity_transform(matrix):
    """ From https://math.stackexchange.com/a/1463487
    Only works for non-negative scaling and similarity transforms (without shear)

    :param matrix: homogeneous transformation matrix of size [D+1, D+1], where the translation vector is in the last column
    :return: translation [D], rotation [D, D], scale [D]
    """
    translation = matrix[:-1, -1]
    mat = matrix[:-1, :-1]
    scale = mat.norm(dim=1, keepdim=True)
    rot = mat / scale
    return translation, rot, scale.squeeze()


def nanstd(tensor, dim: int=None):
    if dim is None or len(tensor.shape) <=1:
        return tensor[~tensor.isnan()].std()
    else:
        leftover_shape = tuple(s for d, s in enumerate(tensor.shape) if d != dim)
        out = torch.empty(*leftover_shape, dtype=tensor.dtype)
        for index in itertools.product(*[range(s) for s in leftover_shape]):
            index_slice = list(index)
            index_slice.insert(dim, slice(None))
            indexed = tensor[index_slice]
            out[index] = indexed[~indexed.isnan()].std()
        return out


def save_meshes(meshes, base_dir, case, sequence, obj_name='fissure'):
    meshdir = os.path.join(base_dir, f"{case}_mesh_{sequence}")
    os.makedirs(meshdir, exist_ok=True)
    for m, mesh in enumerate(meshes):
        o3d.io.write_triangle_mesh(os.path.join(meshdir, f'{case}_{obj_name}{m + 1}_{sequence}.obj'), mesh)


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def topk_alldims(tensor, k):
    res = torch.topk(tensor.view(-1), k=k)

    idx = unravel_index(res.indices, tensor.size())
    return tensor[idx], idx


def get_device(gpu: int):
    if gpu in range(torch.cuda.device_count()):
        device = f'cuda:{gpu}'
        print(f"Using device: {device}")
    else:
        device = 'cpu'
        print(f'Requested GPU with index {gpu} is not available. Only {torch.cuda.device_count()} GPUs detected.')

    return device


@contextlib.contextmanager
def no_print():
    # Store builtin print
    old_print = print

    def dont_print(*args, **kwargs):
        # do nothing!
        pass

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = dont_print
        yield
    finally:
        inspect.builtins.print = old_print


def find_test_fold_for_id(case, sequence, split):
    """ find the fold, where this image has been in the test-split """
    sequence = sequence.replace('moving', 'mov').replace('fixed', 'fix')

    fold_nr = None
    for i, fold in enumerate(split):
        if any(case in name and sequence in name for name in fold['val']):
            fold_nr = i

    if fold_nr is None:
        raise ValueError(f'ID {case}_{sequence} is not present in any cross-validation test split)')

    return fold_nr


def corresponding_point_distance(prediction, target):
    return (prediction - target).pow(2).sum(-1).sqrt()
