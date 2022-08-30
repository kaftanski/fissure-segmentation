import csv
import math
import os.path
from typing import Union, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycpd
import torch
from torch.nn import functional as F

from metrics import point_surface_distance
from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
from shape_model.ssm import save_shape
from utils.tqdm_utils import tqdm_redirect
from visualization import trimesh_on_axis, point_cloud_on_axis


class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device

        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        # theta = torch.solve(v, A)[0]
        return theta

    @staticmethod
    def d(a, b):
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r ** 2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()


def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2 ** 12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D // step, H // step, W // step

    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    theta = TPS.fit(x1[0], y1[0], lambd)

    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1 * H1 * W1
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = TPS.z(x2[j1:j2], x1[0], theta)

    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

    return y2


def visualize(title, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Fixed')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Moving')
    ax.text2D(0.87, 0.92, title,
              horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')


def register_cpd(fixed_pc_np, moving_pc_np):
    # affine pre-registration
    affine = pycpd.AffineRegistration(X=fixed_pc_np, Y=moving_pc_np)
    affine_prereg, (affine_mat, affine_translation) = affine.register()

    # deformable registration
    deformable = pycpd.DeformableRegistration(X=fixed_pc_np, Y=affine_prereg,
                                              alpha=0.001,
                                              # trade-off between regularization/smoothness (>1) and point fit (<1)
                                              beta=2)  # gaussian kernel width of the regularization kernel
    deformed_pc_np, (trf_G, trf_W) = deformable.register()

    displacements = np.matmul(trf_G, trf_W)
    return affine_prereg, affine_mat, affine_translation, deformed_pc_np, displacements


def inverse_transformation_at_sampled_points(deformed_pc_np: np.ndarray, moving_displacements: np.ndarray,
                                             sample_point_cloud: np.ndarray, img_shape: Sequence[int]):
    """

    :param deformed_pc_np: the moved point cloud in world coordinates [N_points, 3]
    :param moving_displacements: the displacements used for each point of the moved pc [N_points, 3]
    :param sample_point_cloud: locations at which the inverse transformation should be sampled [N_points_sample, 3]
    :param img_shape: shape of the original image in world coordinates
    :return: sampled point cloud in moving's space
    """
    # interpolate displacements at fixed points
    D, H, W = img_shape

    def normalize(grid):
        return (grid / torch.tensor([W - 1, H - 1, D - 1], device=grid.device)) * 2  # no -1 for displacements!

    def unnormalize(grid):
        return (grid * torch.tensor([H - 1, W - 1, D - 1], device=grid.device)) / 2

    sample_pc_torch = torch.from_numpy(sample_point_cloud).unsqueeze(0).float()
    deformed_pc_torch = torch.from_numpy(deformed_pc_np).unsqueeze(0).float()
    moving_displacements = torch.from_numpy(moving_displacements).unsqueeze(0).float()
    dense_flow = thin_plate_dense(normalize(deformed_pc_torch) - 1, normalize(moving_displacements),
                                  shape=(W, H, D), step=4, lambd=0.1)
    dense_flow = dense_flow.permute(0, 4, 1, 2, 3)
    sampled_displacements = F.grid_sample(dense_flow, normalize(sample_pc_torch.view(1, -1, 1, 1, 3)) - 1,
                                           align_corners=False).squeeze().t()
    new_moving_pc_np = sample_point_cloud - unnormalize(sampled_displacements.squeeze()).cpu().numpy()
    return new_moving_pc_np


def inverse_affine_transform(point_cloud: np.ndarray, affine_mat: np.ndarray, affine_translation: np.ndarray):
    affine_inverse = np.linalg.inv(affine_mat)
    pc_not_affine = np.matmul((point_cloud - affine_translation)[:, None, :], affine_inverse[None, :, :]).squeeze()
    return pc_not_affine


def simple_correspondence(fixed_pcs: Union[Iterable[o3d.geometry.PointCloud], o3d.geometry.PointCloud],
                          moving_meshes: Union[Iterable[o3d.geometry.TriangleMesh], o3d.geometry.TriangleMesh],
                          img_shape, n_sample_points=1024, undo_affine_reg=True, show=True):
    """ Cave: meshes should be in world coordinates (or at least with an isotropic spacing)

    :param fixed_pcs:
    :param moving_meshes:
    :param img_shape: shape of the original image (with unit spacing)
    :param n_sample_points:
    :return:
    """
    if not isinstance(fixed_pcs, Iterable):
        fixed_pcs = [fixed_pcs]
        moving_meshes = [moving_meshes]

    new_moving_pcs = []
    mean_p2m = []
    std_p2m = []
    hd_p2m = []
    affine_transforms = []

    # TODO: joint registration of left/right fissure
    for fixed_pc, moving in zip(fixed_pcs, moving_meshes):
        moving_pc = moving.sample_points_poisson_disk(number_of_points=n_sample_points)

        fixed_pc_np = np.asarray(fixed_pc.points)
        moving_pc_np = np.asarray(moving_pc.points)

        # register and check result
        affine_prereg, affine_mat, affine_translation, deformed_pc_np, displacements = register_cpd(
            fixed_pc_np, moving_pc_np)
        deformed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deformed_pc_np.astype(np.float32)))
        reg_result = o3d.pipelines.registration.evaluate_registration(source=deformed_pc, target=fixed_pc,
                                                                      max_correspondence_distance=10)
        print(reg_result)

        # compute corresponding points
        new_moving_pc_np = inverse_transformation_at_sampled_points(deformed_pc_np, displacements, fixed_pc_np, img_shape)

        # undo affine transformation
        new_moving_pc_np_not_affine = inverse_affine_transform(new_moving_pc_np, affine_mat, affine_translation)

        # switch to use the affine registered PCs or from the original space
        if undo_affine_reg:
            new_moving_pcs.append(new_moving_pc_np_not_affine)

            # affine transform was undone, just the identity is left
            affine_transforms.append(np.eye(3, 4))
        else:
            new_moving_pcs.append(new_moving_pc_np)

            # remember affine transformation for later
            # (pycpd defines affine transformation as x*A -> we transpose the matrix so we can use A^T*x)
            affine_transforms.append(np.concatenate([affine_mat.T, affine_translation[:, np.newaxis]], axis=1))

        if show:
            # VISUALIZATION
            # show registration steps
            fig = plt.figure()
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')

            visualize('Initial PC', fixed_pc_np, moving_pc_np, ax1)
            visualize('Affine', fixed_pc_np, affine_prereg, ax2)
            visualize('Deformable', fixed_pc_np, deformed_pc_np, ax3)

            # visualize sampling and back-transformation
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            visualize('fixed-sampled_disp', new_moving_pc_np, affine_prereg, ax)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            point_cloud_on_axis(ax, new_moving_pc_np_not_affine, c='r', cmap=None, title='sampled from fixed')
            trimesh_on_axis(ax, np.asarray(moving.vertices), np.asarray(moving.triangles), color='b', title='sampled from fixed', alpha=0.5)
            plt.show()

        # EVALUATION: compute surface distance between sampled moving and GT mesh!
        dist_moved = point_surface_distance(query_points=new_moving_pc_np_not_affine, trg_points=moving.vertices,
                                            trg_tris=moving.triangles)
        p2m = dist_moved.mean()
        std = dist_moved.std()
        hd = dist_moved.max()
        print(f'Point Cloud distance to GT mesh: {p2m.item():.4f} +- {std.item():.4f} (Hausdorff: {hd.item():.4f})')
        mean_p2m.append(p2m)
        std_p2m.append(std)
        hd_p2m.append(hd)

    new_moving_pcs = np.stack(new_moving_pcs, axis=0)
    fixed_pcs = np.stack(fixed_pcs, axis=0)
    affine_transforms = np.stack(affine_transforms, axis=0)
    return new_moving_pcs, fixed_pcs, affine_transforms, {'mean': torch.stack(mean_p2m), 'std': torch.stack(std_p2m), 'hd': torch.stack(hd_p2m)}


if __name__ == '__main__':
    lobes = True
    base_path = 'results/corresponding_points_totalseg'
    out_path = os.path.join(base_path, 'fissures' if not lobes else 'lobes')
    os.makedirs(out_path, exist_ok=True)
    undo_affine = False
    n_sample_points = 1024

    ds = TotalSegmentatorDataset()

    mean_p2m = []
    std_p2m = []
    hd_p2m = []

    f = 1
    sequence = ds.get_id(f)[1]
    # assert sequence == 'fixed'
    fixed_meshes = ds.get_fissure_meshes(f) if not lobes else ds.get_lobe_meshes(f)
    img_fixed, _ = ds[f]

    # sample points from fixed
    fixed_pcs = [mesh.sample_points_poisson_disk(number_of_points=n_sample_points) for mesh in fixed_meshes]
    fixed_pcs_np = np.stack([pc.points for pc in fixed_pcs])
    save_shape(fixed_pcs_np, os.path.join(out_path, f'{"_".join(ds.get_id(f))}_corr_pts.npz'), transforms=None)

    # register each image onto fixed
    for m in tqdm_redirect(range(len(ds))):
        if f == m:
            continue

        # if not ds.get_id(m)[1] == sequence:
        #     # use inhale scans for now only
        #     continue

        corr_points, fixed_pts, transforms, evaluation = \
            simple_correspondence(fixed_pcs, ds.get_fissure_meshes(m) if not lobes else ds.get_lobe_meshes(f),
                                  img_shape=img_fixed.shape, show=False, undo_affine_reg=undo_affine)

        mean_p2m.append(evaluation['mean'])
        std_p2m.append(evaluation['std'])
        hd_p2m.append(evaluation['hd'])

        save_shape(corr_points, os.path.join(out_path, f'{"_".join(ds.get_id(m))}_corr_pts.npz'), transforms=transforms)

    mean_p2m = torch.stack(mean_p2m)
    std_p2m = torch.stack(std_p2m)
    hd_p2m = torch.stack(hd_p2m)
    print('\n====== RESULTS ======')
    print(f'P2M distance: {mean_p2m.mean(0)} +- {std_p2m.mean(0)} (Hausdorff: {hd_p2m.mean(0)}')
    with open(os.path.join(out_path, 'results.csv'), 'w') as csv_results_file:
        writer = csv.writer(csv_results_file)
        writer.writerow(['Affine-Pre-Reg', str(not undo_affine)])
        writer.writerow(['Object'] + [str(i+1) for i in range(mean_p2m.shape[1])] + ['mean'])
        writer.writerow(['mean p2m'] + [v.item() for v in mean_p2m.mean(0)] + [mean_p2m.mean().item()])
        writer.writerow(['std p2m'] + [v.item() for v in std_p2m.mean(0)] + [std_p2m.mean().item()])
        writer.writerow(['hd p2m'] + [v.item() for v in hd_p2m.mean(0)] + [hd_p2m.mean().item()])
