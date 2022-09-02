import csv
import os
import pickle
from typing import Union, Iterable, Sequence

import math
import numpy as np
import open3d as o3d
import pycpd
import pytorch3d
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

from data import ImageDataset
from metrics import point_surface_distance
from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
from shape_model.ssm import save_shape
from utils.tqdm_utils import tqdm_redirect
from utils.utils import new_dir
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
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Fixed', s=5)
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Moving', s=5)
    ax.text2D(0.87, 0.92, title,
              horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')


def register_cpd(fixed_pc_np, moving_pc_np):
    # affine pre-registration
    affine = pycpd.AffineRegistration(X=fixed_pc_np, Y=moving_pc_np)
    affine_prereg, (affine_mat, affine_translation) = affine.register()

    # deformable registration
    deformable = pycpd.DeformableRegistration(X=fixed_pc_np, Y=affine_prereg,
                                              alpha=0.01,
                                              # trade-off between regularization/smoothness (>1) and point fit (<1)
                                              beta=10)  # gaussian kernel width of the regularization kernel
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
    cf_dist = []
    affine_transforms = []

    # TODO: joint registration of left/right fissure
    for fixed_pc, moving in zip(fixed_pcs, moving_meshes):
        moving_pc = moving.sample_points_poisson_disk(number_of_points=n_sample_points, seed=23)

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
        cf, _ = pytorch3d.loss.chamfer_distance(torch.from_numpy(new_moving_pc_np_not_affine[None]).float(),
                                                torch.from_numpy(moving_pc_np[None]).float())
        print(f'Chamfer Distance: {cf:.4f}')

        mean_p2m.append(p2m)
        std_p2m.append(std)
        hd_p2m.append(hd)
        cf_dist.append(cf)

    new_moving_pcs = np.stack(new_moving_pcs, axis=0)
    fixed_pcs = np.stack(fixed_pcs, axis=0)
    affine_transforms = np.stack(affine_transforms, axis=0)
    return new_moving_pcs, fixed_pcs, affine_transforms, \
           {'mean': torch.stack(mean_p2m), 'std': torch.stack(std_p2m), 'hd': torch.stack(hd_p2m), 'cf': torch.stack(cf_dist)}


def register_all(fixed_pcs: Sequence[o3d.geometry.PointCloud],
                 all_moving_meshes: Sequence[Sequence[o3d.geometry.TriangleMesh]], ids, base_dir,
                 n_sample_points=1024, show=False):
    assert len(all_moving_meshes) == len(ids)

    # output directory
    reg_dir = new_dir(base_dir, 'registrations')
    plot_dir = new_dir(reg_dir, 'plots')

    all_affine_transforms = []
    all_moved_pcs = []
    all_displacements = []
    all_moving_pcs = []

    n_correspondences = []

    for m, moving_meshes in enumerate(tqdm_redirect(all_moving_meshes)):
        moving_pcs = []
        displacements = []
        moved_pcs = []
        affine_transforms = []

        corresponding = []

        # register all moving objects into fixed space
        for obj_i, (fixed, moving) in enumerate(zip(fixed_pcs, moving_meshes)):
            fixed_pc_np = np.asarray(fixed.points)

            # sample points evenly from mesh
            moving_pc = moving.sample_points_poisson_disk(number_of_points=n_sample_points)
            moving_pc_np = np.asarray(moving_pc.points)
            moving_pcs.append(moving_pc_np)

            # register affine and deformable
            affine_prereg, affine_mat, affine_translation, deformed_pc_np, disp = register_cpd(
                fixed_pc_np, moving_pc_np)
            deformed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deformed_pc_np.astype(np.float32)))

            # save results
            moved_pcs.append(deformed_pc_np)
            displacements.append(disp)
            # affine transform:
            # pycpd defines affine transformation as x*A -> we transpose the matrix so we can use A^T*x
            affine_transforms.append(np.concatenate([affine_mat.T, affine_translation[:, np.newaxis]], axis=1))

            # check how many points already correspond
            reg_eval = o3d.pipelines.registration.evaluate_registration(
                source=deformed_pc, target=fixed, max_correspondence_distance=10)
            corresponding.append(len(np.asarray(reg_eval.correspondence_set)))

            # VISUALIZATION
            # show registration steps
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')

            visualize('Initial PC', fixed_pc_np, moving_pc_np, ax1)
            visualize('Affine', fixed_pc_np, affine_prereg, ax2)
            visualize('Deformable', fixed_pc_np, deformed_pc_np, ax3)

            fig.savefig(os.path.join(plot_dir, f'{"_".join(ids[m])}_obj{obj_i+1}'), bbox_inches='tight', dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)

        all_displacements.append(np.stack(displacements))
        all_moved_pcs.append(np.stack(moved_pcs))
        all_affine_transforms.append(np.stack(affine_transforms))
        all_moving_pcs.append(np.stack(moving_pcs))
        n_correspondences.append(np.stack(corresponding))

    # output results
    def write(obj, fname):
        with open(os.path.join(reg_dir, fname), 'wb') as file:
            pickle.dump(obj, file)

    write(np.stack(all_displacements), 'displacements.npz')
    write(np.stack(all_moved_pcs), 'moved_pcs.npz')
    write(np.stack(all_moving_pcs), 'moving_pcs.npz')
    write(np.stack(all_affine_transforms), 'transforms.npz')
    write(np.asarray(ids), 'ids.npz')

    n_correspondences = np.stack(n_correspondences)
    portion_of_correspondences = n_correspondences / n_sample_points
    with open(os.path.join(reg_dir, 'correspondences.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object No.'] + [str(obj+1) for obj in range(len(all_moved_pcs[0]))] + ['mean'])
        writer.writerow(['Mean corresponding'] + [str(v) for v in portion_of_correspondences.mean(0)] + [str(portion_of_correspondences.mean())])
        writer.writerow(['StdDev corresponding'] + [str(v) for v in portion_of_correspondences.std(0)] + [str(portion_of_correspondences.std())])


if __name__ == '__main__':
    lobes = True
    total_segmentator = True
    n_sample_points = 1024
    show = False

    # data set
    if total_segmentator:
        ds = TotalSegmentatorDataset()
        f = 1
    else:
        ds = ImageDataset("../data", do_augmentation=False, resample_spacing=1.)
        f = 0
    get_meshes = ds.get_fissure_meshes if not lobes else ds.get_lobe_meshes

    # set output path
    out_path = new_dir("results",
                       "corresponding_points" + "_ts" if total_segmentator else "",
                       "lobes" if lobes else "fissures")

    # get fixed meshes
    fixed_meshes = get_meshes(f)

    # sample points from fixed
    fixed_pcs = [mesh.sample_points_poisson_disk(number_of_points=n_sample_points, seed=42) for mesh in fixed_meshes]
    fixed_pcs_np = np.stack([pc.points for pc in fixed_pcs])
    save_shape(fixed_pcs_np, os.path.join(out_path, f'{"_".join(ds.get_id(f))}.npz'), transforms=None)

    # register each image onto fixed
    moving_meshes = [get_meshes(m) for m in range(len(ds)) if m != f]
    moving_ids = [ds.get_id(m) for m in range(len(ds)) if m != f]  # prevent id mismatch

    # preprocess the registration
    register_all(fixed_pcs, moving_meshes, moving_ids, out_path, n_sample_points=n_sample_points, show=show)
