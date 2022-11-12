import csv
import os
import pickle
from typing import Sequence

import math
import numpy as np
import open3d as o3d
import pycpd
import torch
from matplotlib import pyplot as plt
from pycpd import RigidRegistration
from torch.nn import functional as F

from data import ImageDataset
from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
from shape_model.ssm import save_shape
from utils.tqdm_utils import tqdm_redirect
from utils.utils import new_dir

INTERPOLATION_MODES = ['knn', 'tps']


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


def register_cpd_deformable(fixed_pc_np, moving_pc_np_prereg):
    """ assumes rigid/affine preregistration

    :param fixed_pc_np:
    :param moving_pc_np_prereg:
    :return:
    """
    # deformable registration
    deformable = pycpd.DeformableRegistration(X=fixed_pc_np, Y=moving_pc_np_prereg,
                                              alpha=0.01,
                                              # trade-off between regularization/smoothness (>1) and point fit (<1)
                                              beta=10)  # gaussian kernel width of the regularization kernel
    deformed_pc_np, (trf_G, trf_W) = deformable.register()

    displacements = np.matmul(trf_G, trf_W)
    return deformed_pc_np, displacements


def interpolate_displacements_tps(existing_points, values_at_existing_points, interpolation_points, img_shape):
    D, H, W = img_shape

    def normalize(grid):
        return (grid / torch.tensor([W - 1, H - 1, D - 1], device=grid.device)) * 2  # no -1 for displacements!

    def unnormalize(grid):
        return (grid * torch.tensor([H - 1, W - 1, D - 1], device=grid.device)) / 2

    dense_flow = thin_plate_dense(normalize(existing_points) - 1, normalize(values_at_existing_points),
                                  shape=(W, H, D), step=4, lambd=0.1)
    dense_flow = dense_flow.permute(0, 4, 1, 2, 3)
    sampled_displacements = F.grid_sample(dense_flow, normalize(interpolation_points.view(1, -1, 1, 1, 3)) - 1,
                                          align_corners=False).squeeze().t()
    return unnormalize(sampled_displacements.squeeze()).cpu().numpy()


def interpolate_displacements_weighted_knn(existing_points, values_at_existing_points, interpolation_points, k=5):
    # compute euclidean distance from interpolation points to all existing points
    # dist = pairwise_dist2(interpolation_points, existing_points).sqrt()
    dist = (interpolation_points.unsqueeze(2) - existing_points.unsqueeze(1)).square().sum(-1).sqrt()

    # find nearest k neighbors for each input point
    top_dist, top_idx = dist.topk(k=k, dim=-1, largest=False)  # (batch_size, num_points, k)

    # compute inverse distance weighted values
    inverse_weights = 1 / (top_dist.unsqueeze(-1) + 1e-8)
    inverse_dist_weighted_neighbor_values = inverse_weights * values_at_existing_points[:, top_idx.squeeze()]
    values_at_interpolation_points = inverse_dist_weighted_neighbor_values.sum(dim=-2) / inverse_weights.sum(dim=-2)
    return values_at_interpolation_points.cpu().numpy().squeeze()


def inverse_transformation_at_sampled_points(deformed_pc_np: np.ndarray, moving_displacements: np.ndarray,
                                             sample_point_cloud: np.ndarray, img_shape: Sequence[int],
                                             interpolation_mode='knn'):
    """

    :param deformed_pc_np: the moved point cloud in world coordinates [N_points, 3]
    :param moving_displacements: the displacements used for each point of the moved pc [N_points, 3]
    :param sample_point_cloud: locations at which the inverse transformation should be sampled [N_points_sample, 3]
    :param img_shape: shape of the original image in world coordinates
    :return: sampled point cloud in moving's space
    """
    # interpolate displacements at fixed points
    sample_pc_torch = torch.from_numpy(sample_point_cloud).unsqueeze(0).float()
    deformed_pc_torch = torch.from_numpy(deformed_pc_np).unsqueeze(0).float()
    moving_displacements = torch.from_numpy(moving_displacements).unsqueeze(0).float()

    if interpolation_mode == 'knn':
        interpolated_displacements = interpolate_displacements_weighted_knn(
            deformed_pc_torch, moving_displacements, sample_pc_torch)
    elif interpolation_mode == 'tps':
        interpolated_displacements = interpolate_displacements_tps(
            deformed_pc_torch, moving_displacements, sample_pc_torch, img_shape)
    else:
        raise ValueError(f'No interpolation mode named {interpolation_mode}.')

    new_moving_pc_np = sample_point_cloud - interpolated_displacements
    return new_moving_pc_np


def correspondence_distance_heuristic(pc: o3d.geometry.PointCloud):
    """ 5 % of the longest diagonal through the bounding box

    :param pc: fixed point cloud
    :return: maximum correspondence distance
    """
    volume_diagonal = np.linalg.norm(pc.get_max_bound() - pc.get_min_bound(), ord=2).item()
    corr_dist = volume_diagonal * 0.05
    return corr_dist


def register_all(fixed_pcs_np: Sequence[o3d.geometry.PointCloud],
                 all_moving_meshes: Sequence[Sequence[o3d.geometry.TriangleMesh]], ids, base_dir,
                 n_sample_points=1024, show=False):
    assert len(all_moving_meshes) == len(ids)

    fixed_pcs = [o3d.geometry.PointCloud(o3d.pybind.utility.Vector3dVector(p)) for p in fixed_pcs_np]
    n_points_per_obj = [len(pc) for pc in fixed_pcs_np]

    # output directory
    reg_dir = new_dir(base_dir, 'registrations')
    plot_dir = new_dir(reg_dir, 'plots')

    all_moving_pcs = []
    all_prereg_transforms = []
    all_moved_pcs = []
    all_displacements = []

    n_correspondences = []
    corr_distances = []
    for m, moving_meshes in enumerate(tqdm_redirect(all_moving_meshes)):
        moving_meshes = moving_meshes[:len(fixed_pcs_np)]

        # register all moving objects into fixed space
        moving_pcs = []
        displacements = []
        moved_pcs = []

        corresponding = []
        corresponding_max_dist = []

        # sample point clouds from meshes
        for obj_i, moving in enumerate(moving_meshes):
            # sample points evenly from mesh
            moving_pc = moving.sample_points_poisson_disk(number_of_points=len(fixed_pcs_np[obj_i]))
            moving_pc_np = np.asarray(moving_pc.points)
            moving_pcs.append(moving_pc_np)

        # joint rigid preregistration for all objects
        all_moving_objs = np.concatenate(moving_pcs, axis=0)
        all_fixed_objs = np.concatenate(fixed_pcs_np, axis=0)
        rigid = RigidRegistration(X=all_fixed_objs, Y=all_moving_objs)
        rigid_prereg, (scale, rotation, translation) = rigid.register()
        moving_pcs_prereg = np.split(rigid_prereg, [sum(n_points_per_obj[:i]) for i in range(1, len(n_points_per_obj))], axis=0)
        assert all(len(m_prereg) == len(f_pc) for m_prereg, f_pc in zip(moving_pcs_prereg, fixed_pcs_np))

        # pycpd defines affine transformation as x*A -> we transpose the matrix so we can use A^T*x
        all_prereg_transforms.append({'scale': scale, 'translation': translation, 'rotation': rotation.T})

        # individual deformable registration
        for obj_i, (fixed_pc_np, moving_pc_np_prereg) in enumerate(zip(fixed_pcs_np, moving_pcs_prereg)):
            # register with deformable coherent point drift
            deformed_pc_np, disp = register_cpd_deformable(fixed_pc_np, moving_pc_np_prereg)

            # save results
            moved_pcs.append(deformed_pc_np)
            displacements.append(disp)

            # check how many points already correspond (max distance is defined by correspondence_distance_heuristic)
            deformed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deformed_pc_np.astype(np.float32)))
            corresponding_max_dist.append(correspondence_distance_heuristic(fixed_pcs[obj_i]))
            reg_eval = o3d.pipelines.registration.evaluate_registration(
                source=deformed_pc, target=fixed_pcs[obj_i], max_correspondence_distance=corresponding_max_dist[-1])
            corresponding.append(len(np.asarray(reg_eval.correspondence_set)))

            # VISUALIZATION
            # show registration steps
            fig = plt.figure(figsize=(10, 6))
            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')

            visualize('Initial PC', fixed_pc_np, moving_pcs[obj_i], ax1)
            visualize('Rigid prereg.', fixed_pc_np, moving_pc_np_prereg, ax2)
            visualize('Deformable', fixed_pc_np, deformed_pc_np, ax3)

            fig.savefig(os.path.join(plot_dir, f'{"_".join(ids[m])}_obj{obj_i+1}'), bbox_inches='tight', dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)

        all_displacements.append(displacements)
        all_moved_pcs.append(moved_pcs)
        all_moving_pcs.append(moving_pcs)
        n_correspondences.append(np.stack(corresponding))
        corr_distances.append(np.stack(corresponding_max_dist))

    # output results
    def write(obj, fname):
        with open(os.path.join(reg_dir, fname), 'wb') as file:
            pickle.dump(obj, file)

    write(all_displacements, 'displacements.npz')
    write(all_moved_pcs, 'moved_pcs.npz')
    write(all_moving_pcs, 'moving_pcs.npz')
    write(all_prereg_transforms, 'transforms.npz')
    write(np.asarray(ids), 'ids.npz')

    n_correspondences = np.stack(n_correspondences)
    portion_of_correspondences = n_correspondences / np.array(n_points_per_obj)[None]
    corr_distances = np.stack(corr_distances)
    with open(os.path.join(reg_dir, 'correspondences.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object No.'] + [str(obj+1) for obj in range(len(all_moved_pcs[0]))] + ['mean'])
        writer.writerow(['Mean corresponding'] + [str(v) for v in portion_of_correspondences.mean(0)] + [str(portion_of_correspondences.mean())])
        writer.writerow(['StdDev corresponding'] + [str(v) for v in portion_of_correspondences.std(0)] + [str(portion_of_correspondences.std())])
        writer.writerow(['Max. corr dist'] + [str(v) for v in corr_distances.mean(0)] + [str(corr_distances.mean())])


if __name__ == '__main__':
    lobes = False
    total_segmentator = True
    n_sample_points = 1024
    show = False
    use_mesh = True

    # run_detached_from_pycharm()

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
                       "corresponding_points" + ("_mesh" if use_mesh else "") + ("_ts" if total_segmentator else ""),
                       "lobes" if lobes else "fissures")

    # get fixed meshes
    fixed_meshes = get_meshes(f)
    if use_mesh:
        # downsample mesh to gain ~ 1000 points
        print(f'original meshes: {fixed_meshes}')
        fixed_meshes = [m.simplify_vertex_clustering(voxel_size=3.5, contraction=o3d.geometry.SimplificationContraction.Quadric) for m in fixed_meshes]
        print(f'downsampled meshes: {fixed_meshes}')
        fixed_pcs = [m.vertices for m in fixed_meshes]

        # save faces
        with open(os.path.join(out_path, 'fixed_faces.npz'), 'wb') as faces_file:
            pickle.dump([np.asarray(m.triangles) for m in fixed_meshes], faces_file)

    else:
        # sample points from fixed
        fixed_pcs = [mesh.sample_points_poisson_disk(number_of_points=n_sample_points, seed=42).points for mesh in fixed_meshes]

    fixed_pcs_np = [np.asarray(pc) for pc in fixed_pcs]
    save_shape(fixed_pcs_np, os.path.join(out_path, f'{"_".join(ds.get_id(f))}.npz'), transforms=None)

    # register each image onto fixed
    moving_meshes = [get_meshes(m) for m in range(len(ds)) if m != f]
    moving_ids = [ds.get_id(m) for m in range(len(ds)) if m != f]  # prevent id mismatch

    # preprocess the registration
    register_all(fixed_pcs_np, moving_meshes, moving_ids, out_path, n_sample_points=n_sample_points, show=show)
