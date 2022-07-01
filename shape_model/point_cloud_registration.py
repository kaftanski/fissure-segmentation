from typing import Union, Iterable

import math
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import pycpd
from tqdm.contrib import itertools
from torch.nn import functional as F

from data import ImageDataset


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


def register(fixed_meshes: Union[Iterable[o3d.geometry.TriangleMesh], o3d.geometry.TriangleMesh],
             moving_meshes: Union[Iterable[o3d.geometry.TriangleMesh], o3d.geometry.TriangleMesh],
             img_shape, n_sample_points=512):
    """ Cave: meshes should be in world coordinates (or at least with an isotropic spacing)

    :param fixed_meshes:
    :param moving_meshes:
    :param img_shape: shape of the original image (with unit spacing)
    :param n_sample_points:
    :return:
    """

    if type(fixed_meshes) != type(moving_meshes):
        raise ValueError(f'Type mismatch between fixed_meshes ({type(fixed_meshes)}) and moving_meshes ({type(moving_meshes)})')

    if not isinstance(fixed_meshes, Iterable):
        fixed_meshes = [fixed_meshes]
        moving_meshes = [moving_meshes]

    # TODO: joint registration of left/right fissure
    for fixed, moving in zip(fixed_meshes, moving_meshes):
        fixed_pc = fixed.sample_points_poisson_disk(number_of_points=n_sample_points)
        moving_pc = moving.sample_points_poisson_disk(number_of_points=n_sample_points)

        fixed_pc_np = np.asarray(fixed_pc.points)
        moving_pc_np = np.asarray(moving_pc.points)

        affine = pycpd.AffineRegistration(X=fixed_pc_np, Y=moving_pc_np)
        affine_prereg, affine_params = affine.register()

        deformable = pycpd.DeformableRegistration(X=fixed_pc_np, Y=affine_prereg,
            alpha=0.001,  # trade-off between regularization/smoothness (>1) and point fit (<1)
            beta=2)  # gaussian kernel width of the regularization kernel
        deformed_pc_np, (trf_G, trf_W) = deformable.register()

        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        visualize('Initial PC', fixed_pc_np, moving_pc_np, ax1)
        visualize('Affine', fixed_pc_np, affine_prereg, ax2)
        visualize('Deformable', fixed_pc_np, deformed_pc_np, ax3)
        #plt.show()

        deformed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deformed_pc_np.astype(np.float32)))
        reg_result = o3d.pipelines.registration.evaluate_registration(source=deformed_pc, target=fixed_pc,
                                                                      max_correspondence_distance=10)  # TODO: heuristic
        print(reg_result)

        reg_result_icp = o3d.pipelines.registration.registration_icp(source=deformed_pc, target=fixed_pc,
                                                                     max_correspondence_distance=10)

        # interpolate displacements at fixed points
        D, H, W = img_shape

        def normalize(grid):
            return (grid / torch.tensor([W - 1, H - 1, D - 1], device=grid.device)) * 2  # no -1 for displacements!

        def unnormalize(grid):
            return (grid * torch.tensor([H - 1, W - 1, D - 1], device=grid.device)) / 2

        fixed_pc_torch = torch.from_numpy(fixed_pc_np).unsqueeze(0).float()
        deformed_pc_torch = torch.from_numpy(deformed_pc_np).unsqueeze(0).float()

        displacements = torch.from_numpy(np.matmul(trf_G, trf_W)).unsqueeze(0).float()

        dense_flow = thin_plate_dense(normalize(deformed_pc_torch)-1, normalize(displacements),
                                      shape=(W, H, D), step=4, lambd=0.1)
        dense_flow = dense_flow.permute(0, 4, 1, 2, 3)
        displacements_to_fixed = F.grid_sample(dense_flow, normalize(fixed_pc_torch.view(1, -1, 1, 1, 3))-1,
                                               align_corners=False).squeeze().t()
        new_moving_pc_np = fixed_pc_np - unnormalize(displacements_to_fixed.squeeze()).cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        visualize('fixed-sampled_disp', new_moving_pc_np, affine_prereg, ax)  # TODO: possibly undo affine reg
        plt.show()


if __name__ == '__main__':
    ds = ImageDataset("../data", do_augmentation=False, resample_spacing=1.)

    for f, m in itertools.product(range(len(ds)), range(len(ds)), disable=True):
        if f == m:
            continue

        if not ds.get_id(f)[1] == ds.get_id(m)[1] == 'fixed':
            # use inhale scans for now only
            continue

        img_fixed, _ = ds[f]
        register(ds.get_fissure_meshes(f), ds.get_fissure_meshes(m), img_shape=img_fixed.shape)
