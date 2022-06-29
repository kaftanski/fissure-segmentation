from functools import partial
from typing import Union, Iterable

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d
import pycpd
from tqdm.contrib import itertools

from data import ImageDataset


def visualize(title, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Fixed')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Moving')
    ax.text2D(0.87, 0.92, title,
              horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    # plt.draw()
    # plt.pause(0.001)


def register(fixed_meshes: Union[Iterable[o3d.geometry.TriangleMesh], o3d.geometry.TriangleMesh],
             moving_meshes: Union[Iterable[o3d.geometry.TriangleMesh], o3d.geometry.TriangleMesh],
             n_sample_points=2048):

    if type(fixed_meshes) != type(moving_meshes):
        raise ValueError(f'Type mismatch between fixed_meshes ({type(fixed_meshes)}) and moving_meshes ({type(moving_meshes)})')

    if not isinstance(fixed_meshes, Iterable):
        fixed_meshes = [fixed_meshes]
        moving_meshes = [moving_meshes]

    for fixed, moving in zip(fixed_meshes, moving_meshes):
        fixed_pc = fixed.sample_points_poisson_disk(number_of_points=n_sample_points)
        moving_pc = moving.sample_points_poisson_disk(number_of_points=n_sample_points)

        fixed_pc_np = np.asarray(fixed_pc.points)
        moving_pc_np = np.asarray(moving_pc.points)

        affine = pycpd.AffineRegistration(X=fixed_pc_np, Y=moving_pc_np)
        affine_prereg, affine_params = affine.register()

        deformable = pycpd.DeformableRegistration(X=fixed_pc_np, Y=affine_prereg,
                                                  alpha=2, beta=2)  # TODO: what are these parameters?
        deformed_pc, (trf_G, trf_W) = deformable.register()

        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')

        visualize('Initial PC', fixed_pc_np, moving_pc_np, ax1)
        visualize('Affine', fixed_pc_np, affine_prereg, ax2)
        visualize('Defomable', fixed_pc_np, deformed_pc, ax3)
        plt.show()


if __name__ == '__main__':
    ds = ImageDataset("../data", do_augmentation=False)

    for f, m in itertools.product(range(len(ds)), range(len(ds))):
        if f == m:
            continue

        if not ds.get_id(f)[1] == ds.get_id(m)[1] == 'fixed':
            # use inhale scans for now only
            continue

        register(ds.get_fissure_meshes(f), ds.get_fissure_meshes(m))