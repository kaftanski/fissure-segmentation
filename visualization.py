from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike

plt.rcParams['image.cmap'] = 'gray'


def visualize_with_overlay(image: np.ndarray, segmentation: np.ndarray, title: str = None, alpha=0.5, onehot_encoding: bool = False, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if segmentation.ndim == 2:
        if onehot_encoding:
            segmentation = segmentation.reshape([*segmentation.shape, 1])
        else:
            # encoding is a label map with numbers representing the objects
            labels = np.unique(segmentation)

            # excluding background with value 0
            labels = labels[labels != 0]

            segmentation_onehot = np.zeros((*image.shape, len(labels)))
            for i, l in enumerate(labels):
                segmentation_onehot[segmentation == l, i] = 1

            segmentation = segmentation_onehot

    ax.imshow(image)
    colors = ['g', 'y', 'c', 'r']

    for i in range(segmentation.shape[-1]):
        ax.imshow(np.ma.masked_where(segmentation[:, :, i] == 0, np.full([*segmentation.shape[:2]], fill_value=255)),
                  cmap=ListedColormap([colors[i % len(colors)]]), alpha=alpha, interpolation=None)

    if title:
        ax.set_title(title)


def visualize_point_cloud(points, labels, title='', exclude_background=True, show=True, savepath=None):
    """

    :param points: point cloud with N points, shape: (Nx3)
    :param labels: label for each point, shape (N)
    :param title: figure title
    :param exclude_background: if true, points with label 0 will not be plotted
    :param show: switch for calling pyplot show or not
    :param savepath: if not None, the figure will be saved to this path
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = points.cpu()
    if exclude_background:
        points = points[labels != 0]
        labels = labels[labels != 0]

    colors = ['r', 'g', 'b', 'y']
    cmap = ListedColormap(colors[:len(labels.unique())])

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels.cpu(), cmap=cmap, marker='.')
    # ax.view_init(elev=100., azim=-60.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_trimesh(vertices_list: Sequence[ArrayLike], triangles_list: Sequence[ArrayLike], title: str = '',
                      show=True, savepath=None):
    """

    :param vertices_list: list of vertices, shape (Vx3) tensors
    :param triangles_list: list of triangles, shape (Tx3) tensors, corresponding to the vertices
    :param title: figure title
    :param show: switch for calling pyplot show or not
    :param savepath: if not None, the figure will be saved to this path
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'y']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, color=colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
