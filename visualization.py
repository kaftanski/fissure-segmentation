from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike


def visualize_with_overlay(image: np.ndarray, segmentation: np.ndarray, title: str = None, onehot_encoding: bool = False, ax=None):
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
                   cmap=ListedColormap([colors[i % len(colors)]]), alpha=0.5, interpolation=None)

    if title:
        ax.set_title(title)


def visualize_point_cloud(points, labels, title='', exclude_background=True):
    """

    :param points: point cloud with N points, shape: (Nx3)
    :param labels: label for each point, shape (N)
    :param title: figure title
    :param exclude_background: if true, points with label 0 will not be plotted
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = points.cpu()
    if exclude_background:
        points = points[labels != 0]
        labels = labels[labels != 0]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=labels.cpu(), cmap='tab10', marker='.')
    # ax.view_init(elev=100., azim=-60.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    plt.show()


def visualize_trimesh(vertices_list: Sequence[ArrayLike], triangles_list: Sequence[ArrayLike], title: str = ''):
    """

    :param vertices_list: list of vertices, shape (Vx3) tensors
    :param triangles_list: list of triangles, shape (Tx3) tensors, corresponding to the vertices
    :param title: figure title
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
    plt.show()