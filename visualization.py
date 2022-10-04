from typing import Sequence, Union

import numpy as np
import open3d
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from numpy.typing import ArrayLike

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["image.origin"] = 'lower'


def visualize_with_overlay(image: ArrayLike, segmentation: ArrayLike, title: str = None, alpha=0.5, onehot_encoding: bool = False, ax=None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

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
    colors = ['r', 'g', 'b', 'y']

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

    if isinstance(points, torch.Tensor):
        points = points.cpu()

    if exclude_background:
        points = points[labels != 0]
        labels = labels[labels != 0]

    colors = ['r', 'g', 'b', 'y']
    cmap = ListedColormap(colors[:len(labels.unique())])

    point_cloud_on_axis(ax, points, labels.cpu(), cmap, title=title)
    # ax.view_init(elev=100., azim=-60.)
    
    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def point_cloud_on_axis(ax, points, c, cmap=None, marker='.', title='', label=''):
    if isinstance(points, torch.Tensor):
        points = points.cpu()

    points = points.squeeze()

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, cmap=cmap, marker=marker, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    if label:
        ax.legend()


def visualize_o3d_mesh(mesh: Union[Sequence[open3d.geometry.TriangleMesh], open3d.geometry.TriangleMesh],
                       title: str = '', show=True, savepath=None):
    if not isinstance(mesh, Sequence):
        mesh = [mesh]

    visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in mesh],
                      triangles_list=[np.asarray(m.triangles) for m in mesh],
                      title=title, show=show, savepath=savepath)


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

    colors = ['r', 'g', 'b', 'y', 'c']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        trimesh_on_axis(ax, vertices, triangles, colors[i], title)

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def trimesh_on_axis(ax, vertices, triangles, color, title='', alpha=1., label=''):
    if len(vertices) > 0:
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, color=color, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    if label:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(Patch(facecolor=color, edgecolor=color, label=label, alpha=alpha))
        plt.legend(handles=handles)


def plot_slice(img, s, b=0, c=0, dim=0, title='', save_path=None, show=True):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()

    index = [slice(b, b+1), slice(c, c+1)] + [slice(None)]*dim + [slice(s, s+1)]
    plt.imshow(img[index].squeeze(), cmap='gray')
    if title:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()