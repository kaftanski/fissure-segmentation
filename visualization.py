from typing import Sequence, Union

import math
import numpy as np
import open3d
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
from numpy.typing import ArrayLike
from skimage.color import lab2rgb

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["image.origin"] = 'lower'


class HandlerColorvalues(HandlerBase):
    """
    Handler to put a colormap into a legend. Adapted from: https://stackoverflow.com/a/55501861
    """
    def __init__(self, colors, square_lengths=4, alpha=1., **kw):
        HandlerBase.__init__(self, **kw)
        self.alpha = alpha
        try:
            self.num_x = square_lengths
            self.num_y = square_lengths
            self.colors = colors.reshape(int(math.sqrt(len(colors))), int(math.sqrt(len(colors))), colors.shape[-1])
            subsample_x = self.colors.shape[0] // square_lengths
            subsample_y = self.colors.shape[1] // square_lengths
            self.colors = self.colors[::subsample_x, ::subsample_y][:square_lengths, :square_lengths]

        except ValueError:
            self.num_x = square_lengths**2
            self.num_y = 1
            subsample_factor = len(colors) // self.num_x
            self.colors = colors[::subsample_factor][:self.num_x, None]

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for j in range(self.num_y):
            for i in range(self.num_x):
                s = Rectangle([xdescent + i * width / self.num_x, ydescent + j * width / self.num_y],
                              width / self.num_x, height / self.num_y,
                              fc=self.colors[i, j], transform=trans, alpha=self.alpha)
                stripes.append(s)
        return stripes


class HandlerBremmCmap(HandlerBase):
    """
    Handler to put a colormap into a legend. Adapted from: https://stackoverflow.com/a/55501861
    """
    def __init__(self, num_x=10, num_y=5, alpha=1., **kw):
        HandlerBase.__init__(self, **kw)
        self.alpha = alpha
        self.num_x = num_x
        self.num_y = num_y
        x = torch.linspace(-1., 1, steps=num_x)
        y = torch.linspace(-1., 1, steps=num_y)
        points = torch.stack([arr.reshape(-1) for arr in torch.meshgrid(x, y)], dim=1)
        colors = color_2d_points_bremm(points)
        self.colors = colors.reshape(num_x, num_y, colors.shape[-1])

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for j in range(self.num_y):
            for i in range(self.num_x):
                s = Rectangle([xdescent + i * width / self.num_x, ydescent + j * height / self.num_y],
                              width / self.num_x, height / self.num_y,
                              fc=self.colors[i, j], transform=trans, alpha=self.alpha)
                stripes.append(s)
        return stripes


def visualize_with_overlay(image: ArrayLike, segmentation: ArrayLike, title: str = None, alpha=0.5, onehot_encoding: bool = False, ax=None, colors=('r', 'g', 'b', 'y'), spacing=None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if segmentation.ndim == 2:
        if onehot_encoding:
            segmentation = segmentation.reshape([*segmentation.shape, 2])
        else:
            # encoding is a label map with numbers representing the objects
            labels = np.unique(segmentation)

            # excluding background with value 0
            labels = labels[labels != 0]

            segmentation_onehot = np.zeros((*image.shape, len(labels)))
            for i, l in enumerate(labels):
                segmentation_onehot[segmentation == l, i] = 1

            segmentation = segmentation_onehot

    aspect_ratio = 1 if spacing is None else spacing[0] / spacing[1]
    ax.imshow(image, aspect=aspect_ratio)
    ax.set_axis_off()
    for i in range(segmentation.shape[-1]):
        ax.imshow(np.ma.masked_where(segmentation[:, :, i] == 0, np.full([*segmentation.shape[:2]], fill_value=255)),
                  cmap=ListedColormap([colors[i % len(colors)]]), alpha=alpha, interpolation=None,
                  aspect=aspect_ratio)

    if title:
        ax.set_title(title)


def visualize_point_cloud(points, labels=None, title='', exclude_background=True, show=True, savepath=None,
                          colors=('r', 'g', 'b', 'y'), alpha=1., show_ax=True):
    """

    :param points: point cloud with N points, shape: (Nx3)
    :param labels: label for each point, shape (N)
    :param title: figure title
    :param exclude_background: if true, points with label 0 will not be plotted
    :param show: switch for calling pyplot show or not
    :param savepath: if not None, the figure will be saved to this path
    :param colors: list of colors for each label
    :param alpha: transparency of the points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(points, torch.Tensor):
        points = points.cpu()

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    if labels is None:
        labels = torch.ones(points.shape[0])

    if exclude_background:
        points = points[labels != 0]
        labels = labels[labels != 0]

    # colors = ['r', 'g', 'b', 'y']
    cmap = ListedColormap(colors[:len(labels.unique())])

    point_cloud_on_axis(ax, points, labels.cpu(), cmap, title=title, alpha=alpha)
    # ax.view_init(elev=100., azim=-60.)

    if not show_ax:
        ax.axis('off')

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def point_cloud_on_axis(ax, points, c, cmap=None, marker='.', title='', label='', alpha=1.):
    if isinstance(points, torch.Tensor):
        points = points.cpu()

    points = points.squeeze()

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, cmap=cmap, marker=marker, label=label, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    if label:
        ax.legend()


def visualize_o3d_mesh(mesh: Union[Sequence[open3d.geometry.TriangleMesh], open3d.geometry.TriangleMesh],
                       title: str = '', show=True, savepath=None, alpha=1.):
    if not isinstance(mesh, Sequence):
        mesh = [mesh]

    visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in mesh],
                      triangles_list=[np.asarray(m.triangles) for m in mesh],
                      title=title, show=show, savepath=savepath, alpha=alpha)


def visualize_trimesh(vertices_list: Sequence[ArrayLike], triangles_list: Sequence[ArrayLike], title: str = '',
                      show=True, savepath=None, ax=None, alpha=1.):
    """

    :param vertices_list: list of vertices, shape (Vx3) tensors
    :param triangles_list: list of triangles, shape (Tx3) tensors, corresponding to the vertices
    :param title: figure title
    :param show: switch for calling pyplot show or not
    :param savepath: if not None, the figure will be saved to this path
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    colors = ['r', 'g', 'b', 'y', 'c']
    for i, (vertices, triangles) in enumerate(zip(vertices_list, triangles_list)):
        trimesh_on_axis(ax, vertices, triangles, colors[i], title, alpha=alpha)

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def trimesh_on_axis(ax, vertices, triangles, color='', title='', alpha=1., label='', facecolors=None):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
        triangles = triangles.cpu().numpy()

    if len(vertices) > 0:
        collection = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, alpha=alpha)
        if color:
            collection.set_color(color)
        elif facecolors is not None:
            collection.set_facecolor(facecolors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    if label:
        handles, labels = plt.gca().get_legend_handles_labels()
        if color:
            handles.append(Patch(facecolor=color, edgecolor=color, label=label, alpha=alpha))
            handler_map = None
        elif facecolors is not None:
            # proxy handle
            handle = Rectangle((0, 0), 1, 1, label=label)
            handles.append(handle)

            # handler which actually makes the legend entry
            handler = HandlerBremmCmap(alpha=alpha)
            handler_map = {handle: handler}
        else:
            return

        labels.append(label)
        ax.legend(handles=handles, labels=labels, handler_map=handler_map)


def plot_normals(coords, normals, ax=None, title='', show=True, savepath=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
        normals = normals.detach().cpu().numpy()

    ax.quiver(coords[:, 0], coords[:, 1], coords[:, 2], normals[:, 0], normals[:, 1], normals[:, 2],
              length=coords.max() * 0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if savepath is not None:
        fig.tight_layout()
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


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


def color_2d_points_bremm(points: torch.Tensor):
    """ Bremm et al.: "Assisted Descriptor Selection Based on Visual Comparative Data Analysis" (2011)

    :param points: x y coordinates (Nx2)
    :return:
    """
    # CIELab colorspace
    # fixed lightness value:
    L = 55

    # coordinates are a and b values (stretched to [-100,100], max-range would be [-128, 127])
    ab_range = [-100, 100]
    p_min = points.min(dim=0, keepdim=True)[0]
    p_max = points.max(dim=0, keepdim=True)[0]
    points_norm = (points - p_min) / (p_max - p_min)
    points_to_ab = points_norm * (ab_range[1] - ab_range[0]) + ab_range[0]

    colors = lab2rgb(np.concatenate([np.full((len(points), 1), fill_value=L), points_to_ab.cpu().numpy()], axis=1))
    return colors


def color_2d_mesh_bremm(vertices: torch.Tensor, triangles: torch.Tensor):
    tri_coords = torch.gather(vertices.unsqueeze(2).repeat(1, 1, 3), dim=0, index=triangles.unsqueeze(1).repeat(1, 2, 1))
    vertex_centroids = tri_coords.mean(2)  # arithmetic mean of the three vertices of a triangle are its centroid
    return color_2d_points_bremm(vertex_centroids)