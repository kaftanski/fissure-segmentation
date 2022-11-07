from typing import Sequence, Tuple

import SimpleITK as sitk
import math
import numpy as np
import pytorch3d.structures
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from skimage.measure import marching_cubes
from torch import nn
from tqdm import tqdm

from data import image2tensor, ImageDataset
from metrics import point_surface_distance

SHOW_3D_PLOTS = True


class Plane(nn.Module):
    def __init__(self):
        super(Plane, self).__init__()
        self.normal = nn.Parameter(torch.ones(1, 1, 3))
        self.offset = nn.Parameter(torch.ones(1, 1, 3))

    def forward(self, x):
        # ensure normal vector has unit length
        with torch.no_grad():
            self.normal /= self.normal.norm()

        # scalar product of input vector (translated with offset) with normal vector
        # equals 0 if x lies on the plane
        return ((x - self.offset) * self.normal).sum(-1)

    def get_sample_points(self, n=5000, dim=0, range1=(-1, 1), range2=(-1, 1), return_faces=False):
        device = self.normal.data.device

        steps = int(math.sqrt(n))
        x = torch.linspace(range1[0], range1[1], steps=steps, device=device)
        y = torch.linspace(range2[0], range2[1], steps=steps, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        dims = [0, 1, 2]
        dims.remove(dim)
        x_dim, y_dim = dims
        with torch.no_grad():
            # ensure normal vector has unit length
            self.normal /= self.normal.norm()
            z = ((grid_x - self.offset[0, 0, x_dim]) * self.normal[0, 0, x_dim] +
                 (grid_y - self.offset[0, 0, y_dim]) * self.normal[0, 0, y_dim]) / self.normal[0, 0, dim] + self.offset[0, 0, dim]

        points = torch.stack([grid_x.reshape(-1), grid_x.reshape(-1), z.reshape(-1)], dim=1)
        if return_faces:
            # create faces
            faces = []
            for j in range(steps-1):
                for i in range(steps-1):
                    cur = j*steps + i
                    faces.append([cur, cur+1, cur+steps])
                    faces.append([cur+1, cur+steps, cur+1+steps])

            return points, torch.tensor(faces, device=device)
        else:
            return points


def plot_pointcloud(x, y, z, title=""):
    x = x.detach().cpu()
    y = y.detach().cpu()
    z = z.detach().cpu()

    fig = plt.figure(figsize=(7, 6))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


def plot_pointclouds(x1, y1, z1, x2, y2, z2, title=""):
    x1 = x1.detach().cpu()
    y1 = y1.detach().cpu()
    z1 = z1.detach().cpu()

    x2 = x2.detach().cpu()
    y2 = y2.detach().cpu()
    z2 = z2.detach().cpu()

    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x1, z1, -y1, c='r')
    ax.scatter3D(x2, z2, -y2, c='g')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


def plot_mesh(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    plot_pointcloud(x, y, z, title)


def rigid_fit_3d_plane(mesh: pytorch3d.structures.Meshes):
    num_iter = 2000
    plot_period = num_iter
    loop = tqdm(range(num_iter))

    plane = Plane().to(mesh.device)

    losses = []

    optimizer = torch.optim.Adam(plane.parameters(), lr=0.01)

    for i in loop:
        # We sample 5k points from the surface of the mesh
        sample_trg = sample_points_from_meshes(mesh, 5000)

        # We compare the mesh with the plane
        scalar_products = plane(sample_trg)
        loss = scalar_products.pow(2).mean()

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss
        loop.set_description('Fitting plane ... Current loss = %.4f' % loss)
        losses.append(loss.item())

        # Plot plane
        if SHOW_3D_PLOTS and (i + 1) % plot_period == 0:
            pts, faces = plane.get_sample_points(2500, return_faces=True)
            plot_pointcloud(pts[:, 0], pts[:, 1], pts[:, 2], title="iter: %d" % i)
            # fig = plotly_vis.plot_scene({"Initial Plane": {"plane mesh": Meshes([pts], [faces])}})
            # fig.show()

    plt.figure()
    plt.plot(losses)
    plt.title('Plane fitting loss.')
    plt.show()

    return plane


def fit_plane_to_fissure(fissures: sitk.Image, mask: sitk.Image):
    # Set the device
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:3")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mask_tensor = image2tensor(mask, dtype=torch.bool)
    fissures_tensor = image2tensor(fissures).long()
    fissure_meshes = []
    spacing = fissures.GetSpacing()[::-1]  # spacing in zyx format

    # fit plane to each separate fissure
    labels = fissures_tensor.unique()[1:]
    for f in labels:
        print(f'Fitting fissure {f} ...')
        # construct the 3d object from label image
        verts, faces, normals, values = marching_cubes(volume=(fissures_tensor == f).numpy(), level=0.5,
                                                       spacing=spacing, allow_degenerate=False,
                                                       mask=mask_tensor.numpy())

        # results to torch tensors
        verts = torch.from_numpy(verts).to(device)  # (Vx3)
        faces_idx = torch.from_numpy(faces).to(device)  # (Fx3)

        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        # We construct a Meshes structure for the target mesh
        trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

        # construct the source shape: simple plane that is fitted to target point cloud
        plane = rigid_fit_3d_plane(trg_mesh)

        # get a mesh from the plane
        n_plane_points = 2500
        plane_verts, plane_faces = plane.get_sample_points(
            n=n_plane_points, dim=0,
            range1=(verts[:, 1].min(), verts[:, 1].max()),  # x is in range of the target vertices
            range2=(verts[:, 2].min(), verts[:, 2].max()),  # y as well
            return_faces=True)
        # # transform z-coord into same range as the target as well
        # plane_verts[:, 0] = (plane_verts[:, 0] - plane_verts[:, 0].min()) / (plane_verts[:, 0].max() - plane_verts[:, 0].min()) * (verts[:, 0].max() - verts[:, 0].min()) + verts[:, 0].min()
        src_mesh = Meshes(verts=[plane_verts], faces=[plane_faces])

        # visualize starting point
        if SHOW_3D_PLOTS:
            plot_pointclouds(plane_verts[:, 0], plane_verts[:, 1], plane_verts[:, 2], verts[:, 0], verts[:, 1], verts[:, 2], "Plane over target point cloud")
            plot_mesh(trg_mesh, "Target mesh")
            plot_mesh(src_mesh, "Source mesh")

        # We will learn to deform the source mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices in src_mesh
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

        # The optimizer
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

        # Number of optimization steps
        num_iter = 2000
        # Weight for the chamfer loss
        w_chamfer = 1
        # Weight for mesh edge loss
        w_edge = 1.0
        # Weight for mesh normal consistency
        w_normal = 0.01
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.1
        # Weight for point to mesh distance
        w_point_mesh = 0
        # Plot period for the losses
        plot_period = num_iter
        loop = tqdm(range(num_iter))

        n_sample = 5000
        chamfer_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []
        point_mesh_losses = []

        for i in loop:
            # Initialize optimizer
            optimizer.zero_grad()

            # Deform the mesh
            new_src_mesh = src_mesh.offset_verts(deform_verts)

            # We sample 5k points from the surface of each mesh
            sample_trg = sample_points_from_meshes(trg_mesh, n_sample)
            sample_src = sample_points_from_meshes(new_src_mesh, n_sample)

            # We compare the two sets of pointclouds by computing (a) the chamfer loss
            loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

            # and (b) the edge length of the predicted mesh
            loss_edge = mesh_edge_loss(new_src_mesh)

            # mesh normal consistency
            loss_normal = mesh_normal_consistency(new_src_mesh)

            # mesh laplacian smoothing
            loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

            # point to mesh loss
            # loss_point_mesh = point_mesh_face_distance(trg_mesh, Pointclouds([sample_src.squeeze()]))
            # loss_point_mesh = 0
            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + \
                   loss_edge * w_edge + \
                   loss_normal * w_normal + \
                   loss_laplacian * w_laplacian #+ \
                   # loss_point_mesh * w_point_mesh

            # Print the losses
            loop.set_description('Fitting mesh ... Total loss = %.6f' % loss)

            # Save the losses for plotting
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            normal_losses.append(float(loss_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))
            # point_mesh_losses.append(float(loss_point_mesh.detach().cpu()))

            # Plot mesh
            if SHOW_3D_PLOTS and (i+1) % plot_period == 0:
                plot_mesh(new_src_mesh, title="iter: %d" % i)

            # Optimization step
            loss.backward()
            optimizer.step()

        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        ax.plot(chamfer_losses, label="chamfer loss")
        ax.plot(edge_losses, label="edge loss")
        ax.plot(normal_losses, label="normal loss")
        ax.plot(laplacian_losses, label="laplacian loss")
        # ax.plot(point_mesh_losses, label='point to mesh loss')
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title(f"Loss progression (Fissure {f})", fontsize="16")
        plt.show()

        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
        final_verts = final_verts.detach()
        final_verts.requires_grad_(False)

        # Scale normalize back to the original target size
        final_verts = final_verts * scale + center

        fissure_meshes.append((final_verts, final_faces))

    # convert points into labelmap
    print('Converting fissure meshes to labelmap ...')
    fissures_label_tensor = mesh2labelmap_dist(fissure_meshes, output_shape=fissures_tensor.shape,
                                               img_spacing=fissures.GetSpacing(), mask=mask_tensor, dist_threshold=1.0)
    # fissures_label_tensor = mesh2labelmap_sampling(fissure_meshes, output_shape=fissures_tensor.shape,
    #                                                img_spacing=fissures.GetSpacing(), num_samples=10**7)
    fissures_label_image = sitk.GetImageFromArray(fissures_label_tensor.numpy().astype(np.uint8))
    fissures_label_image.CopyInformation(fissures)
    print('DONE\n')
    return fissures_label_image


def mesh2labelmap_dist(meshes: Sequence[Tuple[torch.Tensor, torch.Tensor]], output_shape: Sequence[int],
                       img_spacing: Sequence[float], dist_threshold: float = 1.0, mask: torch.Tensor = None) -> torch.Tensor:
    """ Constructs a label map from meshes based on a pixel's distance.
        Output labels will be in {1, 2, ..., len(meshes)}.

    :param meshes: multiple meshes consisting of vertices and triangle indices, each getting its own label
    :param output_shape: shape D,H,W of the output labelmap
    :param img_spacing: the image spacing in x,y and z dimension
    :param dist_threshold: minimum distance to mesh (in mm) where a point will be assigned the corresponding label
    :param mask: optional mask to specify query points (distance will be computed from every nonzero element). Shape (DxHxW)
    :return: labelmap
    """
    if mask is not None:
        query_indices = torch.nonzero(mask)
    else:
        query_indices = torch.nonzero(torch.ones(*output_shape))

    dist_to_meshes = torch.zeros(len(query_indices), len(meshes))
    for i, (verts, faces) in enumerate(meshes):
        dist_to_meshes[:, i] = point_surface_distance(query_indices.cpu() * torch.tensor(img_spacing[::-1]), verts.cpu(), faces.cpu())

    min_dist_label = torch.argmin(dist_to_meshes, dim=1, keepdim=True)
    labelled_points = torch.where(torch.take_along_dim(dist_to_meshes, indices=min_dist_label, dim=1) <= dist_threshold,
                                  min_dist_label, -1) + 1
    label_tensor = torch.zeros(*output_shape, dtype=torch.long)
    label_tensor[query_indices[:, 0], query_indices[:, 1], query_indices[:, 2]] = labelled_points.squeeze()
    return label_tensor


if __name__ == '__main__':
    ds = ImageDataset('../data')
    i = 0
    plane_img = fit_plane_to_fissure(ds.get_fissures(i), ds.get_lung_mask(i))
    sitk.WriteImage(plane_img, 'results/fit_plane.nii.gz')
