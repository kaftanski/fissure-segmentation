import math
import os

import data
from data import image2tensor
import SimpleITK as sitk
import torch
from torch import nn
from skimage.measure import marching_cubes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import vtk


mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80


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


def fit_3d_plane(mesh):
    Niter = 2000
    plot_period = 500
    loop = tqdm(range(Niter))

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
        loop.set_description('total_loss = %.4f' % loss)
        losses.append(loss.item())

        # Plot plane
        if i % plot_period == 0:
            pts = plane.get_sample_points()
            plot_pointcloud(pts[:, 0], pts[:, 1], pts[:, 2], title="iter: %d" % i)

    plt.figure()
    plt.plot(losses)
    plt.title('Plane fitting loss.')
    plt.show()

    return plane


def fit_surface_to_fissure(fissures: sitk.Image, mask: sitk.Image, lobescribble: sitk.Image):
    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")

    mask_tensor = image2tensor(mask, dtype=torch.bool)
    fissures_tensor = image2tensor(fissures)
    fissure_meshes = []

    # fit plane to each separate fissure
    labels = fissures_tensor.unique()[1:]
    for f in labels:
        # construct the 3d object from label image
        # TODO: maybe just take voxels as points
        verts, faces, normals, values = marching_cubes(volume=(fissures_tensor == f).numpy(), level=0.5,
                                                       spacing=fissures.GetSpacing()[::-1], allow_degenerate=False,
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

        # We initialize the source shape to be a sphere of radius 1
        n_sample = 4900
        plane = fit_3d_plane(trg_mesh)
        plane_verts, plane_faces = plane.get_sample_points(n=n_sample, dim=0, range1=(verts[:, 1].min(), verts[:, 1].max()), range2=(verts[:, 2].min(), verts[:, 2].max()), return_faces=True)
        src_mesh = Meshes(verts=[plane_verts], faces=[plane_faces])

        # visualize starting point
        plot_pointclouds(plane_verts[:, 0], plane_verts[:, 1], plane_verts[:, 2], verts[:, 0], verts[:, 1], verts[:, 2], "Plane over target point cloud")
        plot_mesh(trg_mesh, "Target mesh")
        plot_mesh(src_mesh, "Source mesh")

        # We will learn to deform the source mesh by offsetting its vertices
        # The shape of the deform parameters is equal to the total number of vertices in src_mesh
        deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

        # The optimizer
        optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

        # Number of optimization steps
        Niter = 2000
        # Weight for the chamfer loss
        w_chamfer = 1.0
        # Weight for mesh edge loss
        w_edge = 1.0
        # Weight for mesh normal consistency
        w_normal = 0.01
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.1
        # Plot period for the losses
        plot_period = 500
        loop = tqdm(range(Niter))

        chamfer_losses = []
        laplacian_losses = []
        edge_losses = []
        normal_losses = []

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

            # Weighted sum of the losses
            loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

            # Print the losses
            loop.set_description('total_loss = %.6f' % loss)

            # Save the losses for plotting
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            normal_losses.append(float(loss_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))

            # Plot mesh
            if i % plot_period == 0:
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
        ax.legend(fontsize="16")
        ax.set_xlabel("Iteration", fontsize="16")
        ax.set_ylabel("Loss", fontsize="16")
        ax.set_title(f"Loss progression (Fissure {f})", fontsize="16")
        plt.show()

        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

        # Scale normalize back to the original target size
        final_verts = final_verts * scale + center

        fissure_meshes.append((final_verts, final_faces))

    # convert points into labelmap
    lung_points = torch.nonzero(mask_tensor) * torch.tensor(fissures.GetSpacing()[::-1])
    dist_to_fissures = torch.zeros(len(labels), len(lung_points))
    for i, (fissure_verts, fissure_faces) in enumerate(fissure_meshes):
        dist_to_fissures[i] = dist_to_mesh(lung_points, fissure_verts, fissure_faces)

    dist_threshold = 1.5  # mm
    min_dist_label = torch.argmin(dist_to_fissures, dim=0) + 1
    fissures_label = torch.where(dist_to_fissures[min_dist_label] <= dist_threshold, min_dist_label, 0)
    fissures_label_image = sitk.GetImageFromArray(fissures_label.numpy())
    fissures_label_image.CopyInformation(fissures)
    return fissures_label_image


def regularize_fissure_segmentations():
    # load data
    base_dir = '/home/kaftan/FissureSegmentation/data'
    ds = data.LungData(base_dir)

    for i in range(len(ds)):
        file = ds.get_filename(i)
        img, fissures = ds[i]
        mask = ds.get_lung_mask(i)
        lobescribbles = ds.get_lobescribbles(i)
        fissures_reg = fit_surface_to_fissure(fissures, mask, lobescribbles)
        output_file = file.replace('_img_', '_fissures_reg_')
        sitk.WriteImage(fissures_reg, output_file)


def dist_to_mesh(input_points, trg_verts, trg_tris):
    """

    :param input_points:
    :param trg_verts:
    :param trg_tris:
    :return: tensor containing euclidean distance from every input point to the closest point on the target mesh
    """
    # construct vtk points object
    vtk_target_points = vtk.vtkPoints()
    for v in trg_verts:
        targ_point = v.tolist()
        vtk_target_points.InsertNextPoint(targ_point)

    # construct corresponding triangles
    vtk_triangles = vtk.vtkCellArray()
    for f in trg_tris:
        v1_index, v2_index, v3_index = f

        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, v1_index)
        triangle.GetPointIds().SetId(1, v2_index)
        triangle.GetPointIds().SetId(2, v3_index)

        vtk_triangles.InsertNextCell(triangle)

    triangle_poly_data = vtk.vtkPolyData()
    triangle_poly_data.SetPoints(vtk_target_points)
    triangle_poly_data.SetPolys(vtk_triangles)

    # # validate correct conversion of numpy matrix into vtk Polydata (with paraview)
    # writer = vtk.vtkPolyDataWriter()
    # writer.SetFileName("/share/data_hastig1/kaftan/projects/bachelor/Implementation/Temp/polydata.vtk")
    # writer.SetInputData(triangle_poly_data)
    # writer.Write()

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(triangle_poly_data)
    locator.BuildLocator()

    # calculate distance from every point to mesh
    squared_distances = []
    for p in input_points:
        pred_point = p.tolist()

        closest_point = [0, 0, 0]
        closest_cell_id = vtk.reference(0)
        sub_ID = vtk.reference(0)  # unused
        squared_dist = vtk.reference(0.0)
        locator.FindClosestPoint(pred_point, closest_point, closest_cell_id, sub_ID, squared_dist)
        squared_distances.append(float(squared_dist))

    return torch.tensor(squared_distances).sqrt()


if __name__ == '__main__':
    regularize_fissure_segmentations()
