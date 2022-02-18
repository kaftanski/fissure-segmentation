import os
from typing import Sequence, Tuple
from pytorch3d.vis import plotly_vis
import SimpleITK as sitk
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import vtk
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency, point_mesh_face_distance,
)
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, Pointclouds
from skimage.measure import marching_cubes
from torch import nn
from tqdm import tqdm

import data
import image_ops
from data import image2tensor
import open3d as o3d


mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

SHOW_3D_PLOTS = False


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
        # TODO: maybe just take voxels as points
        #   - sitk.BinaryThinning
        #   - voxels 2 points
        #   - ball-pivoting algorithm

        # TODO Different idea: no fitting, just use poisson reconstruction?
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
        plane = fit_3d_plane(trg_mesh)

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

        # TODO: experiment with hyper parameters
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
    # fissures_label_image = mesh2labelmap_dist(fissure_meshes, mask)
    fissures_label_image = mesh2labelmap_sampling(fissure_meshes, mask, num_samples=10**7)
    fissures_label_image.CopyInformation(fissures)
    print('DONE\n')
    return fissures_label_image


def mesh2labelmap_dist(fissure_meshes: Sequence[Tuple[torch.Tensor, torch.Tensor]], lung_mask: sitk.Image,
                       dist_threshold: float = 1.0):  # dist_threshold in mm
    mask_tensor = image2tensor(lung_mask, dtype=torch.bool)
    lung_points = torch.nonzero(mask_tensor) * torch.tensor(lung_mask.GetSpacing()[::-1])

    dist_to_fissures = torch.zeros(len(lung_points), len(fissure_meshes))
    for i, (fissure_verts, fissure_faces) in enumerate(fissure_meshes):
        print(f'Fissure {i} ...')
        dist_to_fissures[:, i] = dist_to_mesh(lung_points, fissure_verts, fissure_faces)

    min_dist_label = torch.argmin(dist_to_fissures, dim=1, keepdim=True)
    lung_points_label = torch.where(torch.take_along_dim(dist_to_fissures, indices=min_dist_label, dim=1) <= dist_threshold,
                                    min_dist_label, -1) + 1
    fissures_label_tensor = torch.zeros_like(mask_tensor)
    fissures_label_tensor[torch.nonzero(mask_tensor, as_tuple=True)] = lung_points_label.squeeze()
    fissures_label_image = sitk.GetImageFromArray(fissures_label_tensor.numpy().astype(np.uint8))
    return fissures_label_image


def mesh2labelmap_sampling(fissure_meshes: Sequence[Tuple[torch.Tensor, torch.Tensor]], lung_mask: sitk.Image,
                           num_samples: int = 10000):
    spacing = torch.tensor(lung_mask.GetSpacing()[::-1])
    mask_tensor = image2tensor(lung_mask, dtype=torch.bool)
    label_tensor = torch.zeros(mask_tensor.shape, dtype=torch.long)
    for i, (fissure_verts, fissure_faces) in enumerate(fissure_meshes):
        meshes = Meshes(verts=[fissure_verts], faces=[fissure_faces])
        samples = sample_points_from_meshes(meshes, num_samples=num_samples)
        samples /= spacing.to(samples.device)
        fissure_ind = samples.squeeze().floor().long()
        label_tensor[fissure_ind[:, 0], fissure_ind[:, 1], fissure_ind[:, 2]] = i+1  # TODO: why index out of bounds?

    fissures_label_image = sitk.GetImageFromArray(label_tensor.numpy().astype(np.uint8))
    return fissures_label_image


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

    locator = vtk.vtkCellLocator()
    locator.SetDataSet(triangle_poly_data)
    locator.BuildLocator()

    # calculate distance from every point to mesh
    squared_distances = []
    for p in tqdm(input_points, desc='Closest point on mesh'):
        pred_point = p.tolist()

        closest_point = [0, 0, 0]
        closest_cell_id = vtk.reference(0)
        sub_ID = vtk.reference(0)  # unused
        squared_dist = vtk.reference(0.0)
        locator.FindClosestPoint(pred_point, closest_point, closest_cell_id, sub_ID, squared_dist)
        squared_distances.append(float(squared_dist))

    return torch.tensor(squared_distances).sqrt()


def poisson_reconstruction(fissures):
    # transforming labelmap to unit spacing
    # fissures = image_ops.resample_equal_spacing(fissures, target_spacing=1.)

    fissures_tensor = image2tensor(fissures).long()
    regularized_fissure_tensor = torch.zeros_like(fissures_tensor)
    spacing = fissures.GetSpacing()[::-1]  # spacing in zyx format

    # fit plane to each separate fissure
    labels = fissures_tensor.unique()[1:]
    for f in labels:
        print(f'Fitting fissure {f} ...')
        # extract the current fissure and construct independent image
        label_tensor = (fissures_tensor == f)
        label_image = sitk.GetImageFromArray(label_tensor.numpy().astype(int))
        label_image.CopyInformation(fissures)

        # thin the fissures
        print('\tThinning labelmap and extracting points ...')
        label_image = sitk.BinaryThinning(label_image)
        label_tensor = image2tensor(label_image, dtype=torch.bool)

        # extract point cloud from thinned fissures
        fissure_points = torch.nonzero(label_tensor) * torch.tensor(spacing)

        # convert to open3d point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fissure_points)
        pcd.estimate_normals()

        # compute the mesh
        print('\tPerforming Poisson reconstruction ...')
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6, width=0, scale=1.1, linear_fit=False)[0]

        # cropping
        print('\tPost-processing ...')
        bbox = pcd.get_axis_aligned_bounding_box()
        poisson_mesh_crop = poisson_mesh.crop(bbox)

        # convert mesh to labelmap by sampling points
        print('\tConverting mesh to labelmap ...')
        samples = poisson_mesh_crop.sample_points_uniformly(number_of_points=10**7)
        fissure_samples = torch.from_numpy(np.asarray(samples.points))
        fissure_samples /= torch.tensor(spacing)
        fissure_samples = fissure_samples.long()
        regularized_fissure_tensor[fissure_samples[:, 0], fissure_samples[:, 1], fissure_samples[:, 2]] = f

    regularized_fissures = sitk.GetImageFromArray(regularized_fissure_tensor.numpy().astype(np.uint8))
    regularized_fissures.CopyInformation(fissures)
    print('DONE\n')
    return regularized_fissures


def regularize_fissure_segmentations(mode):
    # load data
    base_dir = '/home/kaftan/FissureSegmentation/data'
    ds = data.LungData(base_dir)
    for i in range(len(ds)):
        file = ds.get_filename(i)

        if 'COPD' in file or 'EMPIRE02' in file:
            print('skipping COPD image')
            continue

        print(f'Regularizing fissures for image: {file.split(os.sep)[-1]}')
        img, fissures = ds[i]
        if fissures is None:
            print('\tno fissure segmentation found, skipping.')
            continue

        if mode == 'plane':
            mask = ds.get_lung_mask(i)
            fissures_reg = fit_plane_to_fissure(fissures, mask)
        elif mode == 'poisson':
            fissures_reg = poisson_reconstruction(fissures)
        else:
            raise ValueError(f'No regularization mode named "{mode}".')

        output_file = file.replace('_img_', f'_fissures_{mode}_')
        sitk.WriteImage(fissures_reg, output_file)


if __name__ == '__main__':
    regularize_fissure_segmentations(mode='poisson')
    # result = poisson_reconstruction(sitk.ReadImage('../data/EMPIRE16_fissures_fixed.nii.gz'))
    # sitk.WriteImage(result, 'results/EMPIRE16_fissures_reg_fixed.nii.gz')
