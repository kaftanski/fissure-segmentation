import os
from typing import Sequence, Tuple, List
from utils import mask_out_verts_from_mesh, mask_to_points
import pytorch3d.structures
from numpy.typing import ArrayLike
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


def mesh2labelmap_sampling(meshes: Sequence[Tuple[torch.Tensor, torch.Tensor]], output_shape: Sequence[int],
                           img_spacing: Sequence[float], num_samples: int = 10**7) -> torch.Tensor:
    """ Constructs a label map from meshes by sampling points and placing them into the image grid.
        Output labels will be in {1, 2, ..., len(meshes)}.

    :param meshes: multiple meshes consisting of vertices and triangle indices, each getting its own label
    :param output_shape: shape D,H,W of the output labelmap
    :param img_spacing: the image spacing in x,y and z dimension
    :param num_samples: number of samples to generate per mesh
    :return: labelmap
    """
    spacing = torch.tensor(img_spacing[::-1])
    label_tensor = torch.zeros(*output_shape, dtype=torch.long)
    for i, (verts, faces) in enumerate(meshes):
        meshes = Meshes(verts=[verts], faces=[faces])
        samples = sample_points_from_meshes(meshes, num_samples=num_samples)
        samples /= spacing.to(samples.device)
        indices = samples.squeeze().floor().long()
        label_tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = i+1  # TODO: why index out of bounds?

    return label_tensor


def point_surface_distance(query_points: ArrayLike, trg_points: ArrayLike, trg_tris: ArrayLike) -> torch.Tensor:
    """ Parallel unsigned distance computation from N query points to a target triangle mesh using Open3d.

    :param query_points: query points for distance computation. ArrayLike of shape (Nx3)
    :param trg_points: vertices of the target mesh. ArrayLike of shape (Vx3)
    :param trg_tris: shared edge triangle index list of the target mesh. ArrayLike of shape (Tx3)
    :return: euclidean distance from every input point to the closest point on the target mesh. Tensor of shape (N)
    """
    # construct ray casting scene with target mesh in it
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(vertex_positions=np.array(trg_points, dtype=np.float32), triangle_indices=np.array(trg_tris, dtype=np.uint32))  # we do not need the geometry ID for mesh

    # distance computation
    dist = scene.compute_distance(np.array(query_points, dtype=np.float32))
    return torch.utils.dlpack.from_dlpack(dist.to_dlpack())


def pointcloud_to_mesh(points: ArrayLike, crop_to_bbox=False, mask: sitk.Image = None, depth=6, width=0, scale=1.1) -> o3d.geometry.TriangleMesh:
    """

    :param points: (Nx3)
    :param crop_to_bbox: crop the resulting mesh to the bounding box of the initial point cloud
    :param mask: binary mask image for vertices of the mesh (e.g. lung mask), will be dilated to prevent artifacts
    :param depth: octree depth, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :param width: width, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :param scale: scale, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :return:
    """
    # convert to open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # very important: make normals consistent and thus prevents weird loops in the reconstruction
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    # compute the mesh
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=width, scale=scale, linear_fit=False)[0]

    # cropping
    if crop_to_bbox:
        bbox = pcd.get_axis_aligned_bounding_box()
        poisson_mesh = poisson_mesh.crop(bbox)

    # masking
    if mask is not None:
        mask = sitk.BinaryDilate(mask, kernelRadius=(1, 1, 1))
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool))
        spacing = torch.tensor(mask.GetSpacing())
        mask_out_verts_from_mesh(poisson_mesh, mask_tensor, spacing)

    return poisson_mesh


def poisson_reconstruction(fissures: sitk.Image, mask: sitk.Image):
    print('Performing surface fitting via Poisson Reconstruction')
    # transforming labelmap to unit spacing
    # fissures = image_ops.resample_equal_spacing(fissures, target_spacing=1.)

    fissures_tensor = image2tensor(fissures).long()
    fissure_meshes = []
    spacing = fissures.GetSpacing()

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
        fissure_points = mask_to_points(label_tensor, spacing)

        # compute the mesh
        print('\tPerforming Poisson reconstruction ...')
        poisson_mesh = pointcloud_to_mesh(fissure_points, crop_to_bbox=True, mask=mask)
        fissure_meshes.append(poisson_mesh)

    # convert mesh to labelmap by sampling points
    print('Converting meshes to labelmap ...')
    regularized_fissure_tensor = o3d_mesh_to_labelmap(fissure_meshes, shape=fissures_tensor.shape, spacing=spacing)
    regularized_fissures = sitk.GetImageFromArray(regularized_fissure_tensor.numpy().astype(np.uint8))
    regularized_fissures.CopyInformation(fissures)

    print('DONE\n')
    return regularized_fissures, fissure_meshes


def o3d_mesh_to_labelmap(o3d_meshes: List[o3d.geometry.TriangleMesh], shape, spacing: Tuple[float], n_samples=10**7) -> torch.Tensor:
    """

    :param o3d_meshes: list of open3d TriangleMesh to convert into one labelmap
    :param shape: shape D,H,W of the output labelmap
    :param spacing: the image spacing in x,y and z dimension
    :param n_samples: number of samples used to convert to a labelmap
    :return: labelmap tensor of given shape
    """
    label_tensor = torch.zeros(*shape, dtype=torch.long)

    for i, mesh in enumerate(o3d_meshes):
        samples = mesh.sample_points_uniformly(number_of_points=n_samples)
        fissure_samples = torch.from_numpy(np.asarray(samples.points))
        fissure_samples /= torch.tensor(spacing)
        fissure_samples = fissure_samples.long().flip(-1)

        # prevent index out of bounds
        for d in range(len(shape)):
            fissure_samples = fissure_samples[fissure_samples[:, d] < shape[d]]

        label_tensor[fissure_samples[:, 0], fissure_samples[:, 1], fissure_samples[:, 2]] = i+1

    return label_tensor


def regularize_fissure_segmentations(mode):
    # load data
    base_dir = '/home/kaftan/FissureSegmentation/data'
    ds = data.LungData(base_dir)
    for i in range(len(ds)):
        file = ds.get_filename(i)

        # if 'COPD' in file:
        #     print('skipping COPD image')
        #     continue

        print(f'Regularizing fissures for image: {file.split(os.sep)[-1]}')
        if ds.fissures[i] is None:
            print('\tno fissure segmentation found, skipping.\n')
            continue

        img, fissures = ds[i]

        if mode == 'plane':
            mask = ds.get_lung_mask(i)
            fissures_reg = fit_plane_to_fissure(fissures, mask)
        elif mode == 'poisson':
            fissures_reg, meshes = poisson_reconstruction(fissures, ds.get_lung_mask(i))
            case, sequence = ds.get_id(i)
            meshdir = os.path.join(base_dir, f"{case}_mesh_{sequence}")
            os.makedirs(meshdir, exist_ok=True)
            for m, mesh in enumerate(meshes):
                o3d.io.write_triangle_mesh(os.path.join(meshdir, f'{case}_fissure{m+1}_{sequence}.obj'), mesh)
        else:
            raise ValueError(f'No regularization mode named "{mode}".')

        output_file = file.replace('_img_', f'_fissures_{mode}_')
        sitk.WriteImage(fissures_reg, output_file)


if __name__ == '__main__':
    regularize_fissure_segmentations(mode='poisson')
    # result = poisson_reconstruction(sitk.ReadImage('../data/EMPIRE16_fissures_fixed.nii.gz'))
    # sitk.WriteImage(result, 'results/EMPIRE16_fissures_reg_fixed.nii.gz')
