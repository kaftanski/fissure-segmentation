import os
import time
from typing import Sequence, Tuple, List

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from numpy.typing import ArrayLike
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

import data
from data import image2tensor
from data_processing.surface_fitting_optimization import fit_plane_to_fissure
from utils.general_utils import mask_out_verts_from_mesh, mask_to_points, remove_all_but_biggest_component, save_meshes


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
        label_tensor[indices[:, 0], indices[:, 1], indices[:, 2]] = i+1

    return label_tensor


def pointcloud_surface_fitting(points: ArrayLike, crop_to_bbox=False, mask: sitk.Image = None, depth=6, width=0, scale=1.1) -> o3d.geometry.TriangleMesh:
    """

    :param points: (Nx3)
    :param crop_to_bbox: crop the resulting mesh to the bounding box of the initial point cloud
    :param mask: binary mask image for vertices of the mesh (e.g. lung mask), will be dilated to prevent artifacts
    :param depth: octree depth, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :param width: width, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :param scale: scale, parameter for o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    :return:
    """
    if np.prod(points.shape) == 0 or points.shape[0] < 4:
        raise ValueError(f'Tried reconstructing mesh from {points.shape[0]} points. Requires at least 4.')

    # convert to open3d point cloud object
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # pcd.points = o3d.utility.Vector3dVector(points)

    # very important: make normals consistent and thus prevents weird loops in the reconstruction
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    # compute the mesh
    print(f'Poisson reconstruction from {pcd}')
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=width, scale=scale, linear_fit=False, n_threads=1)[0]

    # cropping
    if crop_to_bbox:
        print('Cropping to bbox')
        bbox = pcd.get_axis_aligned_bounding_box()
        poisson_mesh = poisson_mesh.crop(bbox)

    # masking
    if mask is not None:
        mask = sitk.BinaryDilate(mask, kernelRadius=(1, 1, 1))
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool))
        spacing = torch.tensor(mask.GetSpacing())
        mask_out_verts_from_mesh(poisson_mesh, mask_tensor, spacing)

    return poisson_mesh


def poisson_reconstruction(fissures: sitk.Image, mask: sitk.Image, return_times=False):
    print('Performing surface fitting via Poisson Reconstruction')
    # transforming labelmap to unit spacing
    # fissures = image_ops.resample_equal_spacing(fissures, target_spacing=1.)

    fissures_tensor = image2tensor(fissures).long()
    fissure_meshes = []
    spacing = fissures.GetSpacing()

    # fit plane to each separate fissure
    labels = fissures_tensor.unique()[1:]
    times = []
    for f in labels:
        print(f'Fitting fissure {f} ...')
        # extract the current fissure and construct independent image
        label_tensor = (fissures_tensor == f)
        label_image = sitk.GetImageFromArray(label_tensor.numpy().astype(int))
        label_image.CopyInformation(fissures)

        # thin the fissures
        print('\tThinning labelmap and extracting points ...')
        start = time.time()
        label_image = sitk.BinaryThinning(label_image)
        label_tensor = image2tensor(label_image, dtype=torch.bool)

        # extract point cloud from thinned fissures
        fissure_points = mask_to_points(label_tensor, spacing)
        times.append(time.time() - start)
        print(f'\tTook {time.time() - start:.4f} s')

        # compute the mesh
        print('\tPerforming Poisson reconstruction ...')
        start = time.time()
        poisson_mesh = pointcloud_surface_fitting(fissure_points, crop_to_bbox=True, mask=mask)

        # post-process: keep only the largest component (that is in the correct body half)
        right = f > 1  # right fissure(s) are label 2 and 3
        remove_all_but_biggest_component(poisson_mesh, right=right,
                                         center_x=(fissures.GetSize()[0] * fissures.GetSpacing()[0]) / 2)
        fissure_meshes.append(poisson_mesh)
        times[-1] += time.time() - start
        print(f'\tTook {time.time() - start:.4f} s')

    # convert mesh to labelmap by sampling points
    print('Converting meshes to labelmap ...')
    regularized_fissure_tensor = o3d_mesh_to_labelmap(fissure_meshes, shape=fissures_tensor.shape, spacing=spacing)
    regularized_fissures = sitk.GetImageFromArray(regularized_fissure_tensor.numpy().astype(np.uint8))
    regularized_fissures.CopyInformation(fissures)

    print('DONE\n')
    if return_times:
        return regularized_fissures, fissure_meshes, times
    else:
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
        if len(mesh.vertices) == 0:
            continue
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
            save_meshes(meshes, base_dir, case, sequence)
        else:
            raise ValueError(f'No regularization mode named "{mode}".')

        output_file = file.replace('_img_', f'_fissures_{mode}_')
        sitk.WriteImage(fissures_reg, output_file)


if __name__ == '__main__':
    regularize_fissure_segmentations(mode='poisson')
    # result = poisson_reconstruction(sitk.ReadImage('../data/EMPIRE16_fissures_fixed.nii.gz'))
    # sitk.WriteImage(result, 'results/EMPIRE16_fissures_reg_fixed.nii.gz')
