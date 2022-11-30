import os.path
from typing import Tuple, List

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from skimage.measure import marching_cubes
from torch.nn import functional as F

from data import LungData
from data_processing.random_walk import compute_laplace_matrix, random_walk
from utils.general_utils import create_o3d_mesh
from visualization import visualize_o3d_mesh


def fill_lobes(lobes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # compute graph laplacian
    print('Computing graph Laplacian matrix')
    L = compute_laplace_matrix(lobes != 0, 'binary')

    # random walk lobe segmentation
    print('Performing random walk')
    probabilities = random_walk(L, labels=lobes, graph_mask=mask)

    # final lobe segmentation
    lobes_rw = torch.where(condition=mask.bool(), input=probabilities.argmax(-1) + 1, other=0)  # background is set to zero
    # assert torch.all(seeds[seeds != 0] == lobes[seeds != 0]), 'Seed scribbles have changed label'

    return lobes_rw


def lobes_to_fissures(lobes: sitk.Image, mask: sitk.Image, device='cuda:2'):
    print('Converting Lobes to Fissures ...')
    # convert SimpleITK images to tensors
    mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool)).bool()
    lobes_tensor = torch.from_numpy(sitk.GetArrayFromImage(lobes).astype(int)).long()

    # fill lobes with random walk
    lobes_filled = fill_lobes(lobes_tensor, mask_tensor)
    # plt.imshow(lobes_filled[lobes_filled.shape[0]//2])
    # plt.show()

    lobes_filled_img = sitk.GetImageFromArray(lobes_filled.numpy())
    lobes_filled_img.CopyInformation(lobes)

    lobes_one_hot = F.one_hot(lobes_filled).permute(3, 0, 1, 2).unsqueeze(0)
    print(lobes_one_hot.shape)

    # Lobe labels / one-hot channels:
    # right lower lobe: 1
    # right upper lobe: 2
    # left lower lobe: 3
    # left upper lobe: 4
    # right middle lobe: 5 (contained in label 2 if right horizontal fissure is not segmented)

    n_lobes = lobes_one_hot.shape[1] - 1  # excluding background

    # create overlapping structures in lobe-channels by channel-wise dilation
    dilation_kernel = torch.tensor([[[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]],
                                    [[0, 1, 0],
                                     [1, 1, 1],
                                     [0, 1, 0]],
                                    [[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]]).view(1, 1, 3, 3, 3).repeat(n_lobes+1, 1, 1, 1, 1)

    dilated_lobes_one_hot = F.conv3d(F.pad(lobes_one_hot.half().to(device), pad=(1, 1, 1, 1, 1, 1)),
                                     dilation_kernel.half().to(device), groups=n_lobes+1)

    # assemble the fissure segmentation (fissures at the boundaries of specific lobes)
    # left fissure (1): between lobes 3 & 4
    lf = torch.logical_and(dilated_lobes_one_hot[0, 3], dilated_lobes_one_hot[0, 4])
    fissure_tensor = torch.zeros_like(lf, dtype=torch.long)
    fissure_tensor[lf] = 1

    # right oblique fissure (2): between lobes 1 & 2 (and 1 & 5 if lobe 5 is present)
    rof = torch.logical_and(dilated_lobes_one_hot[0, 1], dilated_lobes_one_hot[0, 2])
    if n_lobes == 5:
        rof += torch.logical_and(dilated_lobes_one_hot[0, 1], dilated_lobes_one_hot[0, 5])
    fissure_tensor[rof] = 2

    # right horizontal fissure (3): between lobes 2 & 5 (if lobe 5 is present)
    if n_lobes == 5:
        rhf = torch.logical_and(dilated_lobes_one_hot[0, 2], dilated_lobes_one_hot[0, 5])
        fissure_tensor[rhf] = 3

    fissure_img = sitk.GetImageFromArray(fissure_tensor.cpu().numpy().astype(np.uint8))
    fissure_img.CopyInformation(lobes)
    return fissure_img, lobes_filled_img


def find_lobes(fissure_seg: sitk.Image, lung_mask: sitk.Image, exclude_rhf: bool = False) \
        -> Tuple[sitk.Image, List[o3d.geometry.TriangleMesh], bool]:
    """

    :param fissure_seg: fissure segmentation label image
    :param lung_mask: lung mask (binary image)
    :param exclude_rhf: exclude the right horizontal fissure, results in 4 instead of 5 lobes
    :return: the lobe segmentation label image
    """
    print('Computing lobe segmentation from fissures.')

    # change the right horizontal fissure to background, if it is to be excluded
    change_label_filter = sitk.ChangeLabelImageFilter()
    if exclude_rhf:
        change_label_filter.SetChangeMap({3: 0})
        fissure_seg = change_label_filter.Execute(fissure_seg)

    # post-process fissures
    # make fissure segmentation binary (disregard the different fissures)
    fissure_seg_binary = sitk.BinaryThreshold(fissure_seg, upperThreshold=0.5, insideValue=0, outsideValue=1)

    # create inverted lobe mask by combining fissures and not-lung
    lung_mask = sitk.Cast(lung_mask, sitk.sitkUInt8)
    lung_mask = sitk.BinaryErode(lung_mask, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.Or(sitk.Not(lung_mask), fissure_seg_binary)

    # close some gaps
    not_lobes = sitk.BinaryMorphologicalClosing(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.BinaryDilate(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)

    # find connected components in lobes mask
    num_lobes_target = 4 if exclude_rhf else 5
    lobes_mask = sitk.Not(not_lobes)
    lobes_mask = sitk.BinaryMorphologicalOpening(lobes_mask, kernelRadius=(4, 4, 4), kernelType=sitk.sitkBall)

    connected_component_filter = sitk.ConnectedComponentImageFilter()
    lobes_components = connected_component_filter.Execute(lobes_mask)
    obj_cnt = connected_component_filter.GetObjectCount()
    print(f'\tFound {obj_cnt} connected components ...')
    if obj_cnt < num_lobes_target:
        print(f'\tThis is not enough, skipping relabelling.')
        return lobes_components, [], False
    else:
        print('\tSUCCESS!')

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lobes_components_sorted = relabel_filter.Execute(lobes_components)
    print(f'\tThe {num_lobes_target} largest objects have sizes {relabel_filter.GetSizeOfObjectsInPhysicalUnits()[:num_lobes_target]}')

    # extract the 5 biggest objects (the 5 lobes)
    change_label_filter.SetChangeMap({l: 0 for l in range(num_lobes_target+1, relabel_filter.GetOriginalNumberOfObjects()+1)})
    biggest_n_components = change_label_filter.Execute(lobes_components_sorted)

    # relabel lobes (same as in Mattias' dir-lab COPD lobes)
    # right lower lobe: 1
    # right upper lobe: 2
    # left lower lobe: 3
    # left upper lobe: 4
    # right middle lobe: 5 (contained in label 2 if right horizontal fissure is not segmented)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(biggest_n_components)
    centroids = np.array([shape_stats.GetCentroid(l) for l in shape_stats.GetLabels()])
    sort_by_x = np.argsort(centroids[:, 0])

    num_right = 2 if exclude_rhf else 3
    right_lobes = sort_by_x[:num_right]  # smaller x is right
    left_lobes = sort_by_x[num_right:]  # higher x is left
    change_map = {}

    sort_left_by_z = np.argsort(centroids[left_lobes, 2])
    change_map[left_lobes[sort_left_by_z[0]] + 1.] = 3.  # lower in z
    change_map[left_lobes[sort_left_by_z[1]] + 1.] = 4.  # higher in z

    sort_right_by_z = np.argsort(centroids[right_lobes, 2])
    change_map[right_lobes[sort_right_by_z[0]] + 1.] = 1.  # lowest in z
    change_map[right_lobes[sort_right_by_z[-1]] + 1.] = 2.  # highest in z
    if not exclude_rhf:
        change_map[right_lobes[sort_right_by_z[1]] + 1.] = 5.  # middle in z

    change_label_filter.SetChangeMap(change_map)
    lobes_components_relabel = change_label_filter.Execute(biggest_n_components)

    # compute the surface mesh via marching cubes
    lobes_meshes = compute_surface_mesh_marching_cubes(lobes_components_relabel, lung_mask, num_lobes_target)

    return lobes_components_relabel, lobes_meshes, True


def compute_surface_mesh_marching_cubes(label_img: sitk.Image, mask_image: sitk.Image = None,
                                        max_label: int = None):
    """

    :param label_img: the label image to compute the surface of
    :param mask_image: only voxels where mask==True are being considered.
        Be sure to dilate the mask in order to not cut off som of the surface.
    :param max_label: set it to ignore labels greater than this
    :return: one mesh for each labelled object (excluding background)
    """
    meshes = []
    if max_label is None:
        max_label = np.unique(sitk.GetArrayViewFromImage(label_img))[-1]

    for lb in range(1, max_label + 1):
        print(f'\tComputing surface mesh no. {lb}')
        verts, faces, normals, values = marching_cubes(
            volume=(sitk.GetArrayViewFromImage(label_img) == lb),
            level=0.5, spacing=label_img.GetSpacing()[::-1],
            allow_degenerate=False, mask=sitk.GetArrayViewFromImage(mask_image) if mask_image is not None else None)

        verts = np.flip(verts, axis=-1)  # bring coordinates into xyz format
        mesh = create_o3d_mesh(verts=verts, tris=faces)
        meshes.append(mesh)

    return meshes


if __name__ == '__main__':
    data_path = '/home/kaftan/FissureSegmentation/data/'
    ds = LungData(data_path)

    total_lobes = 0
    successes = 0
    for i in range(len(ds)):
        file = ds.get_filename(i)
        case, _, sequence = file.split(os.sep)[-1].split('_')
        sequence = sequence.split('.')[0]
        # if 'EMPIRE' not in case:
        #     continue
        print(f'\nComputing lobes for {case} {sequence}')
        fissures = ds.get_regularized_fissures(i)
        if fissures is None:
            print('\tNo regularized fissures available ... Skipping.')
            continue
        lobes, lobe_meshes, success = find_lobes(fissures, ds.get_lung_mask(i), exclude_rhf=True)
        total_lobes += 1
        if success:
            sitk.WriteImage(lobes, os.path.join(data_path, f'{case}_lobes_{sequence}.nii.gz'))
            for m, mesh in enumerate(lobe_meshes):
                o3d.io.write_triangle_mesh(os.path.join(data_path, f'{case}_mesh_{sequence}', f'{case}_lobe{m + 1}_{sequence}.obj'), mesh)

            visualize_o3d_mesh(lobe_meshes, title=f'{case} {sequence} lobe meshes')
            successes += 1

    print(f'\nResult: {successes} out of {total_lobes} succeeded.')

    # test_case, test_seq = 'EMPIRE01', 'fixed'
    # ind = ds.get_index(test_case, test_seq)
    # lob = ds.get_lobes(ind)
    # mas = ds.get_lung_mask(ind)
    # lobes_to_fissures(lob, mas)
