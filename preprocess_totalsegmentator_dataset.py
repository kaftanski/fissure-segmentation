import os.path
import shutil
from glob import glob
from typing import Iterable

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.morphology import ball, binary_opening
from tqdm import tqdm

from data import ImageDataset, IMG_MIN, IMG_MAX
from data_processing.find_lobes import compute_surface_mesh_marching_cubes
from data_processing.surface_fitting import poisson_reconstruction
from utils.detached_run import run_detached_from_pycharm
from utils.tqdm_utils import tqdm_redirect
from utils.utils import remove_all_but_biggest_component, new_dir, save_meshes

ORIG_DS_PATH = '../TotalSegmentator/Totalsegmentator_dataset/'
PROCESSED_DATA_PATH = '../TotalSegmentator/ThoraxCrop_v2/'

# IDs of images where the 5 lobes are present but cut off somewhere (determined manually)
EXCLUDE_LIST = (57, 58, 67, 135, 165, 199, 212, 215, 256, 264, 266, 294, 321, 428, 509, 542, 555, 566, 607, 651, 682,
                705, 743, 762, 806, 864, 965, 1179, 1257, 1261, 1268, 1307, 1367, 1386)


def find_non_zero_ranges(images: np.ndarray, axis: int = None, open_radius: int = 0):
    """
    find the ranges in which all non-zero pixels lie

    :param images: N images in an array of shape [N, ...]
    :param axis: optionally specify a single axis to compute range for
    :param open_radius: if not 0, perform binary opening with ball structuring element of this radius
    """
    if images.shape[0] > 1:
        sum_img = np.sum(images.squeeze(), axis=0)
    else:
        sum_img = images.squeeze()
    nonzero = (sum_img != 0)

    if open_radius > 0:
        nonzero = binary_opening(nonzero.squeeze(), ball(open_radius)).reshape(nonzero.shape)

    lower_bounds = []
    upper_bounds = []
    axes = [axis] if axis is not None else range(sum_img.ndim)
    for d in axes:
        lb = np.min(np.where(nonzero.any(axis=d), np.argmax(nonzero, axis=d), float('inf')))
        lower_bounds.append(lb)

        ub = sum_img.shape[d] - np.min(np.where(nonzero.any(axis=d), np.argmax(np.flip(nonzero, axis=d), axis=d), float('inf')))
        upper_bounds.append(ub)

    ranges = np.stack([np.array(lower_bounds), np.array(upper_bounds)], axis=1).astype(int)
    return ranges.squeeze()


def combine_labels(filenames, label_values: Iterable = None):
    if label_values is None:
        label_values = range(1, len(filenames) + 1)

    combined_label = None
    for seg_fn, lbl in zip(filenames, label_values):
        label_img = sitk.ReadImage(seg_fn)

        # assert the label image is not empty (i.e. the lobe is depicted)
        if sitk.GetArrayViewFromImage(label_img).max() == 0:
            # print(f'{patid} has no nonzero {os.path.split(seg_fn)[1]}. skipping this patient!')
            return None

        if combined_label is None:
            combined_label = sitk.GetArrayFromImage(label_img) * lbl
        else:
            combined_label[sitk.GetArrayViewFromImage(label_img) > 0] = lbl

    return combined_label


def find_fissures(lobes: sitk.Image, device='cuda:2'):
    lobes_tensor = torch.from_numpy(sitk.GetArrayFromImage(lobes).astype(int)).long()
    lobes_one_hot = F.one_hot(lobes_tensor).permute(3, 0, 1, 2).unsqueeze(0)

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
                                     [0, 0, 0]]]).view(1, 1, 3, 3, 3).repeat(n_lobes + 1, 1, 1, 1, 1)

    dilated_lobes_one_hot = F.conv3d(F.pad(lobes_one_hot.half().to(device), pad=(1, 1, 1, 1, 1, 1)),
                                     dilation_kernel.half().to(device), groups=n_lobes + 1)

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
    return fissure_img


def generate_lung_mask(lobes):
    labels = np.unique(sitk.GetArrayViewFromImage(lobes))

    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({lbl.item(): 1 for lbl in labels[1:]})
    lung_mask = change_label_filter.Execute(lobes)
    return lung_mask


def preprocess_ds():
    new_dir(PROCESSED_DATA_PATH)

    lobe_labels = {
        'lung_lower_lobe_right.nii.gz': 1,
        'lung_upper_lobe_right.nii.gz': 2,
        'lung_lower_lobe_left.nii.gz': 3,
        'lung_upper_lobe_left.nii.gz': 4,
        'lung_middle_lobe_right.nii.gz': 5
    }

    # parse meta data
    meta_data = pd.read_csv(os.path.join(ORIG_DS_PATH, 'meta.csv'), delimiter=';')
    meta_data = meta_data.set_index('image_id')
    print(f'Total amount of images: {meta_data.shape[0]}')

    # only select studies including thorax
    meta_data = meta_data[meta_data['study_type'].str.contains('thorax')]
    print(f'Amount of thorax images: {meta_data.shape[0]}')

    for patid in tqdm(meta_data.index):
        pat_folder = os.path.join(ORIG_DS_PATH, patid)
        img_fn = os.path.join(pat_folder, 'ct.nii.gz')

        # # check metadata
        # reader = sitk.ImageFileReader()
        # reader.SetLoadPrivateTags(True)
        # reader.SetFileName(img_fn)
        # reader.ReadImageInformation()
        # print(reader.GetMetaData('pixdim[1]'), reader.GetMetaData('pixdim[2]'), reader.GetMetaData('pixdim[3]'))
        # print(reader.GetMetaData('dim[1]'), reader.GetMetaData('dim[2]'), reader.GetMetaData('dim[3]'))
        img = sitk.ReadImage(img_fn)

        # load the lobe labels
        seg_folder = os.path.join(pat_folder, 'segmentations')
        combined_lobes_label = combine_labels(filenames=[os.path.join(seg_folder, fn) for fn in lobe_labels.keys()],
                                              label_values=lobe_labels.values())
        if combined_lobes_label is None:
            continue

        # # load the ribcage
        # rib_label_fn = glob.glob(os.path.join(seg_folder, 'rib*.nii.gz'))
        # combined_ribcage_label = combine_labels(rib_label_fn)
        # if combined_ribcage_label is None:
        #     continue

        # find out the spatial range of lobe labels in z-dimension
        # some mis-segmentations
        z_crop_range = find_non_zero_ranges(combined_lobes_label[None], axis=0, open_radius=2).tolist()
        z_pad = 15  # pad by 20 voxels (15*1.5mm = 2.25cm)
        z_crop_range[0] = max(z_crop_range[0] - z_pad, 0)
        z_crop_range[1] = min(z_crop_range[1] + z_pad, combined_lobes_label.shape[0])
        # print(f'Patient {patid}, Lobes in z-Range{z_crop_range}')

        # perform cropping
        img_z_crop = sitk.Extract(img, size=(*img.GetSize()[:2], z_crop_range[1] - z_crop_range[0]),
                                  index=(0, 0, z_crop_range[0]))
        lobe_labels_final = sitk.GetImageFromArray(combined_lobes_label[z_crop_range[0]:z_crop_range[1]])
        lobe_labels_final.CopyInformation(img_z_crop)

        # data has direction (-1,0,0,0,-1,0,0,0,1), so we flip the x and y axis
        img_z_crop = sitk.Flip(img_z_crop, flipAxes=(True, True, False))
        lobe_labels_final = sitk.Flip(lobe_labels_final, flipAxes=(True, True, False))
        print(f'Output size: {img_z_crop.GetSize()}')

        # clamp HU range to [-1000, 1500]
        img_z_crop = sitk.Clamp(img_z_crop, lowerBound=IMG_MIN-1, upperBound=IMG_MAX)

        # compute fissure labels from lobes
        fissure_labels = find_fissures(lobe_labels_final)

        # compute lung mask from lobes
        lung_mask = generate_lung_mask(lobe_labels_final)

        # write all results
        sitk.WriteImage(img_z_crop, os.path.join(PROCESSED_DATA_PATH, f'{patid}_img_fixed.nii.gz'))
        sitk.WriteImage(lobe_labels_final, os.path.join(PROCESSED_DATA_PATH, f'{patid}_lobes_fixed.nii.gz'))
        sitk.WriteImage(fissure_labels, os.path.join(PROCESSED_DATA_PATH, f'{patid}_fissures_fixed.nii.gz'))
        sitk.WriteImage(lung_mask, os.path.join(PROCESSED_DATA_PATH, f'{patid}_mask_fixed.nii.gz'))


def create_meshes():
    img_files = sorted(glob(os.path.join(PROCESSED_DATA_PATH, '*_img_*.nii.gz')))

    for img_file in tqdm_redirect(img_files):
        # load preprocessed data
        case, sequence = os.path.split(img_file)[1].replace('_img_', '_').replace('.nii.gz', '').split('_')
        if int(case.replace('s', '')) in EXCLUDE_LIST:
            print(f'Skipping {case} (incomplete lobes)')
            continue

        print(f'Creating meshes for: {case}')
        fissures = sitk.ReadImage(img_file.replace('_img_', '_fissures_'))
        mask = sitk.ReadImage(img_file.replace('_img_', '_mask_'))
        lobes = sitk.ReadImage(img_file.replace('_img_', '_lobes_'))

        # generate fissure surface meshes with poisson
        regularized_fissures, fissure_meshes = poisson_reconstruction(fissures, mask)
        save_meshes(fissure_meshes, PROCESSED_DATA_PATH, case, sequence, obj_name='fissure')
        sitk.WriteImage(regularized_fissures, img_file.replace('_img_', '_fissures_poisson_'))

        # generate lobe surface meshes with marching cubes
        # do not use the mask as it cuts off surface voxels
        lobe_meshes = compute_surface_mesh_marching_cubes(lobes, mask_image=None)
        for m in lobe_meshes:
            remove_all_but_biggest_component(m)  # use only the biggest connected component
        save_meshes(lobe_meshes, PROCESSED_DATA_PATH, case, sequence, 'lobe')


def remove_excluded_ids(exclude_list=EXCLUDE_LIST):
    for id_num in exclude_list:
        case_id = f's{id_num:04d}'
        pat_files = glob(os.path.join(PROCESSED_DATA_PATH, f'{case_id}_*'))
        print(sorted(pat_files))
        for f in pat_files:
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)


class TotalSegmentatorDataset(ImageDataset):
    def __init__(self, do_augmentation=False):
        super(TotalSegmentatorDataset, self).__init__(PROCESSED_DATA_PATH, do_augmentation=do_augmentation)


if __name__ == '__main__':
    run_detached_from_pycharm()
    # preprocess_ds()
    remove_excluded_ids()
    create_meshes()
    # create_split(5, TotalSegmentatorDataset(), filepath=os.path.join(PROCESSED_DATA_PATH, 'splits_final.pkl.gz'))
