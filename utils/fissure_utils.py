import warnings

import SimpleITK as sitk
import numpy as np
import torch

from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor


def binary_lung_mask_to_left_right(lung_mask: sitk.Image, left_label=1, right_label=2):
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    lung_components = connected_components_filter.Execute(lung_mask)
    obj_cnt = connected_components_filter.GetObjectCount()
    if obj_cnt < 2:
        raise ValueError(f'Found only {obj_cnt} connected components in lung mask. '
                         f'Can\'t determine left and right fissures.')
    elif obj_cnt > 2:
        warnings.warn(f'Found {obj_cnt} connected components in lung mask, but expected 2. '
                      f'Assuming the biggest 2 components are the right & left lung.')

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lung_components_sorted = relabel_filter.Execute(lung_components)
    if obj_cnt > 2:
        print(relabel_filter.GetSizeOfObjectsInPhysicalUnits())

    # extract 2 biggest objects (left & right lung)
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({l: 0 for l in range(3, relabel_filter.GetOriginalNumberOfObjects() + 1)})
    right_left_lung = change_label_filter.Execute(lung_components_sorted)

    # figure out which label is right, which is left
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(right_left_lung)
    centroids = np.array([shape_stats.GetCentroid(l) for l in range(1, 3)])
    cur_right_label, cur_left_label = np.argsort(centroids[:, 0]) + 1  # smaller x is right

    # change labels to be 1 for left, 2 for right lung
    change_map = {cur_left_label.item(): left_label, cur_right_label.item(): right_label}
    change_label_filter.SetChangeMap(change_map)
    right_left_lung = change_label_filter.Execute(right_left_lung)

    return right_left_lung


def binary_to_fissure_segmentation(binary_fissure_seg: torch.Tensor, lung_mask: sitk.Image, resample_spacing=None):
    """ Converts a binary lung fissure segmentation into a left/right fissure label map.
    Also discards any points outside the lung mask.

    :param binary_fissure_seg:
    :param lung_mask:
    :param resample_spacing:
    :return: label map tensor with label 1 for left and 2 for right fissure
    """
    left_label = 1
    right_label = 2

    right_left_lung = binary_lung_mask_to_left_right(lung_mask, left_label=left_label, right_label=right_label)

    if resample_spacing is not None:
        right_left_lung = resample_equal_spacing(right_left_lung, target_spacing=resample_spacing,
                                                 use_nearest_neighbor=True)

    # use lung labels to assign right/left fissure label to binary fissure segmentation
    lung_mask_tensor = sitk_image_to_tensor(right_left_lung).to(binary_fissure_seg.device)
    binary_fissure_seg[lung_mask_tensor == 0] = 0  # discard non lung pixels
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == left_label)] = left_label
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == right_label)] = right_label

    return binary_fissure_seg