import warnings

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt

from data_processing.datasets import ImageDataset


def check_left_right_lung_plausible(component_sizes, max_volume_ratio=10):
    component_sizes = sorted(component_sizes, reverse=True)
    if len(component_sizes) < 2:
        # needs at least two components
        print('Implausible mask: only one component.')
        return False
    else:
        ratio = component_sizes[0] / component_sizes[1]
        if ratio > max_volume_ratio:
            # the largest component is too much larger than the second largest
            print(f'Implausible volume-ratio of first/second largest component: {ratio}')
            return False
        else:
            if len(component_sizes) > 2:
                print(component_sizes)
                warnings.warn(f'Found {len(component_sizes)} connected components in lung mask, but expected 2. '
                              f'Assuming the biggest 2 components are the right & left lung.')
            return True


def maybe_detach_mask(lung_mask, opening_radius=3, _opened=False):
    # get connected components
    connected_components_filter = sitk.ConnectedComponentImageFilter()
    lung_components = connected_components_filter.Execute(lung_mask)

    # get the sizes of components
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    relabel_filter.Execute(lung_components)
    sizes = relabel_filter.GetSizeOfObjectsInPhysicalUnits()

    # check plausibility
    if check_left_right_lung_plausible(sizes):
        print('Done')
        return lung_components, _opened
    else:
        # open the image to detach left/right
        print(f'Opening with radius {opening_radius} to detach left and right lung.')
        lung_mask_opened = sitk.BinaryMorphologicalOpening(lung_mask, kernelRadius=(opening_radius,) * 3)

        # check and maybe open with bigger radius
        return maybe_detach_mask(lung_mask_opened, opening_radius + 2, _opened=True)


def binary_lung_mask_to_left_right(lung_mask: sitk.Image, left_label=1, right_label=2):
    # maybe detach left/right lung (if there are only one component or the two biggest ones are implausible
    lung_components, opened = maybe_detach_mask(lung_mask)

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lung_components_sorted = relabel_filter.Execute(lung_components)

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

    # restore pixels that were lost during opening and assign them to left or right lung
    if opened:
        right_left_lung_arr = sitk.GetArrayFromImage(right_left_lung)
        all_distances = np.zeros((2, *right_left_lung_arr.shape))
        for i in range(2):
            all_distances[i] = distance_transform_edt(right_left_lung_arr != i+1)

        right_left_lung_arr = np.argmin(all_distances, axis=0) + 1
        right_left_lung_arr[sitk.GetArrayViewFromImage(lung_mask) == 0] = 0
        right_left_lung_restored = sitk.GetImageFromArray(right_left_lung_arr)
        right_left_lung_restored.CopyInformation(right_left_lung)
        right_left_lung = right_left_lung_restored

    return right_left_lung


if __name__ == '__main__':
    ds = ImageDataset('../data')
    for i in range(len(ds)):
        print(ds.get_id(i))
        lung_mask = ds.get_lung_mask(i)
        right_left_mask = binary_lung_mask_to_left_right(lung_mask)

        out_fname = ds.lung_masks[i].replace('_mask_', '_masklr_')
        sitk.WriteImage(right_left_mask, out_fname)
