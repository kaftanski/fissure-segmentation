import os.path

import SimpleITK as sitk
import numpy as np

from data import LungData


def find_lobes(fissure_seg: sitk.Image, lung_mask: sitk.Image) -> sitk.Image:
    # post-process fissures
    # make fissure segmentation binary (disregard the 3 different fissures)
    fissure_seg_binary = sitk.BinaryThreshold(fissure_seg, upperThreshold=0.5, insideValue=0, outsideValue=1)

    # create inverted lobe mask by combining fissures and not-lung
    lung_mask = sitk.Cast(lung_mask, sitk.sitkUInt8)
    lung_mask = sitk.BinaryErode(lung_mask, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.Or(sitk.Not(lung_mask), fissure_seg_binary)

    # close some gaps
    not_lobes = sitk.BinaryMorphologicalClosing(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.BinaryDilate(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)

    # find connected components in lobes mask
    lobes_mask = sitk.Not(not_lobes)
    lobes_mask = sitk.BinaryMorphologicalOpening(lobes_mask, kernelRadius=(4, 4, 4), kernelType=sitk.sitkBall)

    connected_component_filter = sitk.ConnectedComponentImageFilter()
    lobes_components = connected_component_filter.Execute(lobes_mask)
    obj_cnt = connected_component_filter.GetObjectCount()
    print(f'\tFound {obj_cnt} connected components ...')
    if obj_cnt < 5:
        print(f'\tThis is not enough, skipping relabelling.')
        return lobes_components

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lobes_components_sorted = relabel_filter.Execute(lobes_components)
    print(f'\tThe 5 largest objects have sizes {relabel_filter.GetSizeOfObjectsInPhysicalUnits()[:5]}')

    # extract the 5 biggest objects (the 5 lobes)
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({l: 0 for l in range(6, relabel_filter.GetOriginalNumberOfObjects() + 1)})
    lobes_components_top5 = change_label_filter.Execute(lobes_components_sorted)

    # relabel lobes (same as in Mattias' dir-lab COPD lobes)
    # right lower lobe: 1
    # right upper lobe: 2
    # left lower lobe: 3
    # left upper lobe: 4
    # right middle lobe: 5 (contained in label 2 if right horizontal fissure is not segmented)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(lobes_components_top5)
    centroids = np.array([shape_stats.GetCentroid(l) for l in shape_stats.GetLabels()])
    sort_by_x = np.argsort(centroids[:, 0])

    right_lobes = sort_by_x[:3]  # smaller x is right
    left_lobes = sort_by_x[3:]  # higher x is left
    change_map = {}

    sort_left_by_z = np.argsort(centroids[left_lobes, 2])
    change_map[left_lobes[sort_left_by_z[0]] + 1.] = 3.  # lower in z
    change_map[left_lobes[sort_left_by_z[1]] + 1.] = 4.  # higher in z

    sort_right_by_z = np.argsort(centroids[right_lobes, 2])
    change_map[right_lobes[sort_right_by_z[0]] + 1.] = 1.  # lowest in z
    change_map[right_lobes[sort_right_by_z[1]] + 1.] = 5.  # middle in z
    change_map[right_lobes[sort_right_by_z[2]] + 1.] = 2.  # highest in z

    change_label_filter.SetChangeMap(change_map)
    lobes_components_top5_relabel = change_label_filter.Execute(lobes_components_top5)

    return lobes_components_top5_relabel


if __name__ == '__main__':
    # data_path = '/home/kaftan/FissureSegmentation/data/'
    # ds = LungData(data_path)
    #
    # for i in range(len(ds)):
    #     file = ds.get_filename(i)
    #     case, _, sequence = file.split('/')[-1].split('_')
    #     sequence = sequence.split('.')[0]
    #     if 'EMPIRE' not in case:
    #         continue
    #     print(f'Computing lobes for {case} {sequence}')
    #     img, fissures = ds[i]
    #     if fissures is None:
    #         print('\tNo fissures available ... Skipping.')
    #         continue
    #     lobes = find_lobes(fissures, ds.get_lung_mask(i))
    #
    #     sitk.WriteImage(lobes, os.path.join(data_path, f'{case}_lobes_{sequence}.nii.gz'))
    mask = sitk.ReadImage('../data/EMPIRE02_mask_fixed.nii.gz')
    fissures = sitk.ReadImage('../data/EMPIRE02_fissures_poisson_fixed.nii.gz')
    lobes = find_lobes(fissures, mask)
    sitk.WriteImage(lobes, os.path.join('results', f'EMPIRE02_lobes_fixed.nii.gz'))
