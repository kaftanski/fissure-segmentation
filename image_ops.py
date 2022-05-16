from typing import Sequence, Union

import SimpleITK as sitk
import numpy as np
import torch


def get_resample_factors(current_spacing: Sequence[float], target_spacing: float = 1.):
    return [spacing/target_spacing for spacing in current_spacing]


def resample_equal_spacing(img: sitk.Image, target_spacing: float = 1., use_nearest_neighbor: bool = False):
    """ resmample an image so that all dimensions have equal spacing

    :param img: input image to resample
    :param target_spacing: the desired spacing for all 3 dimensions in the output
    :return: resampled image
    """
    output_size = [round(size * factor)
                   for size, factor in zip(img.GetSize(), get_resample_factors(img.GetSpacing(), target_spacing))]

    return sitk.Resample(img, size=list(output_size), outputSpacing=(target_spacing,)*3,
                         interpolator=sitk.sitkNearestNeighbor if use_nearest_neighbor else sitk.sitkLinear)


def multiple_objects_morphology(labelmap: sitk.Image, radius: Union[int, Sequence[int]], mode: str = 'dilate'):
    if mode == 'dilate':
        morph_op = sitk.DilateObjectMorphology
    elif mode == 'erode':
        morph_op = sitk.ErodeObjectMorphology
    else:
        raise ValueError(f'No morphology operation named "{mode}". Use "dilate" or "erode".')

    if isinstance(radius, int):
        radius = (radius,)*labelmap.GetDimension()

    objects = np.unique(sitk.GetArrayViewFromImage(labelmap))
    objects = objects[objects != 0]
    for i in objects:
        labelmap = morph_op(sitk.Cast(labelmap, sitk.sitkUInt8), kernelRadius=radius, objectValue=i.item())

    return labelmap


def sitk_image_to_tensor(img: sitk.Image):
    tensor_image = torch.tensor(sitk.GetArrayFromImage(img).squeeze())
    return tensor_image
