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

    if all(i == o for i, o in zip(img.GetSize(), output_size)):
        return img
    else:
        return sitk.Resample(img, size=list(output_size), outputSpacing=(target_spacing,)*3,
                             # set output space the same as input space
                             outputOrigin=img.GetOrigin(), outputDirection=img.GetDirection(),
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
    image_array = sitk.GetArrayFromImage(img)
    if image_array.dtype == np.uint32 or image_array.dtype == np.uint16:
        image_array = image_array.astype(int)
    tensor_image = torch.from_numpy(image_array.squeeze())
    return tensor_image


def tensor_to_sitk_image(tensor: torch.Tensor, meta_src_img: sitk.Image = None):
    """
    :param tensor: tensor to convert to image
    :param meta_src_img: if provided, the image will copy the metadata from this image
    """
    tensor = tensor.detach().cpu().squeeze()
    img = sitk.GetImageFromArray(tensor.numpy())
    if meta_src_img is not None:
        img.CopyInformation(meta_src_img)

    return img


def write_image(img: torch.Tensor, filename: str, meta_src_img: sitk.Image = None,
                undo_resample_spacing: float = None, interpolator=None):
    """
    :param img: image to save
    :param filename: output filename
    :param meta_src_img: if provided, the image will copy the metadata from this image
    :param undo_resample_spacing: isotropic spacing that the img has been resampled to (compared to meta_src_img)
    :param interpolator: sitk interpolation mode for undoing the resampling
    """
    img = tensor_to_sitk_image(img, meta_src_img=None)

    if undo_resample_spacing is not None:
        img.SetSpacing((undo_resample_spacing,) * 3)
        img = sitk.Resample(img, referenceImage=meta_src_img,
                            interpolator=interpolator if interpolator is not None else sitk.sitkLinear)

    if meta_src_img is not None:
        img.CopyInformation(meta_src_img)

    sitk.WriteImage(img, filename)
    return img


def apply_mask(img: sitk.Image, mask: sitk.Image):
    """ set all voxels where the mask != 0 to 0

    :param img:
    :param mask:
    :return:
    """
    img_arr = sitk.GetArrayFromImage(img)
    img_arr[np.logical_not(sitk.GetArrayViewFromImage(mask).astype(bool))] = 0
    new_img = sitk.GetImageFromArray(img_arr)
    new_img.CopyInformation(img)
    return new_img


def load_image_metadata(img_path):
    if not 'nii' in img_path:
        raise NotImplementedError("Only nifty headers supported for now.")

    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(img_path)
    reader.ReadImageInformation()
    size = tuple(eval(reader.GetMetaData(f'dim[{d}]')) for d in range(1, 4))
    spacing = tuple(eval(reader.GetMetaData(f'pixdim[{d}]')) for d in range(1, 4))
    return size, spacing
