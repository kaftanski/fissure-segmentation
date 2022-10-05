import SimpleITK as sitk
import torch

from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor


def binary_to_fissure_segmentation(binary_fissure_seg: torch.Tensor, lr_lung_mask: sitk.Image, resample_spacing=None):
    """ Converts a binary lung fissure segmentation into a left/right fissure label map.
    Also discards any points outside the lung mask.

    :param binary_fissure_seg:
    :param lr_lung_mask:
    :param resample_spacing:
    :return: label map tensor with label 1 for left and 2 for right fissure
    """
    left_label = 1
    right_label = 2

    if resample_spacing is not None:
        lr_lung_mask = resample_equal_spacing(lr_lung_mask, target_spacing=resample_spacing,
                                              use_nearest_neighbor=True)

    # use lung labels to assign right/left fissure label to binary fissure segmentation
    lung_mask_tensor = sitk_image_to_tensor(lr_lung_mask).to(binary_fissure_seg.device)
    binary_fissure_seg[lung_mask_tensor == 0] = 0  # discard non lung pixels
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == left_label)] = left_label
    binary_fissure_seg[torch.logical_and(binary_fissure_seg, lung_mask_tensor == right_label)] = right_label

    return binary_fissure_seg