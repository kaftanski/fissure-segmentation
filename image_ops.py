import SimpleITK as sitk


def resample_equal_spacing(img: sitk.Image, target_spacing: float = 1., use_nearest_neighbor: bool = False):
    """ resmample an image so that all dimensions have equal spacing

    :param img: input image to resample
    :param target_spacing: the desired spacing for all 3 dimensions in the output
    :return: resampled image
    """
    output_size = [round(size * (spacing/target_spacing)) for size, spacing in zip(img.GetSize(), img.GetSpacing())]
    return sitk.Resample(img, size=list(output_size), outputSpacing=(1, 1, 1),
                         interpolator=sitk.sitkNearestNeighbor if use_nearest_neighbor else sitk.sitkLinear)
