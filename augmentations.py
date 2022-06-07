import SimpleITK as sitk
import torch
from torch.nn import functional as F
from utils.image_ops import sitk_image_to_tensor, get_resample_factors
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform


def spacing_resampling_matrix(input_size, input_spacing, target_spacing=1.):
    rescale = get_resample_factors(input_spacing, target_spacing)
    mat = torch.zeros(1, 3, 4)
    mat[:, :3, :3] = torch.diag(torch.tensor(rescale))
    output_size = [int(round(s * f)) for s, f in zip(input_size, rescale)]
    return mat, output_size


def resample_equal_spacing(image: sitk.Image, target_spacing=1.):
    tensor = sitk_image_to_tensor(image).unsqueeze(0).unsqueeze(0)
    resample_mat, output_size = spacing_resampling_matrix(image.GetSize()[::-1], image.GetSpacing()[::-1], target_spacing)
    grid = F.affine_grid(resample_mat, [1, 1, *output_size], align_corners=True).cuda()
    resampled = F.grid_sample(tensor.cuda(), grid, mode='bilinear')
    return resampled


def image_augmentation(img, seg, patch_size=(128, 128, 128)):
    # minimum distance of the extracted patch center to img border, half of patch size
    # -> therefore patches not outside of image bounds
    patch_center_dist_from_border = [int(p//2) for p in patch_size]

    transforms = Compose([
        SpatialTransform(
            patch_size=patch_size,
            patch_center_dist_from_border=patch_center_dist_from_border,
            do_elastic_deform=True, alpha=(0., 1000.),
            do_rotation=True, angle_x=(-0.3, 0.3), angle_y=(-0.3, 0.3), angle_z=(-0.3, 0.3),  # rotation angle in radian
            do_scale=True, scale=(0.8, 1.2),
            random_crop=True),
        MirrorTransform(p_per_sample=0.7)  # 70% chance for mirroring, then 50% for each axis
    ])

    # TODO: maybe add some noise

    data_dict = {"data": img, "seg": seg}
    augmented = transforms(**data_dict)
    return augmented["data"], augmented["seg"]


if __name__ == '__main__':
    from data import LungData
    ds = LungData('../data')
    sizes = []
    minmax = sitk.MinimumMaximumImageFilter()
    min = 0
    max = 0
    for i in range(len(ds)):
        # if ds.fissures[i] is None:
        #     continue
        img = ds.get_image(i)
        sizes.append(spacing_resampling_matrix(img.GetSize(), img.GetSpacing())[1])
        # print(ds.get_id(i), sizes[-1])
        minmax.Execute(img)
        print(ds.get_id(i), minmax.GetMinimum(), minmax.GetMaximum())
        if minmax.GetMinimum() < min:
            min = minmax.GetMinimum()
        if minmax.GetMaximum() > max:
            max = minmax.GetMaximum()

    sizes = torch.tensor(sizes)
    print(sizes.min(dim=0), sizes.max(dim=0))

    print('all min, max:', min, max)