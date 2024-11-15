import SimpleITK as sitk
import torch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from pytorch3d.structures import Meshes
from pytorch3d.transforms import so3_exp_map, Transform3d
from torch.nn import functional as F

from utils.image_ops import sitk_image_to_tensor, get_resample_factors
from visualization import visualize_point_cloud


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
        MirrorTransform(p_per_sample=0.7)  # 70% chance for mirroring, then 50% for each axis -> independently, 35 % per axis
    ])

    # TODO: maybe add some noise

    data_dict = {"data": img, "seg": seg}
    augmented = transforms(**data_dict)
    return augmented["data"], augmented["seg"]


def point_augmentation(point_clouds: torch.Tensor, rotation_amount=0.1, translation_amount=0.1, scale_amount=0.1):
    """ Random rotation (around a random 3d vector) and random translation.

    :param point_clouds: shape (batch x 3 x N), expected to be in pytorch grid coordinates [-1,1]
    :param rotation_amount: rotation angle in range [-rotation_amount*pi, rotation_amount*pi]
    :param translation_amount: translation in range [-translation_amount, translation_amount]
    :param scale_amount: uniform scaling in range [1 - scale_amount, 1 + scale_amount]
    :return: transformed points and transform object
    """
    # random rotations (so3 representation, axis is the direction of the vector, angle its magnitude)
    random_rotation_vectors = torch.rand(len(point_clouds), 3, device=point_clouds.device) * 2 - 1
    unit_vectors = random_rotation_vectors / random_rotation_vectors.norm(dim=1, keepdim=True)
    log_rotation_matrix = unit_vectors * torch.pi * rotation_amount

    # random translations (in grid coords)
    random_translations = (torch.rand(len(point_clouds), 3, device=point_clouds.device) * 2 - 1) * translation_amount

    # random rescaling
    random_rescale = torch.ones(len(point_clouds), 1, device=point_clouds.device) - \
                     torch.rand(len(point_clouds), 1, device=point_clouds.device) * scale_amount

    # compose the transforms
    transforms = compose_transform(log_rotation_matrix, random_translations, random_rescale)
    return transform_points(point_clouds, transforms), transforms


def compose_transform(log_rotation_matrix: torch.Tensor, translation: torch.Tensor, scaling: torch.Tensor):
    t = Transform3d(device=log_rotation_matrix.device) \
        .rotate(so3_exp_map(log_rotation_matrix)) \
        .scale(scaling.expand(-1, 3)) \
        .translate(translation)
    # log_rotation_matrix is an R^3 vector of which the direction is the rotation axis and the magnitude is the
    # angle magnitude of rotation around the axis

    # scaling can be one value (rigid) or 3 different ones (similarity transform)

    return t


def transform_points_with_centering(point_clouds, transforms: Transform3d):
    """ by default, pytorch3d does not rotate around the center of the point cloud but (0,0,0)

    :param point_clouds: shape (batch x 3 x N)
    :param transforms: transform object
    :return: transformed point_clouds
    """
    assert point_clouds.ndim == 3
    translation_to_center = point_clouds.mean(2, keepdim=True)
    return transform_points(point_clouds - translation_to_center, transforms) + translation_to_center


def transform_points(point_clouds, transforms: Transform3d):
    """ rotates around (0,0,0)

    :param point_clouds: shape (batch x 3 x N)
    :param transforms: transform object
    :return: transformed point_clouds
    """
    assert point_clouds.ndim == 3
    point_clouds = point_clouds.transpose(1, 2)
    transformed = transforms.transform_points(point_clouds)
    return transformed.transpose(1, 2)


def transform_meshes(meshes: Meshes, transforms: Transform3d):
    verts_transformed = transforms.transform_points(meshes.verts_padded())
    if meshes.has_verts_normals():
        normals_transformed = transforms.transform_normals(meshes.verts_normals_padded())
    else:
        normals_transformed = None
    meshes_transformed = Meshes(verts=verts_transformed, faces=meshes.faces_padded(), verts_normals=normals_transformed)
    return meshes_transformed


if __name__ == '__main__':
    from data import PointDataset
    #
    # ds = LungData('../data')
    # sizes = []
    # minmax = sitk.MinimumMaximumImageFilter()
    # min = 0
    # max = 0
    # for i in range(len(ds)):
    #     # if ds.fissures[i] is None:
    #     #     continue
    #     img = ds.get_image(i)
    #     sizes.append(spacing_resampling_matrix(img.GetSize(), img.GetSpacing())[1])
    #     # print(ds.get_id(i), sizes[-1])
    #     minmax.Execute(img)
    #     print(ds.get_id(i), minmax.GetMinimum(), minmax.GetMaximum())
    #     if minmax.GetMinimum() < min:
    #         min = minmax.GetMinimum()
    #     if minmax.GetMaximum() > max:
    #         max = minmax.GetMaximum()
    #
    # sizes = torch.tensor(sizes)
    # print(sizes.min(dim=0), sizes.max(dim=0))
    #
    # print('all min, max:', min, max)
    pds = PointDataset(2048, 'enhancement', do_augmentation=True, binary=True, patch_feat=None)
    for i in range(20):
        pts, label = pds[0]
        pts_aug, _ = point_augmentation(pts.unsqueeze(0))
        visualize_point_cloud(pts_aug.squeeze().transpose(0, 1), labels=label)

