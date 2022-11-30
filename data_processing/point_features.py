import os
import time
import warnings

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from constants import KP_MODES, POINT_DIR, POINT_DIR_TS, FEATURE_MODES
from data import LungData, normalize_img
from utils.detached_run import run_detached_from_pycharm
from utils.image_ops import sitk_image_to_tensor, resample_equal_spacing
from utils.image_utils import filter_1d, smooth
from utils.general_utils import pairwise_dist, load_points, kpts_to_grid, sample_patches_at_kpts, ALIGN_CORNERS, kpts_to_world, \
    new_dir


def distinctiveness(img, sigma):
    # init 9-stencil filter
    filter_weights = torch.linspace(-1., 1., steps=9).to(img.device)
    filter_weights /= torch.sum(torch.abs(filter_weights))

    # calculate gradients in each dimension
    F_x = filter_1d(img, filter_weights, 0)
    F_y = filter_1d(img, filter_weights, 1)
    F_z = filter_1d(img, filter_weights, 2)
    print('gradients done')
    # calculate structure tensor
    F_xy = F_x * F_y
    F_xz = F_x * F_z
    F_yz = F_y * F_z

    structure = torch.cat([
        torch.cat([F_x ** 2, F_xy, F_xz], dim=0),
        torch.cat([F_xy, F_y ** 2, F_yz], dim=0),
        torch.cat([F_xz, F_yz, F_z ** 2], dim=0)
    ], dim=1)
    print('structure tensor')
    A = smooth(structure, sigma)
    print('smoothed structure tensor')
    # we had a huge trouble with the formula on the sheet, so we used the formula presented during the lecture
    det_A = torch.det(A.permute(2, 3, 4, 0, 1))
    trace_A = torch.sum(torch.diagonal(A, dim1=0, dim2=1), dim=-1)
    D = det_A / (trace_A + 1e-8)
    return D.view(1, 1, *D.shape)  # torch image format (NxCxDxHxW)


def foerstner_keypoints_wrong(img: torch.Tensor, roi: torch.Tensor, sigma: float = 1.5, distinctiveness_threshold: float = 1e-8, show=False):
    print('start')
    start = time.time()

    D = distinctiveness(img, sigma)
    print('distinctiveness done')

    # non-maximum suppression
    kernel_size = tuple(int(0.025*d) for d in img.shape[2:])
    print(f'NMS with kernel size {kernel_size}')
    padding = tuple(k//2 for k in kernel_size)
    suppressed_D, indices = F.max_pool3d(D, kernel_size=kernel_size,
                                         stride=1, padding=padding, return_indices=True)
    print('non maximum suppression done')
    # converting linear indices to bool tensor
    keypoints = torch.zeros_like(D, dtype=torch.bool).view(-1)
    keypoints[indices] = 1
    keypoints = keypoints.view(D.shape)

    # mask result to roi and threshold distinctiveness
    keypoints_masked = torch.logical_and(keypoints, roi.bool())  # ROI
    keypoints_masked = torch.logical_and(keypoints_masked, D >= distinctiveness_threshold)  # threshold
    keypoints_masked = torch.nonzero(keypoints_masked, as_tuple=False)[:, 2:]  # convert tensor to points
    print('keypoints done')
    print('took {:.4f}s to compute keypoints'.format(time.time() - start))

    if show:
        # VISUALIZATION
        chosen_slice = 200
        plt.imshow(torch.log(torch.clamp(D.squeeze()[:, chosen_slice].cpu(), 1e-3)), 'gray')
        keypoints_slice = torch.nonzero(keypoints.squeeze()[:, chosen_slice] * roi[:, chosen_slice], as_tuple=False)
        plt.plot(keypoints_slice[:, 1], keypoints_slice[:, 0], '+')
        plt.show()

    return keypoints_masked


def mind(img: torch.Tensor, dilation: int = 1, sigma: float = 0.8, ssc: bool = True):
    """ Modality independent neighborhood descriptors (MIND) with 6-neighborhood.
        Source: https://pubmed.ncbi.nlm.nih.gov/21995071/
        Optionally using self-similarity context (SSC, http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf)
        From: , implementation by Lasse Hansen.

    :param img: image (-batch) to compute features for. Shape (B x 1 x D x H x W)
    :param dilation: neighborhood kernel dilation
    :param sigma: noise estimate, used for gaussian filter
    :param ssc: compute features from self-similarity context (SSD between pairs in 6-NH with distance sqrt(2))
    :return: image with MIND features. Shape (B x 12 x D x H x W)
    """
    device = img.device
    dtype = img.dtype

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.Tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]]).long()

    if ssc:
        # compute self-similarity edges

        # squared distances
        dist = pairwise_dist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(dtype).to(device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1

    else:
        # normal 6-neighborhood mind (comparison with center voxel)
        mshift1 = torch.ones(6, 1, 3, 3, 3).to(dtype).to(device)
        mshift2 = torch.zeros(6, 3, 3, 3, dtype=dtype)
        mshift2[six_neighbourhood[:, 0], six_neighbourhood[:, 1], six_neighbourhood[:, 2]] = 1
        mshift2 = mshift2.unsqueeze(1).to(device)

    rpad = nn.ReplicationPad3d(dilation)

    # compute patch-ssd
    mind = smooth(((F.conv3d(rpad(img), mshift1, dilation=dilation) - F.conv3d(rpad(img), mshift2, dilation=dilation)) ** 2), sigma)

    # MIND equation
    mind = mind - torch.min(mind, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)

    if ssc:
        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


def compute_point_features(ds: LungData, case, sequence, kp_dir, feature_mode='mind', device='cuda:0'):
    assert feature_mode in FEATURE_MODES
    print(f'\t{feature_mode.upper()} features')

    img_index = ds.get_index(case, sequence)

    # load keypoints
    kp, _, _, _ = load_points(kp_dir, case, sequence, feat=None)
    kp = kp.transpose(0, 1)

    # image patch features
    if feature_mode == 'mind' or feature_mode == 'mind_ssc':
        # hyperparameters
        mind_sigma = 0.8
        delta = 1
        ssc = feature_mode == 'mind_ssc'

        img = ds.get_image(img_index)
        img = resample_equal_spacing(img, target_spacing=1)
        img_tensor = sitk_image_to_tensor(img).float().to(device)
        kp = kp.to(device)

        # compute mind features for image
        features = mind(img_tensor.view(1, 1, *img_tensor.shape), sigma=mind_sigma, dilation=delta, ssc=ssc)

        # extract features for keypoints
        kp_index = kpts_to_world(kp, img_tensor.shape, align_corners=ALIGN_CORNERS).long()
        features = features[..., kp_index[:, 2], kp_index[:, 1], kp_index[:, 0]].squeeze()

    elif feature_mode == 'image' or feature_mode == 'enhancement':
        # hyperparameters
        patch_size = 5

        # load the image to sample from and resample to unit spacing
        feature_img = ds.get_enhanced_fissures(img_index) if feature_mode == 'enhancement' else ds.get_image(img_index)
        feature_img = resample_equal_spacing(feature_img, target_spacing=1)
        feature_tensor = sitk_image_to_tensor(feature_img).float()
        if not kp.min() >= -1. and kp.max() <= 1.:
            warnings.warn('Keypoints are not given in Pytorch grid coordinates. I am assuming they are world coords.')
            kp = kpts_to_grid(
                kp, shape=(torch.tensor(feature_tensor.shape) * torch.tensor(feature_img.GetSpacing()[::-1])).to(device),
                align_corners=ALIGN_CORNERS)

        features = sample_patches_at_kpts(feature_tensor.unsqueeze(0).unsqueeze(0), kp, patch_size)

        # make a feature vector out of the patch and move features to the first dim
        features = features.squeeze().flatten(start_dim=1).transpose(0, 1)

        if feature_mode == 'image':
            # normalize image intensities
            features = normalize_img(features, max_val=0)  # normalize to HU of water (0) being 1

    else:
        raise ValueError(f'No feature mode named {feature_mode}. Use one of {FEATURE_MODES}.')

    torch.save(features.cpu(), os.path.join(kp_dir, f'{case}_{feature_mode}_{sequence}.pth'))


if __name__ == '__main__':
    run_detached_from_pycharm()

    ts = True

    if ts:
        data_dir = '../TotalSegmentator/ThoraxCrop_v2'
        point_dir = POINT_DIR_TS
    else:
        data_dir = '../data'
        point_dir = POINT_DIR

    ds = LungData(data_dir)

    for kp_mode in KP_MODES:
        if kp_mode == 'noisy' or kp_mode == 'cnn':
            continue

        out_dir = new_dir(point_dir, kp_mode)

        for feat_mode in FEATURE_MODES:
            if feat_mode == 'cnn':
                continue

            for i in range(len(ds)):
                case, sequence = ds.get_id(i)
                print(f'Computing point features for case {case}, {sequence}...')
                if ds.fissures[i] is None:
                    print('\tNo fissure segmentation found.')
                    continue

                compute_point_features(ds, case, sequence, out_dir, feature_mode=feat_mode, device='cuda:3')
