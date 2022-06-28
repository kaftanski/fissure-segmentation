import os.path
import time

from torch import nn

from data_processing.keypoint_extraction import get_foerstner_keypoints, get_noisy_keypoints, get_cnn_keypoints
from utils.image_ops import resample_equal_spacing, multiple_objects_morphology
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from data import LungData
from utils.utils import pairwise_dist, filter_1d, smooth, kpts_to_grid

POINT_DIR = '/home/kaftan/FissureSegmentation/point_data'


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


def compute_point_features(img, fissures, lobes, mask, out_dir, case, sequence, kp_mode='foerstner', use_mind=True):
    print(f'Computing keypoints and point features for case {case}, {sequence}...')
    device = 'cuda:1'
    torch.cuda.empty_cache()

    out_dir = os.path.join(out_dir, kp_mode)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # resample all images to unit spacing
    img = resample_equal_spacing(img, target_spacing=1)
    mask = resample_equal_spacing(mask, target_spacing=1, use_nearest_neighbor=True)
    fissures = resample_equal_spacing(fissures, target_spacing=1, use_nearest_neighbor=True)
    lobes = resample_equal_spacing(lobes, target_spacing=1, use_nearest_neighbor=True)

    img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0).float().to(device)

    # dilate fissures so that more keypoints get assigned foreground labels
    fissures_dilated = multiple_objects_morphology(fissures, radius=2, mode='dilate')
    fissures_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures_dilated).astype(int))

    # dilate lobes to fill gaps from the fissures  # TODO: use lobe filling?
    lobes_dilated = multiple_objects_morphology(lobes, radius=2, mode='dilate')

    if kp_mode == 'foerstner':
        kp = get_foerstner_keypoints(device, img_tensor, mask, sigma=0.5, threshold=1e-8, nms_kernel=7)

    elif kp_mode == 'noisy':
        kp = get_noisy_keypoints(fissures_tensor, device)

    elif kp_mode == 'cnn':
        kp = get_cnn_keypoints(cv_dir='results/recall_loss', case=case, sequence=sequence, device=device)

    else:
        raise ValueError(f'No keypoint-mode named "{kp_mode}".')

    # get label for each point
    kp_cpu = kp.cpu()
    labels = fissures_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    torch.save(labels.cpu(), os.path.join(out_dir, f'{case}_fissures_{sequence}.pth'))
    print(f'\tkeypoints per fissure: {labels.unique(return_counts=True)[1].tolist()}')

    lobes_tensor = torch.from_numpy(sitk.GetArrayFromImage(lobes_dilated).astype(int))
    lobes = lobes_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    torch.save(lobes.cpu(), os.path.join(out_dir, f'{case}_lobes_{sequence}.pth'))
    print(f'\tkeypoints per lobe: {lobes.unique(return_counts=True)[1].tolist()}')

    # coordinate features: transform indices into physical points
    spacing = torch.tensor(img.GetSpacing()[::-1]).unsqueeze(0).to(device)
    points = kpts_to_grid((kp * spacing).flip(-1), torch.tensor(img_tensor.shape[2:], device=device) * spacing.squeeze(),
                          align_corners=True).transpose(0, 1)
    torch.save(points.cpu(), os.path.join(out_dir, f'{case}_coords_{sequence}.pth'))

    # image patch features
    if use_mind:
        # hyperparameters
        mind_sigma = 0.8
        delta = 1
        ssc = False

        torch.cuda.empty_cache()
        print('\tComputing MIND features')
        # compute mind features for image
        mind_features = mind(img_tensor, sigma=mind_sigma, dilation=delta, ssc=ssc)

        # extract features for keypoints
        mind_features = mind_features[..., kp[:, 0], kp[:, 1], kp[:, 2]].squeeze()
        torch.save(mind_features.cpu(), os.path.join(out_dir,
                                                     f'{case}_mind{"_ssc" if ssc else ""}_{sequence}.pth'))

    # # VISUALIZATION
    # for i in range(-5, 5):
    #     chosen_slice = img_tensor.squeeze().shape[1] // 2 + i
    #     plt.imshow(img_tensor.squeeze()[:, chosen_slice].cpu(), 'gray')
    #     keypoints_slice = kp_cpu[kp_cpu[:, 1] == chosen_slice]
    #     plt.plot(keypoints_slice[:, 2], keypoints_slice[:, 0], '+')
    #     plt.gca().invert_yaxis()
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'results/EMPIRE02_fixed_keypoints_{i+5}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    #     plt.show()


if __name__ == '__main__':
    data_dir = '/home/kaftan/FissureSegmentation/data'
    ds = LungData(data_dir)

    for i in range(len(ds)):
        case, _, sequence = ds.get_filename(i).split('/')[-1].split('_')
        sequence = sequence.replace('.nii.gz', '')

        print(f'Computing points for case {case}, {sequence}...')
        if ds.fissures[i] is None:
            print('\tNo fissure segmentation found.')
            continue

        img = ds.get_image(i)
        fissures = ds.get_regularized_fissures(i)
        lobes = ds.get_lobes(i)
        mask = ds.get_lung_mask(i)

        compute_point_features(img, fissures, lobes, mask, POINT_DIR, case, sequence, kp_mode='cnn')
