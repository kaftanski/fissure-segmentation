import glob
import os.path
import time

from torch import nn

from image_ops import resample_equal_spacing
import foerstner
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from data import LungData
from utils import pairwise_dist


class PointDataset(Dataset):
    def __init__(self, sample_points, folder='/home/kaftan/FissureSegmentation/point_data/'):
        files = sorted(glob.glob(os.path.join(folder, '*_points_*')))
        self.sample_points = sample_points
        self.points = []
        self.labels = []
        for file in files:
            case, _, sequence = file.split('/')[-1].split('_')
            sequence = sequence.split('.')[0]
            pts, lbls = load_points(folder, case, sequence)
            self.points.append(pts)
            self.labels.append(lbls)

        self.num_classes = max(len(torch.unique(lbl)) for lbl in self.labels)

    def __getitem__(self, item):
        # randomly sample points
        pts = self.points[item]
        lbls = self.labels[item]
        sample = torch.randperm(pts.shape[1])[:self.sample_points]
        return pts[:, sample], lbls[sample]

    def __len__(self):
        return len(self.points)

    def get_label_frequency(self):
        frequency = torch.zeros(self.num_classes)
        for lbl in self.labels:
            for c in range(self.num_classes):
                frequency[c] += torch.sum(lbl == c)
        frequency /= frequency.sum()
        print(f'Label frequency in point data set: {frequency.tolist()}')
        return frequency


def filter_1d(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6, )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(5, )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
                    weight.view(view)).view(B, C, D, H, W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter_1d(img, weight, 0)
    img = filter_1d(img, weight, 1)
    img = filter_1d(img, weight, 2)

    return img


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


def foerstner_keypoints(img: torch.Tensor, roi: torch.Tensor, sigma: float = 1.5, distinctiveness_threshold: float = 1e-8, show=False):
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


def preprocess_point_features(data_path, point_data_dir, use_coords=True, use_mind=True):
    assert use_coords or use_mind, 'At least one kind of feature should be computed (MIND and/or coords).'

    device = 'cuda:2'

    ds = LungData(data_path)

    # hyperparameters keypoints
    kp_sigma = 0.5
    threshold = 1e-8
    nms_kernel = 7

    # hyperparameters for MIND
    mind_sigma = 0.8
    delta = 1

    # prepare directory
    folder = 'feat_'
    if use_coords:
        folder += 'coords_'
    if use_mind:
        folder += 'mind_'
    folder = folder[:-1]
    out_dir = os.path.join(point_data_dir, folder)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(ds)):
        torch.cuda.empty_cache()

        case, _, sequence = ds.get_filename(i).split('/')[-1].split('_')
        sequence = sequence.replace('.nii.gz', '')
        # if case != 'COPD10':#'COPD' not in case and 'EMPIRE' not in case:
        #     print(f'skipping {case}, {sequence}')
        #     continue
        print(f'Computing points for case {case}, {sequence}...')

        img, fissures = ds[i]
        if fissures is None:
            print('\tNo fissure segmentation found.')
            continue

        mask = ds.get_lung_mask(i)

        # resample all images to unit spacing
        img = resample_equal_spacing(img, target_spacing=1)
        mask = resample_equal_spacing(mask, target_spacing=1)
        fissures = resample_equal_spacing(fissures, target_spacing=1, use_nearest_neighbor=True)

        # dilate fissures so that more keypoints get assigned foreground labels
        fissures_dilate = fissures
        for i in range(1, ds.num_classes):
            fissures_dilate = sitk.DilateObjectMorphology(sitk.Cast(fissures_dilate, sitk.sitkUInt8), kernelRadius=(2, 2, 2), objectValue=i)

        # compute förstner keypoints
        start = time.time()
        img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0).float().to(device)
        kp = foerstner.foerstner_kpts(img_tensor,
                                      mask=torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool)).unsqueeze(0).unsqueeze(0).to(device),
                                      sigma=kp_sigma, thresh=threshold, d=nms_kernel)
        print(f'\tFound {kp.shape[0]} keypoints (took {time.time() - start:.4f})')

        # get label for each point
        fissures_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures_dilate).astype(int)).to(device)
        labels = fissures_tensor[kp[:, 0], kp[:, 1], kp[:, 2]]
        print(f'\tkeypoints per label: {labels.unique(return_counts=True)[1].tolist()}')

        # assemble features for each point
        point_feat = []

        # coordinate features
        if use_coords:
            # transform indices into physical points
            spacing = torch.tensor(img.GetSpacing()[::-1]).unsqueeze(0).to(device)
            point_feat.append(foerstner.kpts_pt(kp * spacing, torch.tensor(img_tensor.shape[2:], device=device) * spacing.squeeze(),
                                                align_corners=True).transpose(0, 1))

        # image patch features
        if use_mind:
            torch.cuda.empty_cache()
            print('\tComputing MIND-SSC features')
            # compute mind features for image
            mind = mind_ssc(img_tensor, sigma=mind_sigma, delta=delta)

            # extract features for keypoints
            point_feat.append(mind[..., kp[:, 0], kp[:, 1], kp[:, 2]].squeeze())

        # save point features
        save_points(torch.cat(point_feat, dim=0), labels, out_dir, case, sequence)


def mind_features(img: torch.Tensor):
    """ https://pubmed.ncbi.nlm.nih.gov/21995071/

    :param img:
    :return:
    """
    device = img.device
    dtype = img.dtype


def mind_ssc(img: torch.Tensor, delta: int = 1, sigma: float = 0.8):
    """ Modality independent neighborhood descriptors (MIND) with self-similarity context (SSC)
        From: http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf, implementation by Lasse Hansen
        Using 6-neighborhood

    :param img: image (-batch) to compute features for. Shape (B x 1 x D x H x W)
    :param delta: neighborhood kernel dilation
    :param sigma: noise estimate, used for gaussian filter
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
    rpad = nn.ReplicationPad3d(delta)

    # compute patch-ssd
    mind = smooth(((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2),
                  sigma)

    # MIND equation
    mind = mind - torch.min(mind, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)

    # permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

    return mind


def save_points(points: torch.Tensor, labels: torch.Tensor, path: str, case: str, sequence: str = 'fixed'):
    torch.save(points.cpu(), os.path.join(path, f'{case}_points_{sequence}.pth'))
    torch.save(labels.cpu(), os.path.join(path, f'{case}_labels_{sequence}.pth'))


def load_points(path: str, case: str, sequence: str = 'fixed'):
    return torch.load(os.path.join(path, f'{case}_points_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_labels_{sequence}.pth'), map_location='cpu')


if __name__ == '__main__':
    preprocess_point_features('/home/kaftan/FissureSegmentation/data', '/home/kaftan/FissureSegmentation/point_data')
    # ds = PointDataset('/home/kaftan/FissureSegmentation/point_data')
    # print(ds[0])
