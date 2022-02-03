import glob
import os.path
import time

import SimpleITK as sitk
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from data import LungData


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

    def __getitem__(self, item):
        # randomly sample points
        pts = self.points[item]
        lbls = self.labels[item]
        sample = torch.randperm(pts.shape[1])[:self.sample_points]
        return pts[:, sample], lbls[sample]

    def __len__(self):
        return len(self.points)


def filter_1d(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6, )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(5, )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H,
                                                                                                            W)


def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter_1d(filter_1d(filter_1d(img, weight, 0), weight, 1), weight, 2)
    return img


def foerstner_keypoints(img: torch.Tensor, roi: torch.Tensor):
    start = time.time()

    # hyperparameter
    sigma = 1.5

    # init 9-stencil filter
    filter_weights = torch.linspace(-1., 1., steps=9).to(img.device)
    filter_weights /= torch.sum(torch.abs(filter_weights))

    # calculate gradients in each dimension
    F_x = filter_1d(img, filter_weights, 0)
    F_y = filter_1d(img, filter_weights, 1)
    F_z = filter_1d(img, filter_weights, 2)

    # calculate structure tensor
    F_xy = F_x * F_y
    F_xz = F_x * F_z
    F_yz = F_y * F_z

    structure = torch.cat([
        torch.cat([F_x ** 2, F_xy, F_xz], dim=0),
        torch.cat([F_xy, F_y ** 2, F_yz], dim=0),
        torch.cat([F_xz, F_yz, F_z ** 2], dim=0)
    ], dim=1)
    A = smooth(structure, sigma)
    # we had a huge trouble with the formula on the sheet, so we used the formula presented during the lecture
    det_A = torch.det(A.permute(2, 3, 4, 0, 1))
    trace_A = torch.sum(torch.diagonal(A, dim1=0, dim2=1), dim=-1)
    D = det_A / (trace_A + 1e-8)

    # non-maximum suppression
    kernel_size = tuple(int(0.025*d) for d in img.shape[2:])
    padding = tuple(k//2 for k in kernel_size)
    suppressed_D, indices = F.max_pool3d(D.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size,
                                         stride=1, padding=padding, return_indices=True)
    # converting linear indices to bool tensor
    keypoints = torch.zeros_like(D).view(-1)
    keypoints[indices] = 1
    # mask result to roi
    keypoints_masked = torch.nonzero(keypoints.view(D.shape) * roi, as_tuple=False)

    print('took {:.4f}s to compute keypoints'.format(time.time() - start))

    # Choose slice to visualize:
    chosen_slice = 200
    plt.imshow(torch.log(torch.clamp(D[:, chosen_slice].cpu(), 1e-3)), 'gray')
    # plot some keypoints
    keypoints_slice = torch.nonzero(keypoints.view(D.shape)[:, chosen_slice] * roi[:, chosen_slice], as_tuple=False)
    plt.plot(keypoints_slice[:, 1], keypoints_slice[:, 0], '+')
    plt.show()

    return keypoints_masked


def preprocess_point_features(data_path, output_path):
    device = 'cpu'  # 'cuda:1'

    ds = LungData(data_path)

    # TODO: reduce number of points (~8k instead of 1-2 M)
    #   transform points into unit sphere
    for i in range(len(ds)):
        case, _, sequence = ds.get_filename(i).split('/')[-1].split('_')
        sequence = sequence.replace('.nii.gz', '')
        print(f'Computing points for case {case}, {sequence}...')

        img, fissures = ds[i]
        if fissures is None:
            continue

        mask = ds.get_lung_mask(i)

        kp = foerstner_keypoints(torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0).float().to(device),
                                 torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool)).to(device))

        fissures_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures).astype(int)).to(device)

        # get label for each point
        labels = fissures_tensor[kp[:, 0], kp[:, 1], kp[:, 2]]

        # transform indices into physical points
        spacing = torch.tensor(img.GetSpacing()[::-1]).unsqueeze(0)
        kp = kp * spacing

        # save points
        save_points(kp.transpose(0, 1), labels, output_path, case, sequence)


def save_points(points: torch.Tensor, labels: torch.Tensor, path: str, case: str, sequence: str = 'fixed'):
    torch.save(points, os.path.join(path, f'{case}_points_{sequence}.pth'))
    torch.save(labels, os.path.join(path, f'{case}_labels_{sequence}.pth'))


def load_points(path: str, case: str, sequence: str = 'fixed'):
    return torch.load(os.path.join(path, f'{case}_points_{sequence}.pth')), \
           torch.load(os.path.join(path, f'{case}_labels_{sequence}.pth'))


if __name__ == '__main__':
    preprocess_point_features('/home/kaftan/FissureSegmentation/data', '/home/kaftan/FissureSegmentation/point_data')
    # ds = PointDataset('/home/kaftan/FissureSegmentation/point_data')
    # print(ds[0])
