import os

import torch
from torch.nn import functional as F


def pairwise_dist(x):
    """ distance from each point in x to itself and others

    :param x: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    xTx = torch.bmm(x, x.transpose(2, 1))
    dist = xx - 2.0 * xTx + xx.transpose(2, 1)
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0  # ensure diagonal is 0
    return dist


def pairwise_dist2(x, y):
    """ distance from each point in x to its corresponding point in y

    :param x: point cloud batch of shape (B x N x 3)
    :param y: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    yy = (y ** 2).sum(2, keepdim=True)
    xTy = torch.bmm(x, y.transpose(2, 1))
    dist = xx - 2.0 * xTy + yy.transpose(2, 1)
    return dist


def save_points(points: torch.Tensor, labels: torch.Tensor, path: str, case: str, sequence: str = 'fixed'):
    torch.save(points.cpu(), os.path.join(path, f'{case}_coords_{sequence}.pth'))
    torch.save(labels.cpu(), os.path.join(path, f'{case}_labels_{sequence}.pth'))


def load_points(path: str, case: str, sequence: str = 'fixed', feat: str = None):
    return torch.load(os.path.join(path, f'{case}_coords_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_labels_{sequence}.pth'), map_location='cpu'), \
           torch.load(os.path.join(path, f'{case}_{feat}_{sequence}.pth'), map_location='cpu') if feat is not None \
               else None


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