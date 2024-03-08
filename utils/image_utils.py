import torch
from scipy.ndimage._filters import _gaussian_kernel1d
from torch.nn import functional as F


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

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter_1d(img, weight, 0)
    img = filter_1d(img, weight, 1)
    img = filter_1d(img, weight, 2)

    return img


def nms(data: torch.Tensor, kernel_size: int):
    """

    :param data: 3d image tensor [B, C, D, H, W]
    :param kernel_size: max pooling kernel size
    :return: suppressed image tensor
    """
    # non-maximum suppression
    pad1 = kernel_size // 2
    pad2 = kernel_size - pad1 - 1
    pad = (pad2, pad1, pad2, pad1, pad2, pad1)
    maxfeat = F.max_pool3d(F.pad(data, pad, mode='replicate'), kernel_size, stride=1)
    return maxfeat


def gaussian_kernel_1d(sigma, order=0, truncate=4.0):
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel1d(sigma, order, radius)
    kernel = torch.from_numpy(kernel).float()
    return kernel


def gaussian_differentiation(img, sigma, order, dim, padding_mode='replicate'):
    weight = gaussian_kernel_1d(sigma, order)
    weight = weight.view(-1, 1, 1).to(img.device)
    return filter_1d(img, weight, dim, padding_mode)
