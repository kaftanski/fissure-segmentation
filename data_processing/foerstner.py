import torch
import torch.nn.functional as F

from utils.image_utils import filter_1d, smooth, nms


def structure_tensor(img, sigma):
    B, C, D, H, W = img.shape

    struct = []
    for i in range(C):
        for j in range(i, C):
            struct.append(smooth((img[:, i, ...] * img[:, j, ...]).unsqueeze(1), sigma))

    return torch.cat(struct, dim=1)


def invert_structure_tensor(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A = e * i - f * f
    B = - b * i + c * f
    C = b * f - c * e
    E = a * i - c * c
    F = - a * f + b * c
    I = a * e - b * b

    det = (a * A + b * B + c * C).unsqueeze(1)

    struct_inv = (1. / det) * torch.stack([A, B, C, E, F, I], dim=1)

    return struct_inv


def invert_structure_tensor_only_trace(struct):
    a = struct[:, 0, ...]
    b = struct[:, 1, ...]
    c = struct[:, 2, ...]
    e = struct[:, 3, ...]
    f = struct[:, 4, ...]
    i = struct[:, 5, ...]

    A = e * i - f * f
    B = - b * i + c * f
    C = b * f - c * e
    E = a * i - c * c
    # F = - a * f + b * c
    I = a * e - b * b

    det = (a * A + b * B + c * C).unsqueeze(1)

    struct_inv = (1. / det) * torch.stack([A, E, I], dim=1)

    return struct_inv


def distinctiveness(img, sigma):
    device = img.device

    filt = torch.tensor([1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0], device=device)
    grad = torch.cat([filter_1d(img, filt, 0),
                      filter_1d(img, filt, 1),
                      filter_1d(img, filt, 2)], dim=1)

    struct_inv = invert_structure_tensor_only_trace(structure_tensor(grad, sigma))

    D = 1. / struct_inv.sum(dim=1, keepdims=True)
    return D


def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8):
    _, _, D, H, W = img.shape
    device = img.device

    dist = distinctiveness(img, sigma)

    # # dynamic kernel size computation for NMS (depending on image dimensions)
    # kernel_size = tuple(int(0.025 * d) for d in img.shape[2:])
    # pad1 = tuple(k // 2 for k in kernel_size)
    # pad2 = tuple(k - p - 1 for k, p in zip(kernel_size, pad1))
    # pad = (pad2[2], pad1[2], pad2[1], pad1[1], pad2[0], pad1[0])
    # print(f'NMS with kernel size {kernel_size} and padding {pad}')

    # non-maximum suppression
    maxfeat = nms(dist, d)

    # erode the keypoint mask (-> no kps at the edges)
    structure_element = torch.tensor([[[0., 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]],
                                      [[0, 1, 0],
                                       [1, 0, 1],
                                       [0, 1, 0]],
                                      [[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]]], device=device)

    mask_eroded = (1 - F.conv3d(1 - mask.float(),
                                structure_element.unsqueeze(0).unsqueeze(0), padding=1).clamp_(0, 1)).bool()

    # mask distinctiveness and extract non-suppressed keypoints
    kpts = torch.nonzero(mask_eroded & (maxfeat == dist) & (dist >= thresh))[:, 2:]
    return kpts
