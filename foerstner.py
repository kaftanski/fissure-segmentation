import torch
import torch.nn.functional as F


def filter1D(img, weight, dim, padding_mode='replicate'):
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

    sigma = torch.tensor([sigma], device=device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N, device=device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img


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


def kpts_pt(kpts_world, shape, align_corners=None):
    device = kpts_world.device
    D, H, W = shape

    kpts_pt_ = (kpts_world.flip(-1) / (torch.tensor([W, H, D], device=device) - 1)) * 2 - 1
    if not align_corners:
        kpts_pt_ *= (torch.tensor([W, H, D], device=device) - 1) / torch.tensor([W, H, D], device=device)

    return kpts_pt_


def distinctiveness(img, sigma):
    device = img.device

    filt = torch.tensor([1.0 / 12.0, -8.0 / 12.0, 0.0, 8.0 / 12.0, -1.0 / 12.0], device=device)
    grad = torch.cat([filter1D(img, filt, 0),
                      filter1D(img, filt, 1),
                      filter1D(img, filt, 2)], dim=1)

    struct_inv = invert_structure_tensor(structure_tensor(grad, sigma))

    D = 1. / (struct_inv[:, 0, ...] + struct_inv[:, 3, ...] + struct_inv[:, 5, ...]).unsqueeze(1)
    return D


def foerstner_kpts(img, mask, sigma=1.4, d=9, thresh=1e-8):
    _, _, D, H, W = img.shape
    device = img.device

    dist = distinctiveness(img, sigma)

    kernel_size = tuple(int(0.025 * d) for d in img.shape[2:])
    pad1 = tuple(k // 2 for k in kernel_size)
    pad2 = tuple(k - p - 1 for k, p in zip(kernel_size, pad1))
    pad = (pad2[2], pad1[2], pad2[1], pad1[1], pad2[0], pad1[0])
    # print(f'NMS with kernel size {kernel_size} and padding {pad}')

    maxfeat = F.max_pool3d(F.pad(dist, pad), kernel_size, stride=1)

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

    kpts = torch.nonzero(mask_eroded & (maxfeat == dist) & (dist >= thresh))[:, 2:]

    return kpts
