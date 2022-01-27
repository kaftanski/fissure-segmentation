""" utility funtions for random walk, provided for FVMBV exercise 5 (WS 20/21)"""

import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pyamg
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import csr_matrix


def crop_image(input_framed, src):
    img_out = torch.from_numpy(np.array(Image.open(BytesIO(base64.b64decode(src[22:]))))).long()
    # find the cropping borders (!bilinear interpolation leads to slightly different values!)
    mask2 = (img_out[:, :, 0] - 120).float().pow(2) + (img_out[:, :, 1] - 102).float().pow(2) + (
                img_out[:, :, 2] - 201).float().pow(2)
    x_crop = F.avg_pool1d(F.avg_pool1d(torch.exp(-0.0005 * mask2).sum(0).view(1, 1, -1), 3, padding=1, stride=1), 3,
                          padding=1, stride=1)
    x_crop = (x_crop.squeeze() == F.max_pool1d(x_crop, 9, stride=1, padding=4).squeeze()).float() * x_crop.squeeze()
    y_crop = F.avg_pool1d(F.avg_pool1d(torch.exp(-0.0005 * mask2).sum(1).view(1, 1, -1), 3, padding=1, stride=1), 3,
                          padding=1, stride=1)
    y_crop = (y_crop.squeeze() == F.max_pool1d(y_crop.view(1, 1, -1), 9, stride=1,
                                               padding=4).squeeze()).float() * y_crop.squeeze()
    x_crop = torch.topk(x_crop, 2, 0)[1]
    y_crop = torch.topk(y_crop, 2, 0)[1]
    cropped_out = img_out[y_crop[1] + 3:y_crop[0] - 3, x_crop[1] + 3:x_crop[0] - 3, :3]
    input = torch.from_numpy(input_framed[5:-5, 5:-5, 1]).float().unsqueeze(0).unsqueeze(1)
    img_gray = F.interpolate(input, size=(cropped_out.shape[:2]), mode='bilinear', align_corners=False).squeeze()
    return cropped_out, img_gray


def extract_scribbles(cropped_out):
    # using crop image, extract coloured scribbles and convert labels
    mask_colour0 = (cropped_out[:, :, 0] != cropped_out[:, :, 1]) | (cropped_out[:, :, 2] != cropped_out[:, :, 1])
    mask_colour = torch.max_pool2d(1 - mask_colour0.float().unsqueeze(0).unsqueeze(0), 5, stride=1,
                                   padding=2).squeeze() == 0
    label = ((cropped_out[:, :, :3].reshape(-1, 3)[mask_colour.view(-1), :]) * torch.Tensor(
        [1, 256, 256 ** 2]).long().view(1, 3)).sum(1)
    unique_colours = torch.unique(label)
    unique_red = (unique_colours // (256 ** 2))
    unique_green = ((unique_colours % (256 ** 2)) // 256)
    unique_blue = (unique_colours % (256))
    unique_rgb = torch.stack((unique_blue, unique_green, unique_red), 1)
    labels = torch.zeros_like(mask_colour).long()
    labels.view(-1)[mask_colour.view(-1)] = (torch.argmin(
        (cropped_out.reshape(-1, 1, 3)[mask_colour.view(-1), :, :].long() - unique_rgb.view(1, -1, 3)).pow(2).sum(2),
        1) + 1)
    return labels, unique_rgb


def display_colours(unique_rgb):
    unique_blue, unique_green, unique_red = torch.split(unique_rgb, 1, 1)
    # show the user defined colours
    img_unique_colour = torch.stack((unique_blue.view(-1, 1).repeat(1, 10).view(-1, 1).repeat(1, 10),
                                     unique_green.view(-1, 1).repeat(1, 10).view(-1, 1).repeat(1, 10),
                                     unique_red.view(-1, 1).repeat(1, 10).view(-1, 1).repeat(1, 10)), 2)
    plt.imshow(img_unique_colour.transpose(1, 0))
    plt.show()


# semi-transparent colour overlay
def overlay_segment(img, seg, unique_rgb):
    img = img.float() / 255
    if (seg.min() == 0):
        seg += 1
    colors = torch.cat((torch.zeros(1, 3), unique_rgb.float()), 0).view(-1, 3) / 255.0
    H, W = seg.squeeze().shape
    device = img.device
    max_label = seg.long().max().item()
    seg_one_hot = F.one_hot(seg.long(), max_label + 1).float()
    seg_color = torch.mm(seg_one_hot.view(-1, max_label + 1), colors[:max_label + 1, :]).view(H, W, 3)
    alpha = torch.clamp(1.0 - 0.5 * (seg > 0).float(), 0, 1.0)
    overlay = (img * alpha).unsqueeze(2) + seg_color * (1.0 - alpha).unsqueeze(2)
    return overlay


# provided function that calls the sparse LSE solver using multi-grid
def sparseMultiGrid(A, b, iterations):  # A sparse torch matrix, b dense vector
    A_ind = A._indices().cpu().data.numpy()
    A_val = A._values().cpu().data.numpy()
    n1, n2 = A.size()
    SC = csr_matrix((A_val, (A_ind[0, :], A_ind[1, :])), shape=(n1, n2))
    ml = pyamg.ruge_stuben_solver(SC, max_levels=6)  # construct the multigrid hierarchy
    # print(ml)                                           # print hierarchy information
    b_ = b.cpu().data.numpy()
    x = b_ * 0
    for i in range(x.shape[1]):
        x[:, i] = ml.solve(b_[:, i], tol=1e-3)
    return torch.from_numpy(x)  # .view(-1,1)


# provided functions that removes/selects rows and/or columns from sparse matrices
def sparse_rows(S, slice):
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(0)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[0, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((inv_ind[mask], S_ind[1, mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (slice.size(0), S.size(1)))
    return S


def sparse_cols(S, slice):
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(1)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[1, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((S_ind[0, mask], inv_ind[mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (S.size(0), slice.size(0)))
    return S
