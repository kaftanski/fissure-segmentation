"""
MIT License

Copyright (c) 2021 autonomousvision

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

https://github.com/autonomousvision/shape_as_points/blob/main/src/utils.py
https://github.com/autonomousvision/shape_as_points/blob/main/src/model.py

"""

import logging
import numbers
import os
import urllib
from collections import OrderedDict

import math
import numpy as np
import open3d as o3d
import torch
import trimesh
import yaml
from igl import adjacency_matrix, connected_components
from plyfile import PlyData
from pytorch3d.ops import marching_cubes
from pytorch3d.renderer import PerspectiveCameras, rasterize_meshes
from pytorch3d.structures import Meshes
from skimage import measure
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

from models.divroc import DiVRoC


class DifferentiableMarchingCubes(torch.autograd.Function):
    # TODO: batched version
    """
    Autograd function for differentiable PSR. Converts grid to mesh using marching cubes.
    Marching cubes is not differentiable, but as written in their paper, the gradients of the mesh can be approximated
    through the surface normals.
    """
    @staticmethod
    def forward(psr_grid):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        # verts, faces, normals = mc_from_psr(psr_grid, pytorchify=True)
        verts, faces = marching_cubes.marching_cubes(psr_grid, isolevel=0,
                                                     return_local_coords=True)  # range [-1,1]

        mesh = Meshes(verts, faces)
        return mesh.verts_padded(), mesh.faces_padded(), mesh.verts_normals_padded()

    @staticmethod
    def setup_context(ctx, inputs, output):
        psr_grid = inputs[0]
        verts, faces, normals = output

        # save tensors
        ctx.save_for_backward(verts, normals)

        # save non-tensor
        ctx.res = psr_grid.shape[2]

    @staticmethod
    def backward(ctx, dL_dVertex, dL_dFace, dL_dNormals):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        vert_pts, normals = ctx.saved_tensors
        res = ctx.res

        # vert_pts = (vert_pts + 1) / 2  # back to range [0, 1] for point_rasterize
        # matrix multiplication between dL/dV and dV/dPSR
        # dV/dPSR = - normals
        grad_vert = torch.matmul(dL_dVertex.permute(1, 0, 2), -normals.permute(1, 2, 0))  # (n_pts, b, 3)
        # grad_grid = point_rasterize(vert_pts, grad_vert.permute(1, 0, 2), [res]*3)  # b x 1 x res x res x res

        # rasterize the gradients
        vert_pts = vert_pts.unsqueeze(-2).unsqueeze(-2)  # (bs, n_pts, 1, 1, 3)
        grad_vert = grad_vert.permute(1, 2, 0).unsqueeze(-1).unsqueeze(-1)  # (bs, 3, n_pts, 1, 1)
        grad_grid = DiVRoC.apply(grad_vert, vert_pts, (vert_pts.shape[0], 1, res, res, res))  # b x 1 x res x res x res
        grad_grid = grad_grid.squeeze(1)  # b x res x res x res

        return grad_grid


##################################################
# Below are functions for DPSR
def fftfreqs(res, dtype=torch.float32, exact=True):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes
    :return:
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1 / r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1 / r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    return omega


def img(x, deg=1):  # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def spec_gaussian_filter(res, sig):
    omega = fftfreqs(res, dtype=torch.float64)  # [dim0, dim1, dim2, d]
    dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
    filter_ = torch.exp(-0.5 * ((sig * 2 * dis / res[0]) ** 2)).unsqueeze(-1).unsqueeze(-1)
    filter_.requires_grad = False

    return filter_


def grid_interp(grid, pts, batched=True):
    """
    :param grid: tensor of shape (batch, *size, in_features)
    :param pts: tensor of shape (batch, num_points, dim) within range (0, 1)
    :return values at query points
    """
    if not batched:
        grid = grid.unsqueeze(0)
        pts = pts.unsqueeze(0)
    dim = pts.shape[-1]
    bs = grid.shape[0]
    size = torch.tensor(grid.shape[1:-1]).to(grid.device).type(pts.dtype)
    cubesize = 1.0 / (size - 1)  # added -1 here as well!

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1)  # (batch, num_points, 2**dim)
    # latent code on neighbor nodes
    if dim == 2:
        lat = grid.clone()[ind_b, ind_n[..., 0], ind_n[..., 1]]  # (batch, num_points, 2**dim, in_features)
    else:
        lat = grid.clone()[
            ind_b, ind_n[..., 0], ind_n[..., 1], ind_n[..., 2]]  # (batch, num_points, 2**dim, in_features)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)
    query_values = torch.sum(lat * weights.unsqueeze(-1), dim=-2)  # (batch, num_points, in_features)
    if not batched:
        query_values = query_values.squeeze(0)

    return query_values


def scatter_to_grid(inds, vals, size):
    """
    Scatter update values into empty tensor of size size.
    :param inds: (#values, dims)
    :param vals: (#values)
    :param size: tuple for size. len(size)=dims
    """
    dims = inds.shape[1]
    assert (inds.shape[0] == vals.shape[0])
    assert (len(size) == dims)
    dev = vals.device
    # result = torch.zeros(*size).view(-1).to(dev).type(vals.dtype)  # flatten
    # # flatten inds
    result = torch.zeros(*size, device=dev).view(-1).type(vals.dtype)  # flatten
    # flatten inds
    fac = [np.prod(size[i + 1:]) for i in range(len(size) - 1)] + [1]
    fac = torch.tensor(fac, device=dev).type(inds.dtype)
    inds_fold = torch.sum(inds * fac, dim=-1)  # [#values,]
    if torch.any(inds_fold < 0) or torch.any(inds_fold >= len(result)):
        logging.warning('Index out of bounds in scatter_to_grid')
    result.scatter_add_(0, inds_fold, vals)
    result = result.view(*size)
    return result


def point_rasterize(pts, vals, size):
    """
    :param pts: point coords, tensor of shape (batch, num_points, dim) within range (0, 1)
    :param vals: point values, tensor of shape (batch, num_points, features) -> normals
    :param size: len(size)=dim tuple for grid size
    :return rasterized values (batch, features, res0, res1, res2)
    """
    # TODO: allow for padded inputs in batches!

    dim = pts.shape[-1]
    assert (pts.shape[:2] == vals.shape[:2])
    assert (pts.shape[2] == dim)
    size_list = list(size)
    size = torch.tensor(size).to(pts.device).float()
    cubesize = 1.0 / (size - 1)  # important change to the open source code: -1!
    bs = pts.shape[0]
    nf = vals.shape[-1]
    npts = pts.shape[1]
    dev = pts.device

    ind0 = torch.floor(pts / cubesize).long()  # (batch, num_points, dim)
    ind1 = torch.fmod(torch.ceil(pts / cubesize), size).long()  # periodic wrap-around
    ind01 = torch.stack((ind0, ind1), dim=0)  # (2, batch, num_points, dim)
    tmp = torch.tensor([0, 1], dtype=torch.long)
    com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
    dim_ = torch.arange(dim).repeat(com_.shape[0], 1)  # (2**dim, dim)
    ind_ = ind01[com_, ..., dim_]  # (2**dim, dim, batch, num_points)
    ind_n = ind_.permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    # ind_b = torch.arange(bs).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0, 1) # (batch, num_points, 2**dim)
    ind_b = torch.arange(bs, device=dev).expand(ind_n.shape[1], ind_n.shape[2], bs).permute(2, 0,
                                                                                            1)  # (batch, num_points, 2**dim)

    # weights of neighboring nodes
    xyz0 = ind0.type(cubesize.dtype) * cubesize  # (batch, num_points, dim)
    xyz1 = (ind0.type(cubesize.dtype) + 1) * cubesize  # (batch, num_points, dim)
    xyz01 = torch.stack((xyz0, xyz1), dim=0)  # (2, batch, num_points, dim)
    pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = xyz01[1 - com_, ..., dim_].permute(2, 3, 0, 1)  # (batch, num_points, 2**dim, dim)
    pos_ = pos_.type(pts.dtype)
    dxyz_ = torch.abs(pts.unsqueeze(-2) - pos_) / cubesize  # (batch, num_points, 2**dim, dim)
    weights = torch.prod(dxyz_, dim=-1, keepdim=False)  # (batch, num_points, 2**dim)

    ind_b = ind_b.unsqueeze(-1).unsqueeze(-1)  # (batch, num_points, 2**dim, 1, 1)
    ind_n = ind_n.unsqueeze(-2)  # (batch, num_points, 2**dim, 1, dim)
    ind_f = torch.arange(nf, device=dev).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)
    # ind_f = torch.arange(nf).view(1, 1, 1, nf, 1)  # (1, 1, 1, nf, 1)

    ind_b = ind_b.expand(bs, npts, 2 ** dim, nf, 1)
    ind_n = ind_n.expand(bs, npts, 2 ** dim, nf, dim).to(dev)
    ind_f = ind_f.expand(bs, npts, 2 ** dim, nf, 1)
    inds = torch.cat([ind_b, ind_f, ind_n], dim=-1)  # (batch, num_points, 2**dim, nf, 1+1+dim)

    # weighted values
    vals = weights.unsqueeze(-1) * vals.unsqueeze(-2)  # (batch, num_points, 2**dim, nf)

    inds = inds.view(-1, dim + 2).permute(1, 0).long()  # (1+dim+1, bs*npts*2**dim*nf)
    vals = vals.reshape(-1)  # (bs*npts*2**dim*nf)
    tensor_size = [bs, nf] + size_list
    raster = scatter_to_grid(inds.permute(1, 0), vals, tensor_size)

    return raster  # [batch, nf, res, res, res]
