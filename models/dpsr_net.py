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

https://github.com/autonomousvision/shape_as_points/blob/main/src/dpsr.py

"""
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes, join_meshes_as_batch

from models.dpsr_utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
import numpy as np
import torch.fft
from pytorch3d.ops import marching_cubes, estimate_pointcloud_normals

from models.modelio import LoadableModel, store_config_args
from models.access_models import get_point_seg_model_class, get_point_seg_model_class_from_args


class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=torch.float32)
        self.scale = scale
        self.shift = shift
        self.register_buffer("G", G)

    def forward(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert (V.shape == N.shape)  # [b, nv, ndims]

        # convert to range (0, 1)
        V = (V + 1) / 2

        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        ras_s = torch.fft.rfftn(ras_p, dim=(2, 3, 4))
        ras_s = ras_s.permute(*tuple([0] + list(range(2, self.dim + 1)) + [self.dim + 1, 1]))
        N_ = ras_s[..., None] * self.G  # [b, dim0, dim1, dim2/2+1, n_dim, 1]

        omega = fftfreqs(self.res, dtype=torch.float32).unsqueeze(-1)  # [dim0, dim1, dim2/2+1, n_dim, 1]
        omega *= 2 * np.pi  # normalize frequencies
        omega = omega.to(V.device)

        DivN = torch.sum(-img(torch.view_as_real(N_[..., 0])) * omega, dim=-2)

        Lap = -torch.sum(omega ** 2, -2)  # [dim0, dim1, dim2/2+1, 1]
        Phi = DivN / (Lap + 1e-6)  # [b, dim0, dim1, dim2/2+1, 2]
        Phi = Phi.permute(*tuple([list(range(1, self.dim + 2)) + [0]]))  # [dim0, dim1, dim2/2+1, 2, b]
        Phi[tuple([0] * self.dim)] = 0
        Phi = Phi.permute(*tuple([[self.dim + 1] + list(range(self.dim + 1))]))  # [b, dim0, dim1, dim2/2+1, 2]

        phi = torch.fft.irfftn(torch.view_as_complex(Phi), s=self.res, dim=(1, 2, 3))

        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1)  # [b, nv]
            if self.shift:  # offset points to have mean of 0
                offset = torch.mean(fv, dim=-1)  # [b,]
                phi -= offset.view(*tuple([-1] + [1] * self.dim))

            phi = phi.permute(*tuple([list(range(1, self.dim + 1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))

            if self.scale:
                phi = -phi / torch.abs(fv0.view(*tuple([-1] + [1] * self.dim))) * 0.5
        return phi


class DPSRNet(LoadableModel):
    @store_config_args
    def __init__(self, seg_net_class, k, in_features, num_classes, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 dpsr_res=(128, 128, 128), dpsr_sigma=10, dpsr_scale=True, dpsr_shift=True):
        """
        :param seg_net_class: point segmentation network, e.g. DGCNNSeg
        :param dpsr_res: tuple of output field resolution. eg., (128,128,128)
        :param dpsr_sigma: degree of gaussian smoothing
        """
        super(DPSRNet, self).__init__()
        seg_net_class = get_point_seg_model_class(seg_net_class)
        self.res = dpsr_res
        self.seg_net = seg_net_class(k=k, in_features=in_features, num_classes=num_classes,
                                     spatial_transformer=spatial_transformer, dynamic=dynamic,
                                     image_feat_module=image_feat_module)
        self.dpsr = DPSR(dpsr_res, dpsr_sigma, dpsr_scale, dpsr_shift)

    def forward(self, x):
        """

        :param x: input of shape (point cloud batch x features x points)
        :return: point segmentation of shape (point cloud batch x num_classes) and
            reconstructed meshes (batch x num_classes)
        """
        seg_logits = self.seg_net(x)
        coords = x[:, :3]  # only take the coords (always the first 3 feature channels)

        # limit points to grid (points might be outside due to augmentation)
        torch.clamp_(coords, min=-1, max=1)

        meshes = self.generate_meshes(coords, seg_logits)
        return seg_logits, meshes

    def generate_meshes(self, coords, seg_logits):
        seg_argmax = seg_logits.argmax(1)  # note: this leads to sparse gradients!
        meshes = []
        for b in range(coords.shape[0]):
            # TODO: parallelize batch (using padded tensors)
            for label in range(1, self.seg_net.num_classes):
                cur_points = coords[b, :, seg_argmax[b].squeeze() == label].transpose(-1, -2)

                if cur_points.shape[-2] < 3:
                    # no mesh possible
                    mesh = Meshes(verts=[], faces=[])
                    meshes.append(mesh)
                    continue

                grid = self.compute_psr_grid(cur_points.unsqueeze(0))
                verts, faces = marching_cubes.marching_cubes(grid, isolevel=0)
                mesh = Meshes(verts=verts, faces=faces)
                meshes.append(mesh)

        return join_meshes_as_batch(meshes)

    def compute_psr_grid(self, points):
        """

        :param points: point cloud batch of shape (B x N_pts x 3)
        :return: PSR grid of shape (B x self.res)
        """
        normals = estimate_pointcloud_normals(
            points, neighborhood_size=min(30, points.shape[1]-1),  # same as in open3d normal estimation (30)
            disambiguate_directions=True)

        # TODO: need the normals for backward(?)

        return self.dpsr(points, normals)

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs_min=50):
        seg_logits = self.seg_net.predict_full_pointcloud(pc, sample_points, n_runs_min)
        coords = pc[:, :3]  # only take the coords (always the first 3 feature channels)
        meshes = self.generate_meshes(coords, seg_logits)
        return seg_logits, meshes
