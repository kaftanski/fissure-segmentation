"""
Modified from:
https://github.com/antao97/UnsupervisedPointCloudReconstruction/

MIT License

Copyright (c) 2019 An Tao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import itertools

import math
from abc import ABC

import numpy as np
import torch
from pytorch3d.structures import Meshes
from torch import nn

from models.dgcnn import SharedFullyConnected
from models.modelio import LoadableModel, store_config_args


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNNFoldingNet(LoadableModel):
    @store_config_args
    def __init__(self, k, n_embedding, n_input_points=1024, decode_mesh=True, deform=False, static=False):
        super(DGCNNFoldingNet, self).__init__()
        self.encoder = DGCNN_Cls_Encoder(k, n_embedding, static=static)
        self.n_input_points = n_input_points

        # number of output points is the closest square number to the number of input points
        m = torch.sqrt(torch.tensor(n_input_points)).round().int().item() ** 2
        if deform:
            self.decoder = DeformingDecoder(n_embedding, m, decode_mesh)
        else:
            self.decoder = FoldingDecoder(n_embedding, m, decode_mesh)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs=50):
        vert_accumulation = torch.zeros(pc.shape[0], self.decoder.m, 3, device=pc.device)
        for r in range(n_runs):
            perm = torch.randperm(pc.shape[1], device=pc.device)[:sample_points]
            pred_decoded = self(pc[:, perm].transpose(1, 2))
            if self.decoder.decode_mesh:
                pred_decoded = pred_decoded.verts_padded()
            vert_accumulation += pred_decoded

        vert_accumulation /= n_runs

        if self.decoder.decode_mesh:
            return Meshes(verts=vert_accumulation, faces=self.decoder.faces)
        else:
            return vert_accumulation


# TODO: try ball-query encoder
class DGCNN_Cls_Encoder(LoadableModel):
    @store_config_args
    def __init__(self, k, n_embedding, static=False):
        super(DGCNN_Cls_Encoder, self).__init__()
        self.static = static
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(n_embedding)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, n_embedding, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        if self.static:
            knn_graph = knn(x[:, :3], self.k)
        else:
            knn_graph = None

        x = get_graph_feature(x, k=self.k, idx=knn_graph)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, idx=knn_graph)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, idx=knn_graph)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k, idx=knn_graph)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)  # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
        # TODO: maybe another Fc-layer
        return feat  # (batch_size, 1, feat_dims)


class Decoder(LoadableModel, ABC):
    @store_config_args
    def __init__(self, m=1024, decode_mesh=True):
        super(Decoder, self).__init__()
        self.m = m  # closest square number to number of encoder input points
        self.folding_points = None
        self.faces = None
        self.decode_mesh = decode_mesh

    def get_folding_points(self, batch_size):
        if self.folding_points is None or self.folding_points.shape[0] != batch_size:
            # pre- (or re-)compute points to fold
            device = next(self.parameters()).device
            if self.decode_mesh:
                self.folding_points, self.faces = get_plane_mesh(n=self.m, xrange=(-0.3, 0.3), yrange=(-0.3, 0.3))
                self.faces = self.faces.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            else:
                self.folding_points = get_plane()

            try:
                self.folding_points = torch.from_numpy(self.folding_points)
            except TypeError:
                pass

            self.folding_points = self.folding_points.unsqueeze(0).repeat(batch_size, 1, 1)
            self.folding_points = self.folding_points.to(device).float()

        return self.folding_points


class FoldingDecoder(Decoder):
    @store_config_args
    def __init__(self, n_embedding, m=1024, decode_mesh=True):
        super(FoldingDecoder, self).__init__(m, decode_mesh)

        self.folding1 = nn.Sequential(
            nn.Conv1d(n_embedding + 2, n_embedding, 1),
            nn.ReLU(),
            nn.Conv1d(n_embedding, n_embedding, 1),
            nn.ReLU(),
            nn.Conv1d(n_embedding, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(n_embedding + 3, n_embedding, 1),
            nn.ReLU(),
            nn.Conv1d(n_embedding, n_embedding, 1),
            nn.ReLU(),
            nn.Conv1d(n_embedding, 3, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (batch_size, feat_dims, num_points)
        points = self.get_folding_points(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points),
                         dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
        if self.decode_mesh:
            return Meshes(folding_result2.transpose(1, 2), self.faces)
        else:
            return folding_result2


class DeformingDecoder(Decoder):
    @store_config_args
    def __init__(self, n_embedding, m=1024, decode_mesh=True):
        super(DeformingDecoder, self).__init__(m, decode_mesh)

        self.deforming1 = nn.Sequential(
            SharedFullyConnected(n_embedding + 3, n_embedding, dim=1),
            SharedFullyConnected(n_embedding, n_embedding, dim=1),
            SharedFullyConnected(n_embedding, 3, dim=1, last_layer=True)
        )

        self.deforming2 = nn.Sequential(
            SharedFullyConnected(n_embedding + 3, n_embedding, dim=1),
            SharedFullyConnected(n_embedding, n_embedding, dim=1),
            SharedFullyConnected(n_embedding, 3, dim=1, last_layer=True)
        )

    def get_folding_points(self, batch_size):
        points = super(DeformingDecoder, self).get_folding_points(batch_size)
        if points.shape[2] == 2:  # plane mode
            # add the third coordinate to enable 3D computation
            points = torch.cat([points, torch.zeros(*points.shape[:2], 1, device=points.device)], dim=2)
        return points

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.m)  # (batch_size, feat_dims, num_points)
        points = self.get_folding_points(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)

        offsets1 = self.deforming1(torch.cat([x, points], dim=1))
        deformed1 = points + offsets1

        offsets2 = self.deforming2(torch.cat([x, deformed1], dim=1))
        deformed2 = deformed1 + offsets2

        if self.decode_mesh:
            return Meshes(deformed2.transpose(1, 2), self.faces)
        else:
            return deformed2


def get_plane_mesh(n=2025, xrange=(-1, 1), yrange=(-1, 1), device='cpu'):
    steps = int(math.sqrt(n))
    x = torch.linspace(xrange[0], xrange[1], steps=steps, device=device)
    y = torch.linspace(yrange[0], yrange[1], steps=steps, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # create faces
    faces = []
    for j in range(steps - 1):
        for i in range(steps - 1):
            cur = j * steps + i
            faces.append([cur, cur + 1, cur + steps])
            faces.append([cur + 1, cur + steps, cur + 1 + steps])

    return points, torch.tensor(faces, device=device)


def get_plane():
    meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    plane = np.array(list(itertools.product(x, y)))
    return plane
