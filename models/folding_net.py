"""
https://github.com/antao97/UnsupervisedPointCloudReconstruction/
"""
from abc import ABC

import torch
from pytorch3d.structures import Meshes
from torch import nn

from models.dgcnn import SharedFullyConnected
from models.dgcnn_opensrc import get_graph_feature
from models.modelio import LoadableModel, store_config_args
from shapes.shape_constructor import get_sphere, get_gaussian, get_plane, get_plane_mesh

SHAPE_TYPES = ['sphere', 'gaussian', 'plane']


class DGCNNFoldingNet(LoadableModel):
    @store_config_args
    def __init__(self, k, n_embedding, shape_type, decode_mesh=True, deform=False):
        super(DGCNNFoldingNet, self).__init__()
        self.encoder = DGCNN_Cls_Encoder(k, n_embedding)
        if deform:
            self.decoder = DeformingDecoder(n_embedding, shape_type, decode_mesh)
        else:
            self.decoder = FoldingDecoder(n_embedding, shape_type, decode_mesh)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DGCNN_Cls_Encoder(LoadableModel):
    @store_config_args
    def __init__(self, k, n_embedding):
        super(DGCNN_Cls_Encoder, self).__init__()
        if k == None:
            self.k = 20
        else:
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
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
        feat = x.unsqueeze(1)  # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)

        return feat  # (batch_size, 1, feat_dims)


class Decoder(LoadableModel, ABC):
    @store_config_args
    def __init__(self, shape_type, decode_mesh=True):
        super(Decoder, self).__init__()
        self.m = 2025  # 45 * 45.
        self.shape_type = shape_type
        self.folding_points = None
        self.faces = None
        self.decode_mesh = decode_mesh

    def get_folding_points(self, batch_size):
        if self.folding_points is None or self.folding_points.shape[0] != batch_size:
            # pre- (or re-)compute points to fold
            device = next(self.parameters()).device
            if self.shape_type == 'plane':
                if self.decode_mesh:
                    self.folding_points, self.faces = get_plane_mesh(n=self.m, xrange=(-0.3, 0.3), yrange=(-0.3, 0.3))
                    self.faces = self.faces.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
                else:
                    self.folding_points = get_plane()
            elif self.shape_type == 'sphere':
                if self.decode_mesh:
                    raise NotImplementedError('No sphere mesh defined yet')
                self.folding_points = get_sphere()
            elif self.shape_type == 'sphere':
                if self.decode_mesh:
                    raise ValueError('No gaussian mesh is possible.')
                self.folding_points = get_gaussian()
            else:
                raise ValueError(f'No shape named "{self.shape_type}". Use one of {SHAPE_TYPES}.')

            try:
                self.folding_points = torch.from_numpy(self.folding_points)
            except TypeError:
                pass

            self.folding_points = self.folding_points.unsqueeze(0).repeat(batch_size, 1, 1)
            self.folding_points = self.folding_points.to(device).float()

        return self.folding_points


class FoldingDecoder(Decoder):
    @store_config_args
    def __init__(self, n_embedding, shape_type, decode_mesh=True):
        super(FoldingDecoder, self).__init__(shape_type, decode_mesh)

        if self.shape_type == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(n_embedding + 2, n_embedding, 1),
                nn.ReLU(),
                nn.Conv1d(n_embedding, n_embedding, 1),
                nn.ReLU(),
                nn.Conv1d(n_embedding, 3, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(n_embedding + 3, n_embedding, 1),
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
    def __init__(self, n_embedding, shape_type, decode_mesh=True):
        super(DeformingDecoder, self).__init__(shape_type, decode_mesh)

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
        deformed2 = points + offsets2

        if self.decode_mesh:
            return Meshes(deformed2.transpose(1, 2), self.faces)
        else:
            return deformed2
