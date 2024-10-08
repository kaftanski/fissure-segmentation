import torch
from torch import nn

from models.point_seg_net import PointSegmentationModelBase
from utils.model_utils import init_weights


class MLPBlock(nn.Module):
    def __init__(self, in_channel, num_neurons_list):
        super(MLPBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(in_channel, num_neurons_list[0], 1, bias=False),
            nn.BatchNorm1d(num_neurons_list[0]),
            nn.LeakyReLU()
        ])

        for i, num_neurons in enumerate(num_neurons_list[1:]):
            self.layers.extend([
                nn.Conv1d(num_neurons_list[i], num_neurons, 1, bias=False),
                nn.BatchNorm1d(num_neurons),
                nn.LeakyReLU()
            ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# implemented after https://arxiv.org/pdf/1612.00593.pdf page 10 section C
# and https://github.com/charlesq34/pointnet/blob/539db60eb63335ae00fe0da0c8e38c791c764d2b/models/transform_nets.py#L10
class TNet(nn.Module):
    def __init__(self, matrix_size=3):
        super(TNet, self).__init__()
        self.matrix_size = matrix_size
        self.layers = nn.Sequential(
            MLPBlock(3, [64, 128, 1024]),
            nn.AdaptiveMaxPool1d(1),
            MLPBlock(1024, [512, 256])
        )
        self.last_layer = nn.Conv1d(32, matrix_size**2, 1, bias=True)

    def forward(self, x):
        return torch.bmm(self.last_layer(self.layers(x)).view(x.shape[0], self.matrix_size, self.matrix_size), x)

    def init_weights(self):
        self.layers.apply(init_weights)
        nn.init.zeros_(self.last_layer.weight.data)
        self.last_layer.bias.data = torch.eye(3, 3, requires_grad=True).flatten()


class PointNetSeg(PointSegmentationModelBase):
    def __init__(self, in_features, num_classes, spatial_transform=False, feature_transform=False, **kwargs):
        super(PointNetSeg, self).__init__(in_features, num_classes, spatial_transform=spatial_transform,
                                          feature_transform=feature_transform)

        # input transformation net
        self.t_net_coord = TNet(matrix_size=3) if spatial_transform else None

        # local branch
        self.local_features = MLPBlock(in_features, [64, 64])

        # feature transformation net
        self.t_net_feat = TNet(matrix_size=64) if feature_transform else None

        # global feature
        self.global_features = nn.Sequential(
            MLPBlock(64, [64, 128, 1024]),
            nn.AdaptiveMaxPool1d(1)
        )

        # segmentation branch
        self.seg_branch = nn.Sequential(
            MLPBlock(64+1024, [256, 128, 64, 64]),
            nn.Conv1d(64, num_classes, 1, bias=True)
        )

        # init weights (special initialisation for TNet)
        self.init_weights()

    def forward(self, x):
        if self.t_net_coord is not None:
            x[:, :3] = self.t_net_coord(x[:, :3])

        x_local = self.local_features(x)
        if self.t_net_feat is not None:
            x_local = self.tnet_feat(x_local)

        x_global = self.global_features(x_local)
        seg_pred = self.seg_branch(torch.cat([x_local, x_global.expand(*x_global.shape[:-1], x_local.shape[-1])], dim=1))
        return seg_pred

    def init_weights(self):
        self.apply(init_weights)
        if self.t_net_coord is not None:
            self.t_net_coord.init_weights()
