import time
import warnings

import torch
from torch import nn
from torch.nn import init

from models.modelio import LoadableModel, store_config_args
from models.utils import init_weights
from utils.utils import pairwise_dist
from torch.nn import functional as F


def create_neighbor_features(x: torch.Tensor, k: int, fixed_knn_graph: torch.Tensor = None) -> torch.Tensor:
    """ Memory efficient implementation of dynamic graph feature computation.

    :param x: features per point, shape: (point cloud batch x features x points)
    :param k: k nearest neighbors that are considered as edges for the graph
    :param fixed_knn_graph: optionally input fixed kNN graph
    :return: edge features per point, shape: (point cloud batch x features x points x k)
    """
    if fixed_knn_graph is None:
        # k nearest neighbors in feature space
        knn_indices = knn(x, k, self_loop=False)  # exclude the point itself from nearest neighbors TODO: why do they include self-loop in their paper???
    else:
        knn_indices = fixed_knn_graph

    knn_indices = knn_indices.reshape(knn_indices.shape[0], -1)
    neighbor_features = torch.take_along_dim(x, indices=knn_indices.unsqueeze(1), dim=-1).view(*x.shape, k)

    # assemble edge features (local and relative features)
    x = x.unsqueeze(-1).repeat(1, 1, 1, k)
    return torch.cat([neighbor_features - x, x], dim=1)


def create_neighbor_features_fast(features: torch.Tensor, k: int) -> torch.Tensor:
    """ Fast implementation of dynamic graph feature computation, needs a lot more VRAM though.

    :param features: features per point, shape: (point cloud batch x features x points)
    :param k: k nearest neighbors that are considered as edges for the graph
    :return: edge features per point, shape: (point cloud batch x features x points x k)
    """
    # pairwise differences of all points
    pairwise_differences = (features.unsqueeze(2) - features.unsqueeze(3))

    # k nearest neighbors on pairwise distances
    _, knn_indices = torch.topk(pairwise_differences.pow(2).sum(1), k+1, dim=2, largest=False)
    knn_indices = knn_indices[..., 1:]  # exclude the point itself from nearest neighbors TODO: why do they include self-loop in their paper???

    # pick relative feature
    edge_features = torch.gather(pairwise_differences, dim=-1,
                                 index=knn_indices.unsqueeze(1).repeat(1, pairwise_differences.shape[1], 1, 1))

    # assemble edge features (local and relative features)
    return torch.cat([features.unsqueeze(-1).repeat(1, 1, 1, k), edge_features], dim=1)


def knn(x, k, self_loop=False):
    # use k+1 and ignore first neighbor to exclude self-loop in graph
    k_modifier = 0 if self_loop else 1

    dist = pairwise_dist(x.transpose(2, 1))
    idx = dist.topk(k=k+k_modifier, dim=-1, largest=False)[1][..., k_modifier:]  # (batch_size, num_points, k)
    return idx


class DGCNNSeg(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, num_classes, spatial_transformer=False, dynamic=True, image_feat_module=False):
        super(DGCNNSeg, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.dynamic = dynamic

        if image_feat_module:
            if in_features < 4:
                raise ValueError('Number of In-Features for DGCNN too low if you want to use the image feature module! '
                                 'Need at 3, as the first 3 are assumed to be the point coordinates.')
            self.image_feature_module = ImageFeatures(in_channels=in_features-3, out_channels=(6, 12))
            in_features = 3 + 12
        else:
            self.image_feature_module = None

        if spatial_transformer:
            self.spatial_transformer = SpatialTransformer(k)
        else:
            self.spatial_transformer = None

        self.ec1 = EdgeConv(in_features, [64, 64], self.k)
        self.ec2 = EdgeConv(64, [64], self.k)
        self.ec3 = EdgeConv(64, [64], self.k)

        self.global_feature = nn.Sequential(
            SharedFullyConnected(3 * 64, 1024, dim=1),
            nn.AdaptiveMaxPool1d(1)
        )

        self.segmentation = nn.Sequential(
            SharedFullyConnected(3 * 64 + 1024, 256, dim=1),
            SharedFullyConnected(256, 256, dim=1),
            # nn.Dropout(p=0.5), TODO: use this?
            SharedFullyConnected(256, 128, dim=1),
            # nn.Dropout(p=0.5),
            SharedFullyConnected(128, num_classes, dim=1)
        )

        self.apply(init_weights)
        if spatial_transformer:
            self.spatial_transformer.init_weights()

    def forward(self, x):
        """

        :param x: input of shape (point cloud batch x features x points)
        :return: point segmentation of shape (point cloud batch x num_classes)
        """
        # compute static kNN graph based on coordinates if net is not supposed to be dynamic
        knn_graph = None if self.dynamic else knn(x[:, :3], self.k, self_loop=False)

        # extract image features for points
        if self.image_feature_module is not None:
            x = self.image_feature_module(x)

        # transform point cloud into canonical space
        if self.spatial_transformer is not None:
            x = self.spatial_transformer(x)

        # edge convolutions
        x1 = self.ec1(x, knn_graph)
        x2 = self.ec2(x1, knn_graph)
        x3 = self.ec3(x2, knn_graph)
        multi_level_features = torch.cat([x1, x2, x3], dim=1)

        # global feature vector
        global_features = self.global_feature(multi_level_features)

        # point segmentation from local and global features
        x = torch.cat([multi_level_features, global_features.repeat(1, 1, multi_level_features.shape[-1])], dim=1)
        x = self.segmentation(x)

        return x

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs_min=50):
        n_leftover_runs = n_runs_min // 5
        n_initial_runs = n_runs_min - n_leftover_runs
        softmax_accumulation = torch.zeros(pc.shape[0], self.num_classes, *pc.shape[2:], device=pc.device)
        for r in range(n_initial_runs):
            perm = torch.randperm(pc.shape[-1], device=pc.device)[:sample_points]
            softmax_accumulation[..., perm] += F.softmax(self(pc[..., perm]), dim=1)

        # look if there are points that have been left out
        left_out_pts = torch.nonzero(softmax_accumulation.sum(1) == 0)[..., 1]
        other_pts = torch.nonzero(softmax_accumulation.sum(1))[..., 1]
        print(f'After {n_initial_runs} runs, {left_out_pts.shape[0]} points have not been seen yet.')
        point_mix = sample_points // 2
        fill_out_num = sample_points - point_mix
        perm = torch.randperm(n_leftover_runs*point_mix, device=pc.device) % len(left_out_pts)
        for r in range(n_leftover_runs):
            lo_pts = left_out_pts[perm[r*point_mix:(r+1)*point_mix]]
            other = torch.randperm(len(other_pts), device=pc.device)[:fill_out_num]
            pts = torch.cat((lo_pts, other), dim=0)
            softmax_accumulation[..., pts] += F.softmax(self(pc[..., pts]), dim=0)

        if (softmax_accumulation.sum(1) == 0).sum() != 0:
            warnings.warn('NOT ALL POINTS HAVE BEEN SEEN')

        return F.softmax(softmax_accumulation, dim=1)


class EdgeConv(nn.Module):
    def __init__(self, in_features, out_features_list, k):
        super(EdgeConv, self).__init__()
        self.k = k

        # multiply in-features because of concatenation of x_i with (x_i - x_j)
        features = [in_features * 2] + list(out_features_list)

        # create MLP
        self.shared_mlp = nn.ModuleList()
        for i in range(len(out_features_list)):
            self.shared_mlp.append(SharedFullyConnected(features[i], features[i + 1]))

    def forward(self, x, fixed_knn_graph=None):
        """

        :param x: input of shape (point cloud batch x features x points)
        :param fixed_knn_graph: specify this if you don't want dynamic graph computation and use the given graph instead
        :return: new point features (point cloud batch x features x points)
        """
        # create edge features of shape (point cloud batch x features x points x k)
        x = create_neighbor_features(x, self.k, fixed_knn_graph)

        # extract features
        for layer in self.shared_mlp:
            x = layer(x)

        # max pool over edges
        x = torch.max(x, dim=-1, keepdim=False)[0]

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, k):
        super(SpatialTransformer, self).__init__()
        self.in_features = 3  # only use coords

        self.ec = EdgeConv(self.in_features, [64, 128], k)
        self.shared_fc = SharedFullyConnected(128, 1024, dim=1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.transform = nn.Linear(256, self.in_features * self.in_features)

    def forward(self, x, fixed_knn_graph=None):
        coords = torch.clone(x[:, :self.in_features])  # convention: coords are always the first 3 channels!
        transform_mat = self.ec(coords, fixed_knn_graph)
        transform_mat = self.shared_fc(transform_mat)
        transform_mat = torch.max(transform_mat, dim=-1, keepdim=False)[0]
        transform_mat = self.mlp(transform_mat)
        transform_mat = self.transform(transform_mat)
        transform_mat = transform_mat.view(x.shape[0], self.in_features, self.in_features)
        coords = coords.transpose(2, 1)
        coords = torch.bmm(coords, transform_mat)  # transform coords
        x = torch.cat([coords.transpose(2, 1), x[:, self.in_features:]], dim=1)
        return x

    def init_weights(self):
        self.apply(init_weights)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(self.in_features, self.in_features))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, bias=False, padding=True, dim=2,
                 negative_slope=1e-2):
        super(ConvBlock, self).__init__()

        if dim == 1:
            conv_layer = nn.Conv1d
            norm_layer = nn.BatchNorm1d
        elif dim == 2:
            conv_layer = nn.Conv2d
            norm_layer = nn.BatchNorm2d
        elif dim == 3:
            conv_layer = nn.Conv3d
            norm_layer = nn.BatchNorm3d
        else:
            raise ValueError(f'There is no Conv layer for dimensionality {dim}.')

        self.layers = nn.Sequential(
            conv_layer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias,
                       stride=stride, padding=(kernel_size//2) if padding else 0),
            norm_layer(num_features=out_channels),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, x):
        return self.layers(x)


class SharedFullyConnected(ConvBlock):
    def __init__(self, in_features, out_features, dim=2):
        super(SharedFullyConnected, self).__init__(
            in_features, out_features, dim=dim,
            kernel_size=1, padding=False, bias=False,
            negative_slope=0.2)  # TODO: test lighter slope

    def forward(self, x):
        return self.layers(x)


class ImageFeatures(nn.Module):
    def __init__(self, in_channels=6, out_channels=(6, 12), kernel_size=1):
        super(ImageFeatures, self).__init__()

        self.layers = nn.ModuleList()
        in_ch = [in_channels, *out_channels[:-1]]
        for i, o in zip(in_ch, out_channels):
            self.layers.append(ConvBlock(in_channels=i, out_channels=o, kernel_size=kernel_size, dim=1))

    def forward(self, x):
        # only use the non-coordinate features (i.e. the channels after the first 3!)
        feat = torch.clone(x[:, 3:])

        for layer in self.layers:
            feat = layer(feat)

        x = torch.concat([x[:, :3], feat], dim=1)
        return x


if __name__ == '__main__':
    # test network
    # test_points = torch.arange(32).view(1, 1, 32).repeat(2, 3, 1).float()
    test_points = torch.randn(32, 3, 1024).to('cuda:2')

    start = time.time()
    own = create_neighbor_features_fast(test_points, k=20)
    print(time.time() - start)

    dgcnn = DGCNNSeg(k=20, in_features=3, num_classes=5).to('cuda:2')
    result = dgcnn(test_points)
