import time

import torch
from torch import nn
from torch.nn import init

from models.modelio import LoadableModel, store_config_args
from models.utils import init_weights
from utils.utils import pairwise_dist


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
    def __init__(self, k, in_features, num_classes, spatial_transformer=False, dynamic=True):
        super(DGCNNSeg, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.dynamic = dynamic

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


class SharedFullyConnected(nn.Module):
    def __init__(self, in_features, out_features, dim=2):
        super(SharedFullyConnected, self).__init__()

        if dim == 1:
            conv_layer = nn.Conv1d
            norm_layer = nn.BatchNorm1d
        elif dim == 2:
            conv_layer = nn.Conv2d
            norm_layer = nn.BatchNorm2d
        else:
            conv_layer = nn.Conv3d
            norm_layer = nn.BatchNorm3d

        self.layers = nn.Sequential(
            conv_layer(in_features, out_features, kernel_size=1, bias=False),
            norm_layer(out_features),
            nn.LeakyReLU(negative_slope=0.2)  # TODO: test lighter slope
        )

    def forward(self, x):
        return self.layers(x)


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
        x[:, :self.in_features] = coords.transpose(2, 1)
        return x

    def init_weights(self):
        self.apply(init_weights)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(self.in_features, self.in_features))


if __name__ == '__main__':
    # test network
    # test_points = torch.arange(32).view(1, 1, 32).repeat(2, 3, 1).float()
    test_points = torch.randn(32, 3, 1024).to('cuda:2')

    start = time.time()
    own = create_neighbor_features_fast(test_points, k=20)
    print(time.time() - start)

    dgcnn = DGCNNSeg(k=20, in_features=3, num_classes=5).to('cuda:2')
    result = dgcnn(test_points)
