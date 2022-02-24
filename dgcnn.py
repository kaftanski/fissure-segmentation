import time

import torch
from torch import nn
from torch.nn import init
from utils import pairwise_dist


def create_neighbor_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """ Fast implementation of dynamic graph feature computation, needs a lot more VRAM though.

    :param x: features per point, shape: (point cloud batch x features x points)
    :param k: k nearest neighbors that are considered as edges for the graph
    :return: edge features per point, shape: (point cloud batch x features x points x k)
    """
    # k nearest neighbors in feature space
    knn_indices = knn(x, k+1)[..., 1:]  # exclude the point itself from nearest neighbors TODO: why do they include self-loop in their paper???
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


def knn(x, k):
    dist = pairwise_dist(x.transpose(2, 1))
    idx = dist.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    return idx


class DGCNNSeg(nn.Module):
    def __init__(self, k, in_features, num_classes):
        super(DGCNNSeg, self).__init__()
        self.k = k
        self.num_classes = num_classes

        self.spatial_transformer = SpatialTransformer(in_features, k)

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
        self.spatial_transformer.init_weights()

    def forward(self, x):
        """

        :param x: input of shape (point cloud batch x features x points)
        :return: point segmentation of shape (point cloud batch x num_classes)
        """
        # transform point cloud into canonical space
        x = self.spatial_transformer(x)

        # edge convolutions
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
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

    def forward(self, x):
        """

        :param x: input of shape (point cloud batch x features x points)
        :return: new point features (point cloud batch x features x points)
        """
        # create edge features of shape (point cloud batch x features x points x k)
        x = create_neighbor_features(x, self.k)

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
    def __init__(self, in_features, k):
        super(SpatialTransformer, self).__init__()
        self.in_features = in_features

        self.ec = EdgeConv(in_features, [64, 128], k)
        self.shared_fc = SharedFullyConnected(128, 1024, dim=1)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.transform = nn.Linear(256, in_features * in_features)

    def forward(self, x):
        transform_mat = self.ec(x)
        transform_mat = self.shared_fc(transform_mat)
        transform_mat = torch.max(transform_mat, dim=-1, keepdim=False)[0]
        transform_mat = self.mlp(transform_mat)
        transform_mat = self.transform(transform_mat)
        transform_mat = transform_mat.view(x.shape[0], self.in_features, self.in_features)
        return torch.bmm(x.transpose(2, 1), transform_mat).transpose(2, 1)

    def init_weights(self):
        self.apply(init_weights)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))


def init_weights(m):
    if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    # test network
    # test_points = torch.arange(32).view(1, 1, 32).repeat(2, 3, 1).float()
    test_points = torch.randn(32, 3, 1024).to('cuda:2')

    start = time.time()
    own = create_neighbor_features_fast(test_points, k=20)
    print(time.time() - start)

    dgcnn = DGCNNSeg(k=20, in_features=3, num_classes=5).to('cuda:2')
    result = dgcnn(test_points)
