import glob
import os.path

import numpy as np
import torch
from torch import nn
from models.modelio import store_config_args, LoadableModel


class SSM(LoadableModel):
    @store_config_args
    def __init__(self, alpha=2.5, target_variance=0.95, dimensionality=3):
        super().__init__()
        self.target_variance = target_variance
        self.alpha = alpha
        self.dim = dimensionality

        # register parameters that will be set by calling SSM.fit
        # these can be used like normal python attributes!
        self.register_parameter('num_modes', None)
        self.register_parameter('percent_of_variance', None)

        self.register_parameter('mean_shape', None)
        self.register_parameter('eigenvalues', None)
        self.register_parameter('eigenvectors', None)

    def fit(self, train_shapes: torch.Tensor):
        """
        :param train_shapes: data matrix of shape (N x F): one row per sample (N), F features each
        """
        # TODO: procrustes analysis
        self.mean_shape = nn.Parameter(train_shapes.mean(0, keepdim=True), requires_grad=False)
        U, S, V = torch.pca_lowrank(train_shapes, q=min(train_shapes.shape), center=True)
        total_variance = S.sum()
        variance_at_sv = (S/total_variance).cumsum(0)

        # number of necessary modes to account for desired portion of the whole variance
        num_modes = (variance_at_sv <= self.target_variance).sum() + 1

        # set config parameters
        self.num_modes = nn.Parameter(num_modes, requires_grad=False)
        self.percent_of_variance = nn.Parameter(variance_at_sv[self.num_modes-1], requires_grad=False)

        # set the model parameters: principal axes
        self.eigenvalues = nn.Parameter(S[None, :self.num_modes], requires_grad=False)
        self.eigenvectors = nn.Parameter(V[None, :, :self.num_modes], requires_grad=False)

        # in case we missed one requires_grad=True Parameter -> set all to False
        self.requires_grad_(False)

    def forward(self, shapes):
        """

        :param shapes: expected dimensionality (batch x Points x 3)
        :return:
        """
        self.assert_trained()

        shapes = shape2vector(shapes)
        projection = torch.matmul(self.eigenvectors.transpose(-1, -2), (shapes - self.mean_shape).unsqueeze(-1))
        return projection.squeeze(-1)

    def decode(self, weights):
        """

        :param weights: batch of weights (batch x self.num_modes)
        :return:
        """
        self.assert_trained()

        reconstruction = self.mean_shape + torch.matmul(self.eigenvectors, weights.unsqueeze(-1)).squeeze(-1)
        return vector2shape(reconstruction, self.dim)

    def random_samples(self, n_samples: int):
        self.assert_trained()

        stddev = torch.sqrt(self.eigenvalues)
        weights = torch.randn(n_samples, self.num_modes) * stddev

        # restrict samples to plausible range +-alpha*stddev
        ranges = self.alpha * stddev
        return torch.clamp(weights, -ranges, ranges)

    def assert_trained(self):
        if self.eigenvectors is None:
            raise ValueError("SSM is not trained yet. You need to call fit before using it.")

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        for key, value in checkpoint['model_state'].items():
            model.register_parameter(key, nn.Parameter(value, requires_grad=False))

        return model


def shape2vector(shape: torch.Tensor):
    return shape.flatten(start_dim=-2)


def vector2shape(vector: torch.Tensor, dimensionality=3):
    assert vector.shape[-1] % dimensionality == 0,\
        f"Vector cannot be unflattened. Last dimension needs be multiple of dimensionality ({dimensionality})."
    return vector.unflatten(dim=-1, sizes=(int(vector.shape[-1] / dimensionality), dimensionality))


if __name__ == '__main__':
    # load data
    shape_folder = "results/corresponding_points"
    files = glob.glob(os.path.join(shape_folder, '*.npy'))
    shapes = []
    for f in files:
        arr = np.load(f)
        arr = np.concatenate((arr[0], arr[1]), axis=0)
        shapes.append(torch.from_numpy(arr))

    shapes = torch.stack(shapes, dim=0)
    shapes = shape2vector(shapes)

    sm = SSM(alpha=2.5, target_variance=0.95)
    sm.fit(shapes)

    sm.save(shape_folder+'/ssm.pth')
    sm_reloaded = SSM.load(shape_folder+'/ssm.pth', 'cpu')

    weights = sm(vector2shape(shapes))
    restored = sm.decode(weights)

    restored_2 = sm.decode(weights)

    samples = sm.random_samples(10)
