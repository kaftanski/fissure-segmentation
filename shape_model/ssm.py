import glob
import os.path

import numpy as np
import torch
from torch import nn
from models.modelio import store_config_args, LoadableModel


class SSM(LoadableModel):
    @store_config_args
    def __init__(self, alpha=2.5, target_variance=0.95, dimensionality=3, _num_modes=None, _percent_of_variance=None):
        super().__init__()
        self.target_variance = target_variance
        self.alpha = alpha
        self.dim = dimensionality

        self.num_modes = _num_modes
        self.percent_of_variance = _percent_of_variance

        # register parameters that will be set by calling SSM.fit
        # these can be used like normal python attributes!
        self.register_parameter('mean_shape', None)
        self.register_parameter('eigenvalues', None)
        self.register_parameter('eigenvectors', None)

        # this module is fixed (parameters should not be adjusted during training)
        self.eval()
        self.training = False

    def fit(self, train_shapes: torch.Tensor):
        """
        :param train_shapes: data matrix of shape (N x F): one row per sample (N), F features each
            (alternatively: batch of N shapes with P points each, shape (N x P x 3)
        """
        if len(train_shapes.shape) == 3 and train_shapes.shape[-1] == self.dim:
            train_shapes = shape2vector(train_shapes)

        # TODO: procrustes analysis
        self.mean_shape = nn.Parameter(train_shapes.mean(0, keepdim=True))
        U, S, V = torch.pca_lowrank(train_shapes, q=min(train_shapes.shape), center=True)
        total_variance = S.sum()
        variance_at_sv = (S/total_variance).cumsum(0)

        # number of necessary modes to account for desired portion of the whole variance
        self.num_modes = (variance_at_sv <= self.target_variance).sum() + 1
        self.percent_of_variance = variance_at_sv[self.num_modes-1]

        # set config parameters
        self.config['num_modes'] = self.num_modes
        self.config['percent_of_variance'] = self.percent_of_variance

        # set the model parameters: principal axes
        self.eigenvalues = nn.Parameter(S[None, :self.num_modes])
        self.eigenvectors = nn.Parameter(V[None, :, :self.num_modes])

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
        weights = weights.view(*weights.shape[:2], 1)
        reconstruction = self.mean_shape + torch.matmul(self.eigenvectors, weights).squeeze(-1)
        return vector2shape(reconstruction, self.dim)

    def random_samples(self, n_samples: int):
        self.assert_trained()

        stddev = torch.sqrt(self.eigenvalues)
        ranges = self.alpha * stddev
        return torch.rand(n_samples, self.num_modes.data, device=stddev.device, dtype=self.eigenvectors.dtype) * 2 * ranges - ranges
        # weights = torch.randn(n_samples, self.num_modes.data, device=stddev.device) * stddev
        #
        # # restrict samples to plausible range +-alpha*stddev
        # return torch.clamp(weights, -ranges, ranges)

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

    def train(self, mode=True):
        # this module is fixed (parameters should not be adjusted during training)
        return self


def shape2vector(shape: torch.Tensor):
    return shape.flatten(start_dim=-2)


def vector2shape(vector: torch.Tensor, dimensionality=3):
    assert vector.shape[-1] % dimensionality == 0,\
        f"Vector cannot be unflattened. Last dimension needs be multiple of dimensionality ({dimensionality})."
    return vector.unflatten(dim=-1, sizes=(int(vector.shape[-1] / dimensionality), dimensionality))


def save_shape(array, filepath):
    np.save(filepath, array)


def load_shape(filepath, return_labels=False):
    arr = np.load(filepath)

    # unpack all objects (first dimension)
    arr_concat = torch.from_numpy(np.concatenate([*arr], axis=0)).float()

    if return_labels:
        # generate pointwise labels
        labels = torch.from_numpy(
            np.concatenate([np.full(arr.shape[1], fill_value=i + 1) for i in range(arr.shape[0])]))
        return arr_concat, labels
    else:
        return arr_concat


if __name__ == '__main__':
    # load data
    shape_folder = "results/corresponding_points"
    files = glob.glob(os.path.join(shape_folder, '*.npy'))
    shapes = []
    for f in files:
        shapes.append(load_shape(f))

    shapes = torch.stack(shapes, dim=0)
    shapes = shape2vector(shapes)

    sm = SSM(alpha=3, target_variance=0.95)
    sm.fit(shapes)

    sm.save(shape_folder+'/ssm.pth')
    sm_reloaded = SSM.load(shape_folder+'/ssm.pth', 'cpu')

    weights = sm(vector2shape(shapes))
    restored = sm.decode(weights)

    restored_2 = sm.decode(weights)

    samples = sm.random_samples(10)
