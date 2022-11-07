import glob
import os.path
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from losses.dgssm_loss import corresponding_point_distance
from models.modelio import store_config_args, LoadableModel
from shape_model.LPCA.model import LPCA


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

        # this module is fixed (parameters should not be adjusted during training)
        self.requires_grad_(False)

    def fit(self, train_shapes: torch.Tensor):
        """
        :param train_shapes: data matrix of shape (N x F): one row per sample (N), F features each
            (alternatively: batch of N shapes with P points each, shape (N x P x 3)
        """
        if len(train_shapes.shape) == 3 and train_shapes.shape[-1] == self.dim:
            train_shapes = shape2vector(train_shapes)

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
        model.register_parameters_from_state_dict(checkpoint['model_state'])
        return model

    def register_parameters_from_state_dict(self, state_dict):
        for key, value in state_dict.items():
            self.register_parameter(key, nn.Parameter(value, requires_grad=False))


class LSSM(SSM):
    """
    Using kernelized LSSM from:
    @inproceedings{miccai2020,
        Author = {Matthias Wilms and Jan Ehrhardt and Nils Daniel Forkert},
        Title = {A Kernelized Multi-level Localization Method for Flexible Shape Modeling with Few Training Data},
        Booktitle = {Medical Image Computing and Computer Assisted Intervention -- {MICCAI 2020}},
        Year = {2020}
    }

    Original LSSM paper (with normal LPCA):
    @article{media2017,
        Title = {Multi-resolution multi-object statistical shape models based on the locality assumption},
        Author = {Matthias Wilms and Heinz Handels and Jan Ehrhardt},
        Journal = {Medical Image Analysis},
        Year = {2017},
        Number = {5},
        Pages = {17--29},
        Volume = {38}
    }
    """

    def __init__(self, alpha=2.5, target_variance=0.95, dimensionality=3):
        super(LSSM, self).__init__(alpha, target_variance, dimensionality)
        self.lpca = LPCA(num_levels=5, target_variation=target_variance)

    def fit(self, train_shapes: torch.Tensor):
        if len(train_shapes.shape) == 3 and train_shapes.shape[-1] == self.dim:
            train_shapes = shape2vector(train_shapes)

        device = train_shapes.device

        # run the training (only implemented in Numpy)
        train_shapes = train_shapes.cpu().numpy().T  # model expects data matrix with shapes in columns
        mean_shape, eigenvectors, eigenvalues, num_modes, percent_of_variance = \
            self.lpca.lpca(train_shapes)

        # convert the results to pytorch Parameters
        self.eigenvalues = nn.Parameter(torch.from_numpy(eigenvalues).unsqueeze(0).float(), requires_grad=False)
        self.eigenvectors = nn.Parameter(torch.from_numpy(eigenvectors).unsqueeze(0).float(), requires_grad=False)
        self.mean_shape = nn.Parameter(torch.from_numpy(mean_shape.T).float(), requires_grad=False)
        self.num_modes = nn.Parameter(torch.tensor(num_modes), requires_grad=False)
        self.percent_of_variance = nn.Parameter(torch.tensor(percent_of_variance, dtype=torch.float), requires_grad=False)

        self.requires_grad_(False)  # just to be sure nothing will be changed during training
        self.to(device)


def shape2vector(shape: torch.Tensor):
    return shape.flatten(start_dim=-2)


def vector2shape(vector: torch.Tensor, dimensionality=3):
    assert vector.shape[-1] % dimensionality == 0,\
        f"Vector cannot be unflattened. Last dimension needs be multiple of dimensionality ({dimensionality})."
    return vector.unflatten(dim=-1, sizes=(int(vector.shape[-1] / dimensionality), dimensionality))


def save_shape(array, filepath, transforms=None):
    """

    :param array: shape [N_objects, N_points, 3]
    :param filepath: path to save to
    :param transforms: affine pre-registration used when constructing the corresponding point data set (shape [N_objects, 3, 4]).
        uses identity by default.
    """
    if transforms is None:
        transforms = {'scale': 1, 'rotation': np.eye(3, 3), 'translation': np.zeros(3), 'is_applied': True}
    with open(filepath, 'wb') as file:
        pickle.dump(OrderedDict({'shape': array, 'transform': transforms}), file)


def load_shape(filepath, return_labels=False):
    file = np.load(filepath, allow_pickle=True)
    arr = torch.from_numpy(file['shape']).float()
    trf = file['transform']
    trf['rotation'] = torch.from_numpy(trf['rotation']).float()
    trf['translation'] = torch.from_numpy(trf['translation']).float()

    if arr.ndim == 3:
        # generate pointwise labels
        labels = torch.from_numpy(
            np.concatenate([np.full(arr.shape[1], fill_value=i + 1) for i in range(arr.shape[0])]))

        # unpack all objects (first dimension)
        arr = torch.cat([*arr], dim=0)

        if return_labels:
            return arr, trf, labels

    else:
        if return_labels:
            # label file expected to be in the same folder as the shape
            base_path = os.path.split(filepath)[0]
            labels = torch.from_numpy(
                np.load(os.path.join(base_path, "labels.npz"), allow_pickle=True))
            return arr, trf, labels

    return arr, trf


if __name__ == '__main__':
    # load data
    shape_folder = "results/corresponding_points/fissures/simple"
    files = sorted(glob.glob(os.path.join(shape_folder, '*_corr_pts.npz')))
    shapes = []
    for f in files:
        shapes.append(load_shape(f)[0])

    shapes = torch.stack(shapes, dim=0)

    train_index = int(0.8 * len(shapes))
    train_shapes = shapes[:train_index]
    test_shapes = shapes[train_index:]

    sm = SSM(alpha=3, target_variance=0.95)
    sm.fit(shape2vector(train_shapes))

    # test reconstruction
    test_predictions = sm.decode(sm(test_shapes))
    error = corresponding_point_distance(test_predictions, test_shapes)
    print(error.mean().item(), '+-', error.std().item())

    # test saving and loading
    sm.save(shape_folder+'/ssm.pth')
    sm_reloaded = SSM.load(shape_folder+'/ssm.pth', 'cpu')

    weights = sm(shapes)
    restored = sm.decode(weights)

    restored_2 = sm.decode(weights)
    assert torch.allclose(restored_2, restored)
