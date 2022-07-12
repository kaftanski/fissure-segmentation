import glob

import os

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from shape_model.ssm import SSM, load_shape
from visualization import point_cloud_on_axis

if __name__ == '__main__':
    # load model
    device = 'cuda:3'
    model = SSM.load('./results/corresponding_points/ssm.pth', device)
    model.to(device)
    model.eval()

    # load data
    shape_folder = "results/corresponding_points"
    files = glob.glob(os.path.join(shape_folder, '*.npy'))
    shapes = []
    for f in files:
        shapes.append(load_shape(f))

    shapes = torch.stack(shapes, dim=0).to(device)

    # reconstruction
    weights = model(shapes)
    restored = model.decode(weights)

    # # plot some PCs
    errors = []
    for i, (pred, targ) in enumerate(zip(restored, shapes)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, pred.cpu(), c='r', cmap=None, title='SSM reconstruction', label='prediction')
        point_cloud_on_axis(ax, targ.cpu(), c='b', cmap=None, title='SSM reconstruction', label='target')
        fig.savefig(f'./results/shape_models/reconstructions/rec_{i}.png', dpi=100)
        plt.show()

        errors.append((pred-targ).pow(2).sum(-1).sqrt())

    errors = torch.stack(errors, dim=0)
    print('Reconstruction error (corresponding points):', errors.mean().item(), '+-', errors.std().item(), f'(Hausdorff {errors.max().item()})')

    # generate some samples
    sampled_weights = model.random_samples(100)
    samples = model.decode(sampled_weights)

    for i, sample in enumerate(samples):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, sample.cpu(), c='r', cmap=None, title='SSM sample')
        fig.savefig(f'./results/shape_models/samples/smpl_{i}.png', dpi=100)
        plt.show()
