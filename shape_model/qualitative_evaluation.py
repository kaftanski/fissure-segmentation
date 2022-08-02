import glob

import os

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from losses.ssm_loss import corresponding_point_distance
from shape_model.ssm import SSM, load_shape
from visualization import point_cloud_on_axis, visualize_point_cloud


def visualize_reconstruction(pred: torch.Tensor, targ: torch.Tensor, savepath: str = None, show: bool = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_cloud_on_axis(ax, pred.cpu(), c='r', cmap=None, title='SSM reconstruction', label='prediction')
    point_cloud_on_axis(ax, targ.cpu(), c='b', cmap=None, title='SSM reconstruction', label='target')

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_samples(model: SSM, n_samples: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # generate some samples
    sampled_weights = model.random_samples(n_samples)
    samples = model.decode(sampled_weights)

    for i, sample in enumerate(samples):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, sample.cpu(), c='r', cmap=None, title='SSM sample')
        fig.savefig(os.path.join(out_dir, f'smpl_{i}.png'), dpi=300, bbox_inches='tight')
        plt.show()


def latent_interpolation(shape_from: torch.Tensor, shape_to: torch.Tensor, model: SSM, steps: int,
                         savepath: str = None, show: bool = True):

    n_cols = steps + 4

    fig = plt.figure(figsize=(3*n_cols, 5))
    ax0 = fig.add_subplot(1, n_cols, 1, projection='3d')
    point_cloud_on_axis(ax0, shape_from, c='b', cmap=None, title='Training Shape 1')

    weights_from = model(shape_from)
    weights_to = model(shape_to)
    weights_diff = weights_to - weights_from
    for s in range(steps + 2):
        interp_weights = weights_from + s / (steps + 1) * weights_diff
        if s == 0:
            assert torch.allclose(interp_weights, weights_from)
            title = 'Recons. 1'
        elif s == steps+1:
            assert torch.allclose(interp_weights, weights_to)
            title = 'Recons. 2'
        else:
            title = f'Interp. Step {s}'

        reconstruction = model.decode(interp_weights)
        ax_s = fig.add_subplot(1, n_cols, s + 2, projection='3d')
        point_cloud_on_axis(ax_s, reconstruction, c='r', cmap=None, title=title)

    ax_end = fig.add_subplot(1, n_cols, n_cols, projection='3d')
    point_cloud_on_axis(ax_end, shape_to, c='b', cmap=None, title='Training Shape 2')

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)


@torch.no_grad()
def mode_plot(ssm: SSM, modes_to_plot=3, plots_per_mode=5, savepath=None):
    device = ssm.eigenvalues.device

    fig = plt.figure(figsize=(3*plots_per_mode, 5*modes_to_plot))
    for mode in range(modes_to_plot):
        coeff_range = torch.linspace(-ssm.alpha, ssm.alpha, steps=plots_per_mode, device=device) * ssm.eigenvalues[0, mode].sqrt()
        for plot in range(plots_per_mode+1):
            if plot == 0:
                ax = fig.add_subplot(modes_to_plot, plots_per_mode + 1, mode * (plots_per_mode + 1) + plot + 1)
                ax.text(x=1, y=0.5, s=f'$\lambda_{mode+1}$', horizontalalignment="right", verticalalignment="center")
                ax.set_axis_off()
            else:
                ax = fig.add_subplot(modes_to_plot, plots_per_mode + 1, mode * (plots_per_mode + 1) + plot + 1,
                                     projection='3d')
                coefficients = torch.zeros(1, ssm.num_modes, device=device)
                coefficients[0, mode] = coeff_range[plot-1]
                shape = ssm.decode(coefficients).squeeze()
                point_cloud_on_axis(ax, shape, c='r', cmap=None)

            if mode == 0:
                if plot == 1:
                    ax.set_title('$-\\alpha\sqrt{\lambda_i}$')
                if plot == plots_per_mode:
                    ax.set_title('$\\alpha\sqrt{\lambda_i}$')

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':
    result_dir = './results/shape_models/'

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

    # plot some reconstructions and compute error
    rec_dir = os.path.join(result_dir, 'reconstructions')
    os.makedirs(rec_dir, exist_ok=True)
    errors = []
    for i, (pred, targ) in enumerate(zip(restored, shapes)):
        visualize_reconstruction(pred, targ, os.path.join(rec_dir, f'rec_{i}.png'))
        errors.append((pred-targ).pow(2).sum(-1).sqrt())

    errors = torch.stack(errors, dim=0)
    print('Reconstruction error (corresponding points):', errors.mean().item(), '+-', errors.std().item(),
          f'(Hausdorff {errors.max().item()})')

    # try half precision
    model.half()
    weights = model(shapes.half())
    restored = model.decode(weights)
    errors = corresponding_point_distance(shapes, restored)
    print('Half precision:', errors.mean().item(), '+-', errors.std().item(), 'HD:', errors.max())

    # visualize random samples
    visualize_samples(model, 100, os.path.join(result_dir, 'samples'))

    # visualize interpolations
    interp_dir = os.path.join(result_dir, 'interpolations')
    os.makedirs(interp_dir, exist_ok=True)
    n = 10
    for i in range(n):
        ind_from, ind_to = torch.randperm(len(shapes))[:2]
        latent_interpolation(shapes[ind_from:ind_from+1], shapes[ind_to:ind_to+1], model, steps=3,
                             savepath=os.path.join(interp_dir, f'interp_{ind_from}_{ind_to}.png'), show=True)
