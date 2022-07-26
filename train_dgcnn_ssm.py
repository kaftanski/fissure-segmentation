import os
import warnings

import torch
from matplotlib import pyplot as plt

from cli.cl_args import get_dgcnn_ssm_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from data import CorrespondingPointDataset
from losses.ssm_loss import corresponding_point_distance
from models.dg_ssm import DGSSM
from shape_model.ssm import SSM
from train import run
from visualization import point_cloud_on_axis


def test(ds: CorrespondingPointDataset, device, out_dir, show):
    # device = 'cpu'

    model = DGSSM.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    pred_dir = os.path.join(out_dir, 'test_predictions')
    plot_dir = os.path.join(pred_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    corr_point_errors = torch.zeros(len(ds), ds.num_classes)
    ssm_error_baseline = torch.zeros_like(corr_point_errors)
    weight_stats = torch.zeros(len(ds), model.ssm.num_modes.data)
    for i in range(len(ds)):
        case, sequence = ds.ids[i]

        input_pts, input_lbls = ds.get_full_pointcloud(i)
        input_pts = input_pts.unsqueeze(0).to(device)
        corr_pts = ds.corr_points[i].unsqueeze(0).to(device)

        with torch.no_grad():
            # make whole model prediction
            pred_weights = model.dgcnn.predict_full_pointcloud(input_pts)
            reconstructions = model.ssm.decode(pred_weights)

            # test only the SSM for baseline reconstruction
            reconstruction_baseline = model.ssm.decode(model.ssm(corr_pts))

        weight_stats[i] += pred_weights.squeeze().cpu()

        error = corresponding_point_distance(reconstructions, corr_pts).cpu()
        baseline_error = corresponding_point_distance(reconstruction_baseline, corr_pts).cpu()
        for c in range(ds.num_classes):
            corr_point_errors[i, c] = error[0, ds.corr_points.label == c+1].mean()
            ssm_error_baseline[i, c] = baseline_error[0, ds.corr_points.label == c + 1].mean()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, reconstructions.cpu(), c='r', cmap=None, title='DG-SSM prediction', label='prediction')
        point_cloud_on_axis(ax, corr_pts.cpu(), c='b', cmap=None, title='DG-SSM prediction', label='target')
        fig.savefig(os.path.join(plot_dir, f'{case}_{sequence}_reconstruction.png'), bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f'Corr. point distance: {corr_point_errors.mean().item():.4f} +- {corr_point_errors.std().item():.4f}')
    print(f'SSM reconstruction error: {ssm_error_baseline.mean().item():.4f} +- {ssm_error_baseline.std().item():.4f}')
    print(f'StdDev of pred. weights: {weight_stats.std(dim=0)} \n\tModel StdDev: {model.ssm.eigenvalues.sqrt().squeeze()}')


if __name__ == '__main__':
    parser = get_dgcnn_ssm_train_parser()
    args = parser.parse_args()

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)
    else:
        store_args(args=args, out_dir=args.output)

    if args.data == 'lobes':
        raise NotImplementedError()
    if args.binary:
        warnings.warn('--binary option has no effect when training the DG-SSM')
    if args.exclude_rhf:
        warnings.warn('--exclude_rhf option has no effect when training the DG-SSM')

    ds = CorrespondingPointDataset(sample_points=args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                                   patch_feat='mind' if args.patch else None)

    # setup model
    in_features = ds[0][0].shape[0]
    model = DGSSM(k=args.k, in_features=in_features,
                  spatial_transformer=args.transformer, dynamic=not args.static,
                  ssm_alpha=args.alpha, ssm_targ_var=args.target_variance)

    run(ds, model, test, args)