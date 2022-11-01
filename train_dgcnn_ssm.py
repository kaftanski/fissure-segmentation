import os
import warnings

import torch
from matplotlib import pyplot as plt

from augmentations import compose_transform
from cli.cl_args import get_dgcnn_ssm_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from data import CorrespondingPointDataset
from data_processing.keypoint_extraction import POINT_DIR, POINT_DIR_TS
from losses.dgssm_loss import corresponding_point_distance, DGSSMLoss
from models.dg_ssm import DGSSM
from shape_model.qualitative_evaluation import mode_plot
from shape_model.ssm import vector2shape
from train import run, write_results
from utils.detached_run import maybe_run_detached_cli
from visualization import point_cloud_on_axis


def test(ds: CorrespondingPointDataset, device, out_dir, show):
    ds.do_augmentation_correspondingly = False

    model = DGSSM.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    pred_dir = os.path.join(out_dir, 'test_predictions')
    plot_dir = os.path.join(pred_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # show behavior of ssm modes
    mode_plot(model.ssm, savepath=os.path.join(plot_dir, 'ssm_modes.png'), show=show)

    corr_point_dist = torch.zeros(len(ds), ds.num_classes)
    corr_point_dist_sd = torch.zeros_like(corr_point_dist)
    corr_point_dist_hd = torch.zeros_like(corr_point_dist)
    corr_point_dist_hd95 = torch.zeros_like(corr_point_dist)
    ssm_error_baseline = torch.zeros_like(corr_point_dist)
    weight_stats = torch.zeros(len(ds), model.ssm.num_modes.data)
    weight_stats_ssm = torch.zeros_like(weight_stats)
    for i in range(len(ds)):
        case, sequence = ds.ids[i]

        input_pts, input_lbls = ds.get_full_pointcloud(i)
        input_pts = input_pts.unsqueeze(0).to(device)
        corr_pts_affine_reg = ds.corr_points[i].unsqueeze(0).to(device)
        corr_pts_not_affine = ds.corr_points.get_points_without_affine_reg(i).to(device)
        with torch.no_grad():
            # make whole model prediction
            prediction = model.dgcnn.predict_full_pointcloud(input_pts)  # TODO: does this limit the expressiveness?
            pred_weights, pred_rotation, pred_translation, pred_scaling = model.split_prediction(prediction)
            reconstructions = model.ssm.decode(pred_weights)

            pred_transform = compose_transform(pred_rotation, pred_translation, pred_scaling)
            reconstructions = pred_transform.transform_points(reconstructions)
            reconstructions = ds.unnormalize_pc(reconstructions, i)

            # test SSM separately for baseline reconstruction
            ssm_weights = model.ssm(ds.normalize_pc(corr_pts_affine_reg, index=i))
            reconstruction_baseline = model.ssm.decode(ssm_weights)
            reconstruction_baseline = ds.unnormalize_pc(reconstruction_baseline, index=i)

        weight_stats[i] += pred_weights.squeeze().cpu()
        weight_stats_ssm[i] += ssm_weights.squeeze().cpu()

        error = corresponding_point_distance(reconstructions, corr_pts_not_affine).cpu()
        baseline_error = corresponding_point_distance(reconstruction_baseline, corr_pts_affine_reg).cpu()
        for c in range(ds.num_classes):
            corr_point_dist[i, c] = error[0, ds.corr_points.label == c+1].mean()
            corr_point_dist_sd[i, c] = error[0, ds.corr_points.label == c+1].std()
            corr_point_dist_hd[i, c] = error[0, ds.corr_points.label == c+1].max()
            corr_point_dist_hd95[i, c] = torch.quantile(error[0, ds.corr_points.label == c+1], q=0.95)

            ssm_error_baseline[i, c] = baseline_error[0, ds.corr_points.label == c + 1].mean()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, reconstructions.cpu(), c='r', cmap=None, title='DG-SSM prediction', label='prediction')
        point_cloud_on_axis(ax, corr_pts_affine_reg.cpu(), c='b', cmap=None, title='DG-SSM prediction', label='target')
        fig.savefig(os.path.join(plot_dir, f'{case}_{sequence}_reconstruction.png'), bbox_inches='tight', dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f'Corr. point distance: {corr_point_dist.mean().item():.4f} mm +- {corr_point_dist.std().item():.4f} mm')
    print(f'SSM reconstruction error: {ssm_error_baseline.mean().item():.4f} mm +- {ssm_error_baseline.std().item():.4f} mm')

    # sanity check / baseline: distance from mean shape to test shapes
    mean_shape_distance = corresponding_point_distance(ds.unnormalize_mean_pc(vector2shape(model.ssm.mean_shape.data)),  # todo: this is not really applicable, we'd need the mean image sizes of the train-ds!
                                                       ds.corr_points.get_shape_datamatrix_with_affine_reg().to(device))
    print(f'Distance between mean SSM-shape and test shapes: {mean_shape_distance.mean().item():.4f} mm +- {mean_shape_distance.std().item():.4f} mm')

    # compare range of predicted weights to model knowledge
    print(f'StdDev of pred. weights: {weight_stats.std(dim=0)} \n\tModel StdDev: {model.ssm.eigenvalues.sqrt().squeeze()}')

    # results
    mean_corr_pt_dist = corr_point_dist.mean(0)
    std_corr_pt_dist = corr_point_dist.std(0)

    mean_corr_pt_dist_sd = corr_point_dist_sd.mean(0)
    std_corr_pt_dist_sd = corr_point_dist_sd.std(0)

    mean_corr_pt_dist_hd = corr_point_dist_hd.mean(0)
    std_corr_pt_dist_hd = corr_point_dist_hd.std(0)

    mean_corr_pt_dist_hd95 = corr_point_dist_hd95.mean(0)
    std_corr_pt_dist_hd95 = corr_point_dist_hd95.std(0)

    dice_dummy = torch.zeros_like(mean_corr_pt_dist)

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), dice_dummy, dice_dummy,
                  mean_corr_pt_dist, std_corr_pt_dist, mean_corr_pt_dist_sd, std_corr_pt_dist_sd,
                  mean_corr_pt_dist_hd, std_corr_pt_dist_hd, mean_corr_pt_dist_hd95, std_corr_pt_dist_hd95,
                  ssm_baseline_error=ssm_error_baseline.mean(0), ssm_baseline_error_std=ssm_error_baseline.std(0),
                  mean_distance_to_mean_shape=mean_shape_distance.mean(),
                  std_distance_to_mean_shape=mean_shape_distance.std())

    return dice_dummy, dice_dummy, mean_corr_pt_dist, std_corr_pt_dist, mean_corr_pt_dist_sd, std_corr_pt_dist_sd, \
           mean_corr_pt_dist_hd, std_corr_pt_dist_hd, mean_corr_pt_dist_hd95, std_corr_pt_dist_hd95


if __name__ == '__main__':
    parser = get_dgcnn_ssm_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    if args.data == 'lobes':
        raise NotImplementedError()
    if args.binary:
        warnings.warn('--binary option has no effect when training the DG-SSM')
    if args.exclude_rhf:
        warnings.warn('--exclude_rhf option has no effect when training the DG-SSM')

    # construct dataset
    if args.ds == 'data':
        point_dir = POINT_DIR
        img_dir = '../data'
    elif args.ds == 'ts':
        point_dir = POINT_DIR_TS
        img_dir = '../TotalSegmentator/ThoraxCrop'
    else:
        raise ValueError(f'No dataset named {args.ds}')

    corr_pt_dir = f'results/corresponding_points{"_ts" if args.ds=="ts" else ""}/{args.data}/{args.corr_mode}'

    ds = CorrespondingPointDataset(point_folder=point_dir, image_folder=img_dir, corr_folder=corr_pt_dir,
                                   sample_points=args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                                   patch_feat=args.patch, undo_affine_reg=args.predict_affine, do_augmentation=True)

    # setup model
    in_features = ds[0][0].shape[0]
    model = DGSSM(k=args.k, in_features=in_features,
                  spatial_transformer=args.transformer, dynamic=not args.static,
                  ssm_alpha=args.alpha, ssm_targ_var=args.target_variance, lssm=args.lssm,
                  predict_affine_params=args.predict_affine)

    if not args.predict_affine:
        # set the loss weight for affine params to zero
        if args.loss_weights is None:
            args.loss_weights = [DGSSMLoss.DEFAULT_W_POINT, DGSSMLoss.DEFAULT_W_COEFFICIENTS, 0]
        else:
            args.loss_weights[2] = 0

    # save setup
    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    run(ds, model, test, args)
