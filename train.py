import csv
import os
from typing import List, Tuple

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch

import model_trainer
from cli.cl_args import get_dgcnn_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from constants import POINT_DIR, POINT_DIR_TS, DEFAULT_SPLIT, DEFAULT_SPLIT_TS, IMG_DIR, IMG_DIR_TS
from data import PointDataset, load_split_file, save_split_file, LungData, CorrespondingPointDataset
from data_processing.find_lobes import lobes_to_fissures
from data_processing.surface_fitting import pointcloud_surface_fitting, o3d_mesh_to_labelmap
from losses.access_losses import get_loss_fn
from losses.dgssm_loss import corresponding_point_distance
from metrics import assd, label_mesh_assd, batch_dice
from models.dgcnn import DGCNNSeg
from utils.detached_run import maybe_run_detached_cli
from utils.fissure_utils import binary_to_fissure_segmentation
from utils.utils import kpts_to_world, mask_out_verts_from_mesh, remove_all_but_biggest_component, mask_to_points, \
    points_to_label_map, create_o3d_mesh, nanstd
from visualization import visualize_point_cloud, visualize_o3d_mesh


def train(model, ds, device, out_dir, args):
    # set up loss function
    class_weights = ds.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = get_loss_fn(args.loss, class_weights, args.loss_weights)

    if isinstance(ds, CorrespondingPointDataset):
        train_shapes = ds.get_normalized_corr_datamatrix_with_affine_reg().to(device)
        model.fit_ssm(train_shapes)

        ev_before = model.ssm.eigenvectors.data.clone()
        ms_before = model.ssm.mean_shape.data.clone()

        # compute the train reconstruction error
        with torch.no_grad():
            reconstructions = model.ssm.decode(model.ssm(train_shapes))
            error = corresponding_point_distance(reconstructions, train_shapes)
            print('SSM train reconstruction error:', error.mean().item(), '+-', error.std().item())

    # run training
    trainer = model_trainer.ModelTrainer(model, ds, criterion, out_dir, device, args)
    trainer.run(initial_epoch=0)

    if isinstance(ds, CorrespondingPointDataset):
        # assert that the SSM has not been changed
        assert torch.all(model.ssm.eigenvectors.data == ev_before), \
            'SSM parameters have changed. This should not have happened'
        assert torch.all(model.ssm.mean_shape.data == ms_before), \
            'SSM parameters have changed. This should not have happened'

        with torch.no_grad():
            reconstructions = model.ssm.decode(model.ssm(train_shapes))
            error = corresponding_point_distance(reconstructions, train_shapes)
            print('SSM train reconstruction error:', error.mean().item(), '+-', error.std().item())


def compute_mesh_metrics(meshes_predict: List[List[o3d.geometry.TriangleMesh]],
                         meshes_target: List[List[o3d.geometry.TriangleMesh]],
                         ids: List[Tuple[str, str]] = None,
                         show: bool = False, spacings=None, plot_folder=None):
    # metrics
    # test_dice = torch.zeros(len(meshes_predict), len(meshes_predict[0]))
    avg_surface_dist = torch.zeros(len(meshes_predict), len(meshes_target[0]))
    std_surface_dist = torch.zeros_like(avg_surface_dist)
    hd_surface_dist = torch.zeros_like(avg_surface_dist)
    hd95_surface_dist = torch.zeros_like(avg_surface_dist)

    for i, (all_parts_predictions, all_parts_targets) in enumerate(zip(meshes_predict, meshes_target)):
        pred_points = []
        for j, targ_part in enumerate(all_parts_targets):
            try:
                pred_part = all_parts_predictions[j]
                if isinstance(pred_part, torch.Tensor):
                    asd, sdsd, hdsd, hd95sd, points = label_mesh_assd(pred_part, targ_part, spacing=spacings[i])
                    pred_points.append(points)
                else:
                    asd, sdsd, hdsd, hd95sd = assd(pred_part, targ_part)

                avg_surface_dist[i, j] += asd
                std_surface_dist[i, j] += sdsd
                hd_surface_dist[i, j] += hdsd
                hd95_surface_dist[i, j] += hd95sd

            except IndexError:
                print(f'Fissure {j+1} is missing from prediction.')
                avg_surface_dist[i, j] += float('NaN')
                std_surface_dist[i, j] += float('NaN')
                hd_surface_dist[i, j] += float('NaN')
                hd95_surface_dist[i, j] += float('NaN')

        # visualize results
        if (plot_folder is not None) or show:
            if ids is not None:
                case, sequence = ids[i]
                title_prefix = f'{case}_{sequence}'
            else:
                title_prefix = f'sample {i}'

            if pred_points:
                all_points = torch.concat(pred_points, dim=0)
                lbls = torch.concat([torch.ones(len(pts), dtype=torch.long) + p for p, pts in enumerate(pred_points)])
                visualize_point_cloud(all_points, labels=lbls, title=title_prefix + ' points prediction', show=show,
                                      savepath=None if plot_folder is None else os.path.join(plot_folder, f'{title_prefix}_point_cloud_pred.png'))
            else:
                visualize_o3d_mesh(all_parts_predictions, title=title_prefix + ' surface prediction', show=show,
                                   savepath=os.path.join(plot_folder, f'{title_prefix}_mesh_pred.png'))

            visualize_o3d_mesh(all_parts_targets, title=title_prefix + ' surface target', show=show,
                              savepath=os.path.join(plot_folder, f'{title_prefix}_mesh_targ.png'))

    # compute average metrics (excluding NaN values)
    mean_assd = avg_surface_dist.nanmean(0)
    std_assd = nanstd(avg_surface_dist, 0)

    mean_sdsd = std_surface_dist.nanmean(0)
    std_sdsd = nanstd(std_surface_dist, 0)

    mean_hd = hd_surface_dist.nanmean(0)
    std_hd = nanstd(hd_surface_dist, 0)

    mean_hd95 = hd95_surface_dist.nanmean(0)
    std_hd95 = nanstd(hd95_surface_dist, 0)

    # compute proportion of missing objects
    percent_missing = avg_surface_dist.isnan().float().mean(0) * 100

    return mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing


def test(ds: PointDataset, device, out_dir, show):
    print('\nTESTING MODEL ...\n')
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    img_ds = LungData(ds.image_folder)

    net = DGCNNSeg.load(os.path.join(out_dir, 'model.pth'), device=device)
    net.to(device)
    net.eval()

    # get the non-binarized labels from the dataset
    dataset_binary = ds.binary
    ds.binary = False

    # directory for output predictions
    pred_dir = os.path.join(out_dir, 'test_predictions')
    mesh_dir = os.path.join(pred_dir, 'meshes')
    label_dir = os.path.join(pred_dir, 'labelmaps')
    plot_dir = os.path.join(pred_dir, 'plots')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # compute all predictions
    all_pred_meshes = []
    all_targ_meshes = []
    ids = []
    test_dice = torch.zeros(len(ds), ds.num_classes)
    for i in range(len(ds)):
        inputs, lbls = ds.get_full_pointcloud(i)
        inputs = inputs.unsqueeze(0).to(device)
        lbls = lbls.unsqueeze(0).to(device)

        with torch.no_grad():
            out = net.predict_full_pointcloud(inputs, ds.sample_points, n_runs_min=50)

        labels_pred = out.argmax(1)

        # convert points back to world coordinates
        pts = inputs[0, :3]  # coords are the first 3 features
        case, sequence = ds.ids[i]
        img_index = img_ds.get_index(case, sequence)
        spacing = torch.tensor(ds.spacings[i], device=device)
        shape = torch.tensor(ds.img_sizes_index[i][::-1], device=device) * spacing.flip(0)
        pts = kpts_to_world(pts.to(device).transpose(0, 1), shape)  # points in millimeters

        # POST-PROCESSING prediction
        mask_img = img_ds.get_lung_mask(img_index)
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask_img).astype(bool))

        if ds.lobes:
            # load the right target meshes (2 fissures in case of 4 lobes, 3 otherwise)
            meshes_target = img_ds.get_fissure_meshes(img_index)[:2 if len(lbls.unique()[1:]) == 4 else 3]

            # voxelize point labels (sparse)
            lobes_tensor = torch.zeros_like(mask_tensor, dtype=torch.long)
            pts_index = (pts / spacing).flip(-1).round().long().cpu()
            lobes_tensor[pts_index[:, 0], pts_index[:, 1], pts_index[:, 2]] = labels_pred.squeeze().cpu()

            # point labels are seeds for random walk, filling in the gaps
            sparse_lobes_img = sitk.GetImageFromArray(lobes_tensor.numpy().astype(np.uint8))
            sparse_lobes_img.CopyInformation(mask_img)
            fissure_pred_img, lobes_pred_img = lobes_to_fissures(sparse_lobes_img, mask_img, device=device)

            # write out intermediate results
            sitk.WriteImage(sparse_lobes_img, os.path.join(label_dir, f'{case}_lobes_pointcloud_{sequence}.nii.gz'))
            sitk.WriteImage(fissure_pred_img, os.path.join(label_dir, f'{case}_fissures_from_lobes_{sequence}.nii.gz'))
            sitk.WriteImage(lobes_pred_img, os.path.join(label_dir, f'{case}_lobes_pred_{sequence}.nii.gz'))

        else:
            meshes_target = img_ds.get_fissure_meshes(img_index)[:2 if ds.exclude_rhf else 3]

            if net.num_classes == 2:  # binary prediction
                # voxelize point labels
                fissure_tensor, pts_index = points_to_label_map(pts, labels_pred.squeeze(), mask_tensor.shape, spacing=ds.spacings[i])

                # infer right/left fissure labels from lung mask
                fissure_tensor = binary_to_fissure_segmentation(fissure_tensor, lr_lung_mask=img_ds.get_left_right_lung_mask(i))

                # assign labels to points
                labels_pred = fissure_tensor[pts_index[:, 0], pts_index[:, 1], pts_index[:, 2]].unsqueeze(0)

        # visualize point clouds
        visualize_point_cloud(pts, labels_pred.squeeze(), title=f'{case}_{sequence} point cloud prediction', show=show,
                              savepath=os.path.join(plot_dir, f'{case}_{sequence}_point_cloud_pred.png'))
        visualize_point_cloud(pts, lbls.squeeze(), title=f'{case}_{sequence} point cloud target', show=show,
                              savepath=os.path.join(plot_dir, f'{case}_{sequence}_point_cloud_targ.png'))

        # compute point dice score
        test_dice[i] += batch_dice(labels_pred, lbls, ds.num_classes)

        # mesh fitting for each label
        meshes_predict = []
        for j in range(len(meshes_target)):  # excluding background
            label = j+1
            try:
                if not ds.lobes:
                    # if ds.kp_mode == 'foerstner':
                    #     # using poisson reconstruction with octree-depth 3 because of sparse point cloud
                    #     depth = 3
                    # else:
                    #     # point cloud contains more foreground points because of pre-seg CNN or enhancement
                    #     depth = 6
                    depth = 6
                    mesh_predict = pointcloud_surface_fitting(pts[labels_pred.squeeze() == label].cpu().numpy().astype(float),
                                                              crop_to_bbox=True, depth=depth)
                else:
                    # extract the fissure points from labelmap
                    fissure_pred_pts = mask_to_points(torch.from_numpy(sitk.GetArrayFromImage(fissure_pred_img).astype(int)) == label,
                                                      spacing=fissure_pred_img.GetSpacing())

                    # fit surface to the points with depth 6, because points are dense
                    mesh_predict = pointcloud_surface_fitting(fissure_pred_pts, crop_to_bbox=True, depth=6)

            except ValueError as e:
                # no points have been predicted to be in this class
                print(e)
                mesh_predict = create_o3d_mesh(verts=np.array([]), tris=np.array([]))

            # post-process surfaces
            mask_out_verts_from_mesh(mesh_predict, mask_tensor, spacing)  # apply lung mask
            right = label > 1  # right fissure(s) are label 2 and 3
            remove_all_but_biggest_component(mesh_predict, right=right, center_x=shape[2]/2)  # only keep the biggest connected component

            meshes_predict.append(mesh_predict)

            # write out meshes
            o3d.io.write_triangle_mesh(os.path.join(mesh_dir, f'{case}_fissure{label}_pred_{sequence}.obj'),
                                       mesh_predict)

        # write out label images (converted from surface reconstruction)
        # predicted labelmap
        labelmap_predict = o3d_mesh_to_labelmap(meshes_predict, shape=ds.img_sizes_index[i][::-1], spacing=ds.spacings[i])
        label_image_predict = sitk.GetImageFromArray(labelmap_predict.numpy().astype(np.uint8))
        label_image_predict.CopyInformation(mask_img)
        sitk.WriteImage(label_image_predict, os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))

        # target labelmap
        labelmap_target = o3d_mesh_to_labelmap(meshes_target, shape=ds.img_sizes_index[i][::-1], spacing=ds.spacings[i])
        label_image_target = sitk.GetImageFromArray(labelmap_target.numpy().astype(np.uint8))
        label_image_target.CopyInformation(mask_img)
        sitk.WriteImage(label_image_target, os.path.join(label_dir, f'{case}_fissures_targ_{sequence}.nii.gz'))

        # remember meshes for evaluation
        all_pred_meshes.append(meshes_predict)
        all_targ_meshes.append(meshes_target)
        ids.append((case, sequence))

    # restore previous setting
    ds.binary = dataset_binary

    # compute average metrics
    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing = compute_mesh_metrics(
        all_pred_meshes, all_targ_meshes, ids=ids, show=show, plot_folder=plot_dir)

    print(f'Test dice per class: {mean_dice} +- {std_dice}')
    print(f'ASSD per fissure: {mean_assd} +- {std_assd}')

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing


def write_results(filepath, mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95,
                  std_hd95, proportion_missing=None, **additional_metrics):
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        if mean_dice is not None:
            writer.writerow(['Class'] + [str(i) for i in range(mean_dice.shape[0])] + ['mean'])
            writer.writerow(['Mean Dice'] + [d.item() for d in mean_dice] + [mean_dice.mean().item()])
            writer.writerow(['StdDev Dice'] + [d.item() for d in std_dice] + [std_dice.mean().item()])
            writer.writerow([])
        writer.writerow(['Fissure'] + [str(i+1) for i in range(mean_assd.shape[0])] + ['mean'])
        writer.writerow(['Mean ASSD'] + [d.item() for d in mean_assd] + [mean_assd.mean().item()])
        writer.writerow(['StdDev ASSD'] + [d.item() for d in std_assd] + [std_assd.mean().item()])
        writer.writerow(['Mean SDSD'] + [d.item() for d in mean_sdsd] + [mean_sdsd.mean().item()])
        writer.writerow(['StdDev SDSD'] + [d.item() for d in std_sdsd] + [std_sdsd.mean().item()])
        writer.writerow(['Mean HD'] + [d.item() for d in mean_hd] + [mean_hd.mean().item()])
        writer.writerow(['StdDev HD'] + [d.item() for d in std_hd] + [std_hd.mean().item()])
        writer.writerow(['Mean HD95'] + [d.item() for d in mean_hd95] + [mean_hd95.mean().item()])
        writer.writerow(['StdDev HD95'] + [d.item() for d in std_hd95] + [std_hd95.mean().item()])

        if proportion_missing is None:
            proportion_missing = torch.zeros_like(mean_assd, dtype=torch.float)
        writer.writerow(['proportion missing'] + [d.item() for d in proportion_missing] + [proportion_missing.mean().item()])

        for key, value in additional_metrics.items():
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass

            if isinstance(value, torch.Tensor):
                writer.writerow([key] + [v.item() for v in value])
            else:
                writer.writerow([key, value])


def cross_val(model, ds, split_file, device, test_fn, args):
    print('============ CROSS-VALIDATION ============')
    split = load_split_file(split_file)
    save_split_file(split, os.path.join(args.output, 'cross_val_split.np.pkl'))
    test_dice = []
    test_assd = []
    test_sdsd = []
    test_hd = []
    test_hd95 = []
    test_missing = []
    train_times_min = []
    for fold, tr_val_fold in enumerate(split):
        print(f"------------ FOLD {fold} ----------------------")
        train_ds, val_ds = ds.split_data_set(tr_val_fold)

        fold_dir = os.path.join(args.output, f'fold{fold}')
        if not args.test_only:
            os.makedirs(fold_dir, exist_ok=True)
            # reset model for the current fold
            model = type(model)(**model.config)
            train(model, train_ds, device, fold_dir, args)

        if not args.train_only:
            mean_dice, _, mean_assd, _, mean_sdsd, _, mean_hd, _, mean_hd95, _, percent_missing = test_fn(
                val_ds, device, fold_dir, args.show)

            if percent_missing is None:
                percent_missing = torch.zeros_like(mean_assd)

            test_dice.append(mean_dice)
            test_assd.append(mean_assd)
            test_sdsd.append(mean_sdsd)
            test_hd.append(mean_hd)
            test_hd95.append(mean_hd95)
            test_missing.append(percent_missing)

            # read the train time file
            with open(os.path.join(fold_dir, 'train_time.csv'), 'r') as time_file:
                reader = csv.reader(time_file)
                for row in reader:
                    if 'train time' in row[0]:
                        continue
                    else:
                        train_times_min.append(eval(row[0]))

    test_dice = torch.stack(test_dice, dim=0)
    test_assd = torch.stack(test_assd, dim=0)
    test_sdsd = torch.stack(test_sdsd, dim=0)
    test_hd = torch.stack(test_hd, dim=0)
    test_hd95 = torch.stack(test_hd95, dim=0)
    test_missing = torch.stack(test_missing, dim=0)

    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    mean_assd = test_assd.mean(0)
    std_assd = test_assd.std(0)

    mean_sdsd = test_sdsd.mean(0)
    std_sdsd = test_sdsd.std(0)

    mean_hd = test_hd.mean(0)
    std_hd = test_hd.std(0)

    mean_hd95 = test_hd95.mean(0)
    std_hd95 = test_hd95.std(0)

    train_times_min = torch.tensor(train_times_min)

    # print out results
    print('\n============ RESULTS ============')
    print(f'Mean dice per class: {mean_dice} +- {std_dice}')

    # output file
    write_results(os.path.join(args.output, 'cv_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, test_missing.mean(0),
                  mean_train_time_in_min=train_times_min.mean(), stddev_train_time_in_min=train_times_min.std())


def run(ds, model, test_fn, args):
    assert not (args.train_only and args.test_only)
    print(args)

    test_fn = get_deterministic_test_fn(test_fn)

    # set the device
    if args.gpu in range(torch.cuda.device_count()):
        device = f'cuda:{args.gpu}'
        print(f"Using device: {device}")
    else:
        device = 'cpu'
        print(f'Requested GPU with index {args.gpu} is not available. Only {torch.cuda.device_count()} GPUs detected.')

    # setup directories
    os.makedirs(args.output, exist_ok=True)

    if not args.test_only:
        if args.split is None:
            args.split = DEFAULT_SPLIT if args.ds == 'data' else DEFAULT_SPLIT_TS

        cross_val(model, ds, args.split, device, test_fn, args)

    else:
        split_file = os.path.join(args.output, 'cross_val_split.np.pkl')
        if args.fold is None:
            # test with all folds
            cross_val(model, ds, split_file, device, test_fn, args)
        else:
            # test with the specified fold from the split file
            folder = os.path.join(args.output, f'fold{args.fold}')
            _, test_ds = ds.split_data_set(load_split_file(split_file)[args.fold])
            test_fn(test_ds, device, folder, args.show)


def get_deterministic_test_fn(test_fn):
    def wrapped_fn(*args, **kwargs):
        torch.random.manual_seed(42)
        return test_fn(*args, **kwargs)

    return wrapped_fn


if __name__ == '__main__':
    parser = get_dgcnn_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # load data
    if args.data in ['fissures', 'lobes']:
        if not args.coords and not args.patch:
            print('No features specified, defaulting to coords as features. '
                  'To specify, provide arguments --coords and/or --patch.')

        if args.ds == 'data':
            point_dir = POINT_DIR
            img_dir = IMG_DIR
        elif args.ds == 'ts':
            point_dir = POINT_DIR_TS
            img_dir = IMG_DIR_TS
        else:
            raise ValueError(f'No dataset named {args.ds}')

        print(f'Using point data from {point_dir}')
        ds = PointDataset(args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                          folder=point_dir, image_folder=img_dir,
                          patch_feat=args.patch,
                          exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes', binary=args.binary)
    else:
        raise ValueError(f'No data set named "{args.data}". Exiting.')

    # setup model
    in_features = ds[0][0].shape[0]
    net = DGCNNSeg(k=args.k, in_features=in_features, num_classes=ds.num_classes,
                   spatial_transformer=args.transformer, dynamic=not args.static)

    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # run the chosen configuration
    run(ds, net, test, args)
