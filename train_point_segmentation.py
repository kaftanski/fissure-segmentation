import csv
import os
import time
from typing import List, Tuple

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch

from cli.cli_args import get_point_segmentation_parser
from cli.cli_utils import load_args_for_testing, store_args, load_args
from constants import POINT_DIR_COPD, POINT_DIR_TS, DEFAULT_SPLIT_TS, IMG_DIR_COPD, IMG_DIR_TS_PREPROC
from data_processing.datasets import PointDataset, load_split_file, save_split_file, LungData
from data_processing.find_lobes import lobes_to_fissures
from data_processing.surface_fitting import pointcloud_surface_fitting, o3d_mesh_to_labelmap
from evaluation.metrics import assd, label_mesh_assd, batch_dice
from losses.access_losses import get_loss_fn
from model_training import model_trainer
from models.access_models import get_point_seg_model_class_from_args
from utils.detached_run import maybe_run_detached_cli
from utils.fissure_utils import binary_to_fissure_segmentation
from utils.general_utils import kpts_to_world, mask_out_verts_from_mesh, remove_all_but_biggest_component, \
    mask_to_points, \
    points_to_label_map, create_o3d_mesh, nanstd, get_device, no_print
from utils.model_utils import param_and_op_count
from utils.visualization import visualize_point_cloud, visualize_o3d_mesh


def train(model, ds, device, out_dir, args):
    # set up loss function
    class_weights = ds.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = get_loss_fn(args.loss, class_weights, args.loss_weights)

    # run training
    trainer = model_trainer.ModelTrainer(model, ds, criterion, out_dir, device, args)
    trainer.run(initial_epoch=0)


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
        pred_points = []  # if predictions is voxel-based, this will store the sampled point-cloud
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


def test(ds: PointDataset, device, out_dir, show, args):
    print('\nTESTING MODEL ...\n')

    img_ds = LungData(ds.image_folder)

    model_class = get_point_seg_model_class_from_args(args)

    net = model_class.load(os.path.join(out_dir, 'model.pth'), device=device)
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
        for j in range(net.num_classes-1):  # excluding background
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
    write_results(os.path.join(out_dir, f'test_results{"_copd" if args.copd else ""}.csv'),
                  mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing


def speed_test(ds: PointDataset, device, out_dir):
    args = load_args(os.path.join(out_dir))
    model_class = get_point_seg_model_class_from_args(args)

    net = model_class.load(os.path.join(out_dir, 'fold0', 'model.pth'), device=device)
    net.to(device)
    net.eval()

    img_ds = LungData(ds.image_folder)

    # prepare for measurement of inference times
    torch.manual_seed(42)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_inference_times = []
    all_post_proc_times = []
    points_per_fissure = []
    for i in range(len(ds)):
        inputs, lbls = ds.get_full_pointcloud(i)
        inputs = inputs.unsqueeze(0).to(device)

        # convert points back to world coordinates
        pts = inputs[0, :3].cpu()  # coords are the first 3 features
        case, sequence = ds.ids[i]
        spacing = torch.tensor(ds.spacings[i])
        shape = torch.tensor(ds.img_sizes_index[i][::-1]) * spacing.flip(0)
        pts = kpts_to_world(pts.transpose(0, 1), shape)  # points in millimeters

        # load mask for post-processing
        img_index = img_ds.get_index(case, sequence)
        mask_img = img_ds.get_lung_mask(img_index)
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask_img).astype(bool))

        with no_print():
            # measure inference time
            with torch.no_grad():
                torch.cuda.synchronize(device)  # don't forget to synchronize the correct device (cuda:0 if not specified)
                starter.record(torch.cuda.current_stream(device))  # choose the stream on correct device
                out = net.predict_full_pointcloud(inputs, ds.sample_points, n_runs_min=50)

            labels_pred = out.argmax(1)
            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

            labels_pred = labels_pred.cpu()
            # measure post-processing time
            start_post_proc = time.time()

            # mesh fitting for each label
            for j in range(net.num_classes - 1):  # excluding background
                label = j + 1
                try:
                    depth = 6
                    mesh_predict = pointcloud_surface_fitting(pts[labels_pred.squeeze() == label].numpy().astype(float),
                        crop_to_bbox=True, depth=depth)

                except ValueError as e:
                    # no points have been predicted to be in this class
                    continue

                # post-process surfaces
                mask_out_verts_from_mesh(mesh_predict, mask_tensor, spacing)  # apply lung mask
                right = label > 1  # right fissure(s) are label 2 and 3
                remove_all_but_biggest_component(mesh_predict, right=right,
                                                 center_x=shape[2] / 2)  # only keep the biggest connected component

            all_post_proc_times.append(time.time() - start_post_proc)

        unique_lbls, n_points = labels_pred.cpu().unique(return_counts=True)
        accum = torch.zeros(ds.num_classes)
        accum[unique_lbls] += n_points
        points_per_fissure.append(accum[1:])
        print(f'[inference + post-proc time] {all_inference_times[-1]:.4f} + {all_post_proc_times[-1]:.4f} s')

    write_speed_results(out_dir, all_inference_times, all_post_proc_times, points_per_fissure)


def write_speed_results(out_dir, all_inference_times, all_post_proc_times=None, points_per_fissure=None):
    all_inference_times = torch.tensor(all_inference_times)

    if all_post_proc_times is not None:
        all_post_proc_times = torch.tensor(all_post_proc_times)
        total_times = all_inference_times + all_post_proc_times
    else:
        all_post_proc_times = torch.zeros_like(all_inference_times)
        total_times = all_inference_times

    if points_per_fissure is not None:
        points_per_fissure = torch.stack(points_per_fissure).float()

    out_file = os.path.join(out_dir, 'inference_time_node2.csv')
    with open(out_file, 'w') as time_file:
        writer = csv.writer(time_file)
        writer.writerow(['Inference', 'Inference_std', 'Post-Processing', 'Post-Processing_std', 'Total', 'Total_std']
                        + (['Points_per_Fissure', 'Points_per_Fissure_std'] if points_per_fissure is not None else []))
        writer.writerow([all_inference_times.mean().item(), all_inference_times.std().item(),
                         all_post_proc_times.mean().item(), all_post_proc_times.std().item(),
                         total_times.mean().item(), total_times.std().item()]
                        + ([points_per_fissure.mean().item(), points_per_fissure.std(0).mean().item()] if points_per_fissure is not None else []))


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
        train_ds, val_ds = ds.split_data_set(tr_val_fold, fold_nr=fold)

        fold_dir = os.path.join(args.output, f'fold{fold}')
        if not args.test_only:
            os.makedirs(fold_dir, exist_ok=True)
            # reset model for the current fold
            model = type(model)(**model.config)
            train(model, train_ds, device, fold_dir, args)

        if not args.train_only:
            mean_dice, _, mean_assd, _, mean_sdsd, _, mean_hd, _, mean_hd95, _, percent_missing = test_fn(
                val_ds, device, fold_dir, args.show, args)

            if percent_missing is None:
                percent_missing = torch.zeros_like(mean_assd)

            test_dice.append(mean_dice)
            test_assd.append(mean_assd)
            test_sdsd.append(mean_sdsd)
            test_hd.append(mean_hd)
            test_hd95.append(mean_hd95)
            test_missing.append(percent_missing)

            # read the train time file
            try:
                with open(os.path.join(fold_dir, 'train_time.csv'), 'r') as time_file:
                    reader = csv.reader(time_file)
                    for row in reader:
                        if 'train time' in row[0]:
                            continue
                        else:
                            train_times_min.append(eval(row[0]))
            except FileNotFoundError:
                train_times_min.append(0.0)

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
    write_results(os.path.join(args.output, f'cv_results{"_copd" if args.copd else ""}.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, test_missing.mean(0),
                  mean_train_time_in_min=train_times_min.mean(), stddev_train_time_in_min=train_times_min.std())


def run(ds, model, test_fn, args):
    assert not (args.train_only and args.test_only)
    if 'model' in args.__dict__ and args.model == 'PointTransformer' and not args.coords:
        raise NotImplemented('Coords have to be chosen as features if training PointTransformer.')

    print(args)

    test_fn = get_deterministic_test_fn(test_fn)

    # set the device
    device = get_device(args.gpu)

    # setup directories
    os.makedirs(args.output, exist_ok=True)

    if not args.test_only:
        if args.split is None:
            args.split = DEFAULT_SPLIT_TS

        cross_val(model, ds, args.split, device, test_fn, args)

    else:
        split_file = os.path.join(args.output, 'cross_val_split.np.pkl')
        if args.fold is None:
            # test with all folds
            cross_val(model, ds, split_file, device, test_fn, args)
        else:
            # test with the specified fold from the split file
            folder = os.path.join(args.output, f'fold{args.fold}')
            _, test_ds = ds.split_data_set(load_split_file(split_file)[args.fold], fold_nr=args.fold)
            test_fn(test_ds, device, folder, args.show, args)


def get_deterministic_test_fn(test_fn):
    def wrapped_fn(*args, **kwargs):
        torch.random.manual_seed(42)
        return test_fn(*args, **kwargs)

    return wrapped_fn


if __name__ == '__main__':
    parser = get_point_segmentation_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only or args.speed or args.copd:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # load data
    if args.data in ['fissures', 'lobes']:
        if not args.coords and not args.patch:
            print('No features specified, defaulting to coords as features. '
                  'To specify, provide arguments --coords and/or --patch.')

        if args.copd:
            point_dir = POINT_DIR_COPD
            img_dir = IMG_DIR_COPD
        else:
            point_dir = POINT_DIR_TS
            img_dir = IMG_DIR_TS_PREPROC

        if args.copd:
            print('Validating with COPD dataset')
            args.test_only = True
            args.speed = False
        else:
            print(f'Using point data from {point_dir}')
        ds = PointDataset(args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                          folder=point_dir, image_folder=img_dir,
                          patch_feat=args.patch,
                          exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes', binary=args.binary, copd=args.copd)
    else:
        raise ValueError(f'No data set named "{args.data}". Exiting.')

    # setup folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # run the speed test
    if args.speed:
        speed_test(ds, get_device(args.gpu), args.output)
        exit(0)

    # setup model
    in_features = ds[0][0].shape[0]
    model_class = get_point_seg_model_class_from_args(args)
    net = model_class(in_features=in_features, num_classes=ds.num_classes, k=args.k,
                      spatial_transformer=args.transformer, dynamic=not args.static).to(get_device(args.gpu))

    param_and_op_count(net, (1, *ds[0][0].shape), out_dir=args.output)

    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # run the chosen configuration
    run(ds, net, test, args)
