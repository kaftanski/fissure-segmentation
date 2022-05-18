import csv
import os
from typing import List, Tuple

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from torch import nn

import model_trainer
from cli.cl_args import get_dgcnn_train_parser
from data import PointDataset, load_split_file, save_split_file, LungData
from data_processing.find_lobes import lobes_to_fissures
from data_processing.surface_fitting import pointcloud_surface_fitting, o3d_mesh_to_labelmap
from metrics import assd, label_mesh_assd, batch_dice
from models.dgcnn import DGCNNSeg
from utils import kpts_to_world, mask_out_verts_from_mesh, remove_all_but_biggest_component, mask_to_points
from visualization import visualize_point_cloud, visualize_trimesh


def train(model, ds, batch_size, device, learn_rate, epochs, show, out_dir):
    # loss function
    class_weights = ds.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # run training
    trainer = model_trainer.ModelTrainer(model, ds, criterion, learn_rate, batch_size, device, epochs, out_dir, show)
    trainer.run(initial_epoch=0)


def compute_mesh_metrics(meshes_predict: List[List[o3d.geometry.TriangleMesh]],
                         meshes_target: List[List[o3d.geometry.TriangleMesh]],
                         ids: List[Tuple[str, str]] = None,
                         show: bool = False, spacings=None, plot_folder=None):
    # metrics
    # test_dice = torch.zeros(len(meshes_predict), len(meshes_predict[0]))
    avg_surface_dist = torch.zeros(len(meshes_predict), len(meshes_predict[0]))
    std_surface_dist = torch.zeros_like(avg_surface_dist)
    hd_surface_dist = torch.zeros_like(avg_surface_dist)
    hd95_surface_dist = torch.zeros_like(avg_surface_dist)

    for i, (all_parts_predictions, all_parts_targets) in enumerate(zip(meshes_predict, meshes_target)):
        pred_points = []
        for j, (pred_part, targ_part) in enumerate(zip(all_parts_predictions, all_parts_targets)):
            if isinstance(pred_part, torch.Tensor):
                asd, sdsd, hdsd, hd95sd, points = label_mesh_assd(pred_part, targ_part, spacing=spacings[i])
                pred_points.append(points)
            else:
                asd, sdsd, hdsd, hd95sd = assd(pred_part, targ_part)

            avg_surface_dist[i, j] += asd
            std_surface_dist[i, j] += sdsd
            hd_surface_dist[i, j] += hdsd
            hd95_surface_dist[i, j] += hd95sd

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
                visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in all_parts_predictions],
                                  triangles_list=[np.asarray(m.triangles) for m in all_parts_predictions],
                                  title=title_prefix + ' surface prediction', show=show,
                                  savepath=os.path.join(plot_folder, f'{title_prefix}_mesh_pred.png'))

            visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in all_parts_targets],
                              triangles_list=[np.asarray(m.triangles) for m in all_parts_targets],
                              title=title_prefix + ' surface target', show=show,
                              savepath=os.path.join(plot_folder, f'{title_prefix}_mesh_targ.png'))

    # compute average metrics
    mean_assd = avg_surface_dist.mean(0)
    std_assd = avg_surface_dist.std(0)

    mean_sdsd = std_surface_dist.mean(0)
    std_sdsd = std_surface_dist.std(0)

    mean_hd = hd_surface_dist.mean(0)
    std_hd = hd_surface_dist.std(0)

    mean_hd95 = hd95_surface_dist.mean(0)
    std_hd95 = hd95_surface_dist.std(0)

    return mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


def test(ds, device, out_dir, show):
    print('\nTESTING MODEL ...\n')

    img_ds = LungData('../data/')

    net = DGCNNSeg.load(os.path.join(out_dir, 'model.pth'), device=device)
    net.to(device)
    net.eval()

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
            out = net(inputs)

        # compute point dice score
        labels_pred = out.argmax(1)
        test_dice[i] += batch_dice(labels_pred, lbls, ds.num_classes)

        # convert points back to world coordinates
        pts = ds.get_coords(i)
        case, sequence = ds.ids[i]
        img_index = img_ds.get_index(case, sequence)
        image = img_ds.get_image(img_index)
        spacing = torch.tensor(image.GetSpacing(), device=device)
        shape = torch.tensor(image.GetSize()[::-1], device=device) * spacing.flip(0)
        pts = kpts_to_world(pts.to(device).transpose(0, 1), shape)  # points in millimeters

        # visualize point clouds
        visualize_point_cloud(pts, labels_pred.squeeze(), title=f'{case}_{sequence} point cloud prediction', show=show,
                              savepath=os.path.join(plot_dir, f'{case}_{sequence}_point_cloud_pred.png'))
        visualize_point_cloud(pts, lbls.squeeze(), title=f'{case}_{sequence} point cloud target', show=show,
                              savepath=os.path.join(plot_dir, f'{case}_{sequence}_point_cloud_targ.png'))

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
            sparse_lobes_img.CopyInformation(image)
            fissure_pred_img, lobes_pred_img = lobes_to_fissures(sparse_lobes_img, mask_img, device=device)

            # write out intermediate results
            sitk.WriteImage(sparse_lobes_img, os.path.join(label_dir, f'{case}_lobes_pointcloud_{sequence}.nii.gz'))
            sitk.WriteImage(fissure_pred_img, os.path.join(label_dir, f'{case}_fissures_from_lobes_{sequence}.nii.gz'))
            sitk.WriteImage(lobes_pred_img, os.path.join(label_dir, f'{case}_lobes_pred_{sequence}.nii.gz'))

        else:
            meshes_target = img_ds.get_fissure_meshes(img_index)[:2 if ds.exclude_rhf else 3]

        # mesh fitting for each label
        meshes_predict = []
        for j in range(len(meshes_target)):  # excluding background
            label = j+1

            if not ds.lobes:
                # using poisson reconstruction with octree-depth 3 because of sparse point cloud
                mesh_predict = pointcloud_surface_fitting(pts[labels_pred.squeeze() == label].cpu(), crop_to_bbox=True,
                                                          depth=3)
            else:
                # extract the fissure points from labelmap
                fissure_pred_pts = mask_to_points(torch.from_numpy(sitk.GetArrayFromImage(fissure_pred_img).astype(int)) == label,
                                                  spacing=fissure_pred_img.GetSpacing())

                # fit surface to the points with depth 6, because points are dense
                mesh_predict = pointcloud_surface_fitting(fissure_pred_pts, crop_to_bbox=True, depth=6)

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
        labelmap_predict = o3d_mesh_to_labelmap(meshes_predict, shape=image.GetSize()[::-1], spacing=image.GetSpacing())
        label_image_predict = sitk.GetImageFromArray(labelmap_predict.numpy().astype(np.uint8))
        label_image_predict.CopyInformation(image)
        sitk.WriteImage(label_image_predict, os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))

        # target labelmap
        labelmap_target = o3d_mesh_to_labelmap(meshes_target, shape=image.GetSize()[::-1], spacing=image.GetSpacing())
        label_image_target = sitk.GetImageFromArray(labelmap_target.numpy().astype(np.uint8))
        label_image_target.CopyInformation(image)
        sitk.WriteImage(label_image_target, os.path.join(label_dir, f'{case}_fissures_targ_{sequence}.nii.gz'))

        # remember meshes for evaluation
        all_pred_meshes.append(meshes_predict)
        all_targ_meshes.append(meshes_target)
        ids.append((case, sequence))

    # compute average metrics
    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics(
        all_pred_meshes, all_targ_meshes, ids=ids, show=show, plot_folder=plot_dir)

    print(f'Test dice per class: {mean_dice} +- {std_dice}')
    print(f'ASSD per fissure: {mean_assd} +- {std_assd}')

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


def write_results(filepath, mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95):
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


def cross_val(model, ds, split_file, batch_size, device, learn_rate, epochs, show, out_dir, test_fn, test_only=False):
    print('============ CROSS-VALIDATION ============')
    split = load_split_file(split_file)
    save_split_file(split, os.path.join(out_dir, 'cross_val_split.np.pkl'))
    test_dice = torch.zeros(len(split), ds.num_classes)
    test_assd = torch.zeros(len(split), ds.num_classes-1 if not ds.lobes else int(ds.num_classes / 2))
    test_sdsd = torch.zeros_like(test_assd)
    test_hd = torch.zeros_like(test_assd)
    test_hd95 = torch.zeros_like(test_assd)
    for fold, tr_val_fold in enumerate(split):
        print(f"------------ FOLD {fold} ----------------------")
        train_ds, val_ds = ds.split_data_set(tr_val_fold)

        fold_dir = os.path.join(out_dir, f'fold{fold}')
        if not test_only:
            os.makedirs(fold_dir, exist_ok=True)
            train(model, train_ds, batch_size, device, learn_rate, epochs, show, fold_dir)

        mean_dice, _, mean_assd, _, mean_sdsd, _, mean_hd, _, mean_hd95, _ = test_fn(val_ds, device, fold_dir, show)

        test_dice[fold] += mean_dice
        test_assd[fold] += mean_assd
        test_sdsd[fold] += mean_sdsd
        test_hd[fold] += mean_hd
        test_hd95[fold] += mean_hd95

        # TODO: compute confusion matrix

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

    # print out results
    print('\n============ RESULTS ============')
    print(f'Mean dice per class: {mean_dice} +- {std_dice}')

    # output file
    write_results(os.path.join(out_dir, 'cv_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)


def run(ds, model, test_fn, args):
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
            train(model, ds, args.batch, device, args.lr, args.epochs, args.show, args.output)
            test_fn(ds, device, args.output, args.show)
        else:
            cross_val(model, ds, args.split, args.batch, device, args.lr, args.epochs, args.show, args.output, test_fn)

    else:
        split_file = os.path.join(args.output, 'cross_val_split.np.pkl')
        if args.fold is None:
            # test with all folds
            cross_val(model, ds, split_file, args.batch, device, args.lr, args.epochs, args.show, args.output, test_fn,
                      test_only=True)
        else:
            # test with the specified fold from the split file
            folder = os.path.join(args.output, f'fold{args.fold}')
            _, test_ds = ds.split_data_set(load_split_file(split_file)[args.fold])
            test_fn(test_ds, device, folder, args.show)


if __name__ == '__main__':
    parser = get_dgcnn_train_parser()
    args = parser.parse_args()
    print(args)

    # load data
    if args.data in ['fissures', 'lobes']:
        if not args.coords and not args.patch:
            print('No features specified, defaulting to coords as features. '
                  'To specify, provide arguments --coords and/or --patch.')
        point_dir = '../point_data/'
        print(f'Using point data from {point_dir}')
        features = 'mind' if args.patch else None
        ds = PointDataset(args.pts, kp_mode=args.kp_mode, use_coords=args.coords, folder=point_dir, patch_feat=features,
                          exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes')
    else:
        raise ValueError(f'No data set named "{args.data}". Exiting.')

    # setup model
    in_features = ds[0][0].shape[0]
    net = DGCNNSeg(k=args.k, in_features=in_features, num_classes=ds.num_classes,
                   spatial_transformer=args.transformer, dynamic=not args.static)

    # run the chosen configuration
    run(ds, net, test, args)

    # save setup
    setup_dict = {
        'data': ds.folder,
        'n_points': ds.sample_points,
        'batch_size': args.batch,
        'graph_k': args.k,
        'use_coords': args.coords,
        'use_features': args.patch,
        'learn_rate': args.lr,
        'epochs': args.epochs,
        'exclude_rhf': ds.exclude_rhf,
        'lobes': ds.lobes,
        'dgcnn_input_transformer': args.transformer,
        'dynamic': not args.static
    }
    with open(os.path.join(args.output, 'setup.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(setup_dict.keys()))
        writer.writeheader()
        writer.writerow(setup_dict)
