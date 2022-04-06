import argparse
import csv
import os
from copy import deepcopy
from typing import List, Tuple
import open3d as o3d
import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import random_split, DataLoader

from data import FaustDataset, PointDataset, load_split_file, save_split_file, LungData
from dgcnn import DGCNNSeg
from metrics import assd, label_mesh_assd
from data_processing.surface_fitting import pointcloud_surface_fitting, o3d_mesh_to_labelmap
from data_processing.find_lobes import lobes_to_fissures
from utils import kpts_to_world, mask_out_verts_from_mesh, remove_all_but_biggest_component, mask_to_points
from visualization import visualize_point_cloud, visualize_trimesh


def batch_dice(prediction, target, n_labels):
    labels = torch.arange(n_labels)
    dice = torch.zeros(prediction.shape[0], n_labels).to(prediction.device)

    pred_flat = prediction.flatten(start_dim=1)
    targ_flat = target.flatten(start_dim=1)
    for l in labels:
        label_pred = pred_flat == l
        label_target = targ_flat == l
        dice[:, l] = 2 * (label_pred * label_target).sum(-1) / (label_pred.sum(-1) + label_target.sum(-1) + 1e-8)

    return dice.mean(0).cpu()


def train(ds, batch_size, graph_k, transformer, dynamic, use_coords, use_features, device, learn_rate, epochs, show, out_dir):
    print('\nTRAINING MODEL ...\n')

    val_split = int(len(ds) * 0.2)
    train_ds, valid_ds = random_split(ds, lengths=[len(ds) - val_split, val_split])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    in_features = train_ds[0][0].shape[0]

    # network
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes,
                   spatial_transformer=transformer, dynamic=dynamic)
    net.to(device)

    # optimizer and loss
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)  # TODO: weight decay?
    class_frequency = ds.get_label_frequency()
    class_weights = 1 - class_frequency
    class_weights *= ds.num_classes
    print(f'Class weights: {class_weights.tolist()}')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # learnrate scheduling  # TODO: try cosine annealing
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50,
                                                           threshold=1e-4, cooldown=50, verbose=True)

    # statistic logging
    train_loss = torch.zeros(epochs)
    valid_loss = torch.zeros_like(train_loss)
    train_dice = torch.zeros(epochs, ds.num_classes)
    valid_dice = torch.zeros_like(train_dice)
    best_model = None
    best_epoch = 0
    every_n_epochs = epochs
    for epoch in range(epochs):
        # TRAINING
        net.train()
        for pts, feat, lbls in train_dl:
            inputs = []
            if use_coords:
                inputs.append(pts)
            elif use_features:
                inputs.append(feat)

            inputs = torch.cat(inputs, dim=1).to(device)
            lbls = lbls.to(device)

            # forward & backward pass
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            # statistics
            train_loss[epoch] += loss.item() * len(inputs) / len(train_ds)
            train_dice[epoch] += batch_dice(out.argmax(1), lbls, ds.num_classes) * len(inputs) / len(train_ds)

        # VALIDATION
        net.eval()
        for pts, feat, lbls in valid_dl:
            inputs = []
            if use_coords:
                inputs.append(pts)
            elif use_features:
                inputs.append(feat)

            inputs = torch.cat(inputs, dim=1).to(device)
            lbls = lbls.to(device)

            # forward pass
            with torch.no_grad():
                out = net(inputs)

            loss = criterion(out, lbls)

            # statistics
            valid_loss[epoch] += loss.item() / len(valid_dl)
            valid_dice[epoch] += batch_dice(out.argmax(1), lbls, ds.num_classes) / len(valid_dl)

        # update learnrate
        scheduler.step(valid_loss[epoch])

        # status output
        print(f'[{epoch:4}] TRAIN: {train_loss[epoch]:.4f} loss, dice {train_dice[epoch]}, mean {train_dice[epoch, :].mean()}\n'
              f'      VALID: loss {valid_loss[epoch]:.4f}, dice {valid_dice[epoch]}, mean {valid_dice[epoch, :].mean()}')

        # save best snapshot
        if valid_dice[epoch, :].mean() >= valid_dice[best_epoch, :].mean():
            best_model = deepcopy(net.state_dict())
            best_epoch = epoch

        # visualization
        if show and not (epoch + 1) % every_n_epochs:
            visualize_point_cloud(pts[0].transpose(0, 1), out.argmax(1)[0])

    # training plot
    plt.figure()
    plt.title(f'Training Progression. Best model from epoch {best_epoch} ({valid_dice[:, :].mean(1)[best_epoch]:.4f} val. dice).')
    plt.plot(train_loss, c='b', label='train loss')
    plt.plot(valid_loss, c='r', label='validation loss')
    plt.plot(valid_dice[:, :].mean(1), c='g', label='mean validation dice')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"training_progression.png"))
    if show:
        plt.show()
    else:
        plt.close()

    # save best model
    model_path = os.path.join(out_dir, 'model.pth')
    print(f'Saving best model from epoch {best_epoch} to path "{model_path}"')
    torch.save(best_model, model_path)

    # save setup
    setup_dict = {
        'data': ds.folder,
        'n_points': ds.sample_points,
        'batch_size': batch_size,
        'graph_k': graph_k,
        'use_coords': use_coords,
        'use_features': use_features,
        'learn_rate': learn_rate,
        'epochs': epochs,
        'exclude_rhf': ds.exclude_rhf,
        'lobes': ds.lobes,
        'dgcnn_input_transformer': transformer,
        'dynamic': dynamic
    }
    with open(os.path.join(out_dir, 'setup.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(setup_dict.keys()))
        writer.writeheader()
        writer.writerow(setup_dict)


def compute_mesh_metrics(meshes_predict: List[List[o3d.geometry.TriangleMesh]],
                         meshes_target: List[List[o3d.geometry.TriangleMesh]],
                         ids: List[Tuple[str, str]] = None,
                         show: bool = False, spacings=None):
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
        if show:
            if ids is not None:
                case, sequence = ids[i]
                title_prefix = f'{case}_{sequence} '
            else:
                title_prefix = f'sample {i}'

            if pred_points:
                all_points = torch.concat(pred_points, dim=0)
                lbls = torch.concat([torch.ones(len(pts), dtype=torch.long) + p for p, pts in enumerate(pred_points)])
                visualize_point_cloud(all_points, labels=lbls, title=title_prefix + 'points prediction')
            else:
                visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in all_parts_predictions],
                                  triangles_list=[np.asarray(m.triangles) for m in all_parts_predictions],
                                  title=title_prefix + 'surface prediction')

            visualize_trimesh(vertices_list=[np.asarray(m.vertices) for m in all_parts_targets],
                              triangles_list=[np.asarray(m.triangles) for m in all_parts_targets],
                              title=title_prefix + 'surface target')

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


def test(ds, graph_k, transformer, dynamic, use_coords, use_features, device, out_dir, show):
    print('\nTESTING MODEL ...\n')

    img_ds = LungData('../data/')
    in_features = ds[0][0].shape[0]

    # load model
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes,
                   spatial_transformer=transformer, dynamic=dynamic)
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(out_dir, 'model.pth'), map_location=device))
    net.eval()

    # directory for output predictions
    pred_dir = os.path.join(out_dir, 'test_predictions')
    mesh_dir = os.path.join(pred_dir, 'meshes')
    label_dir = os.path.join(pred_dir, 'labelmaps')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # compute all predictions
    all_pred_meshes = []
    all_targ_meshes = []
    ids = []
    test_dice = torch.zeros(len(ds), ds.num_classes)
    for i in range(len(ds)):
        pts, feat, lbls = ds.get_full_pointcloud(i)
        inputs = []
        if use_coords:
            inputs.append(pts)
        elif use_features:
            inputs.append(feat)

        inputs = torch.cat(inputs, dim=0).unsqueeze(0).to(device)
        lbls = lbls.unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(inputs)

        # compute point dice score
        labels_pred = out.argmax(1)
        test_dice[i] += batch_dice(labels_pred, lbls, ds.num_classes)

        # convert points back to world coordinates
        case, sequence = ds.ids[i]
        img_index = img_ds.get_index(case, sequence)
        image = img_ds.get_image(img_index)
        spacing = torch.tensor(image.GetSpacing(), device=device)
        shape = torch.tensor(image.GetSize()[::-1], device=device) * spacing.flip(0)
        pts = kpts_to_world(pts.to(device).transpose(0, 1), shape)  # points in millimeters

        # visualize point clouds
        if show:
            visualize_point_cloud(pts, labels_pred.squeeze(), title=f'{case}_{sequence} point cloud prediction')
            visualize_point_cloud(pts, lbls.squeeze(), title=f'{case}_{sequence} point cloud target')

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

    mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics(all_pred_meshes, all_targ_meshes, ids=ids, show=show)

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


def cross_val(ds, split_file, batch_size, graph_k, transformer, dynamic, use_coords, use_features, device, learn_rate, epochs, show, out_dir, test_only=False):
    print('============ CROSS-VALIDATION ============')
    split = load_split_file(split_file)
    save_split_file(split, os.path.join(out_dir, 'cross_val_split.np.pkl'))
    test_dice = torch.zeros(len(split), ds.num_classes)
    test_assd = torch.zeros(len(split), ds.num_classes-1)
    test_sdsd = torch.zeros_like(test_assd)
    test_hd = torch.zeros_like(test_assd)
    test_hd95 = torch.zeros_like(test_assd)
    for fold, tr_val_fold in enumerate(split):
        print(f"------------ FOLD {fold} ----------------------")
        train_ds, val_ds = ds.split_data_set(tr_val_fold)

        fold_dir = os.path.join(out_dir, f'fold{fold}')
        if not test_only:
            os.makedirs(fold_dir, exist_ok=True)
            train(train_ds, batch_size, graph_k, transformer, dynamic, use_coords, use_features, device, learn_rate, epochs, show, fold_dir)

        mean_dice, _, mean_assd, _, mean_sdsd, _, mean_hd, _, mean_hd95, _ = test(val_ds, graph_k, transformer, dynamic,
                                                                                  use_coords, use_features, device, fold_dir, show)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCNN for lung fissure segmentation.')
    parser.add_argument('--epochs', default=1000, help='max. number of epochs', type=int)
    parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--gpu', default=2, help='gpu index to train on', type=int)
    parser.add_argument('--data', help='data set', default='fissures', type=str, choices=['fissures', 'faust', 'lobes'])
    parser.add_argument('--k', default=20, help='number of neighbors for graph computation', type=int)
    parser.add_argument('--pts', default=1024, help='number of points per forward pass', type=int)
    parser.add_argument('--coords', const=True, default=False, help='use point coords as features', nargs='?')
    parser.add_argument('--patch', const=True, default=False, help='use image patch around points as features', nargs='?')
    parser.add_argument('--batch', default=32, help='batch size', type=int)
    parser.add_argument('--output', default='./results', help='output data path', type=str)
    parser.add_argument('--show', const=True, default=False, help='turn on plots (will only be saved by default)', nargs='?')
    parser.add_argument('--exclude_rhf', const=True, default=False, help='exclude the right horizontal fissure from the model', nargs='?')
    parser.add_argument('--split', default=None, type=str, help='cross validation split file')
    parser.add_argument('--transformer', const=True, default=False, help='use spatial transformer module in DGCNN', nargs='?')
    parser.add_argument('--static', const=True, default=False, help='do not use dynamic graph computation in DGCNN', nargs='?')
    parser.add_argument('--test_only', const=True, default=False, help='do not train model', nargs='?')
    parser.add_argument('--fold', default=None, help='specify if only one fold should be evaluated (needs to be in range of folds in the split file)', type=int)
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
        ds = PointDataset(args.pts, folder=point_dir, patch_feat=features, exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes')
    elif args.data == 'faust':
        print(f'Using FAUST data set')
        ds = FaustDataset(args.pts)
    else:
        print(f'No data set named "{args.data}". Exiting.')
        exit(1)

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
            train(ds, args.batch, args.k, args.transformer, not args.static, args.coords, args.patch, device, args.lr, args.epochs, args.show, args.output)
            test(ds, args.k, args.transformer, not args.static, args.coords, args.patch, device, args.output, args.show)
        else:
            cross_val(ds, args.split, args.batch, args.k, args.transformer, not args.static, args.coords, args.patch, device, args.lr, args.epochs, args.show, args.output)

    else:
        split_file = os.path.join(args.output, 'cross_val_split.np.pkl')
        if args.fold is None:
            # test with all folds
            cross_val(ds, split_file, args.batch, args.k, args.transformer, not args.static, args.coords, args.patch, device, args.lr, args.epochs, args.show, args.output, test_only=True)
        else:
            # test with the specified fold from the split file
            test_folds = [args.fold]
            folder = os.path.join(args.output, f'fold{args.fold}')
            _, test_ds = ds.split_data_set(load_split_file(split_file)[args.fold])
            test(test_ds, args.k, args.transformer, not args.static, args.coords, args.patch, device, folder, args.show)
