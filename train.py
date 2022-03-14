import csv
from copy import deepcopy

import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
import torch
import argparse
from torch.utils.data import random_split, DataLoader

from data import FaustDataset, PointDataset, load_split_file, save_split_file, LungData
from dgcnn import DGCNNSeg
from metrics import assd, ssd
from surface_fitting import pointcloud_to_mesh
from utils import kpts_to_world


def batch_dice(prediction, target, n_labels):
    labels = torch.arange(n_labels)
    dice = torch.zeros(prediction.shape[0], n_labels).to(prediction.device)
    for l in labels:
        label_pred = prediction == l
        label_target = target == l
        dice[:, l] = 2 * (label_pred * label_target).sum(-1) / (label_pred.sum(-1) + label_target.sum(-1) + 1e-8)

    return dice.mean(0).cpu()


def visualize_point_cloud(points, labels):
    """

    :param points: point cloud with N points, shape: (3 x N)
    :param labels: label for each point, shape (N)
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = points.cpu()
    ax.scatter(points[0], points[1], points[2], c=labels.cpu(), cmap='tab20', marker='.')
    ax.view_init(elev=100., azim=-60.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def train(ds, batch_size, graph_k, transformer, use_coords, use_features, device, learn_rate, epochs, show, out_dir):
    print('\nTRAINING MODEL ...\n')

    val_split = int(len(ds) * 0.2)
    train_ds, valid_ds = random_split(ds, lengths=[len(ds) - val_split, val_split])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    in_features = train_ds[0][0].shape[0]

    # network
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes,
                   spatial_transformer=transformer)
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
            visualize_point_cloud(pts[0], out.argmax(1)[0])

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
        'dgcnn_input_transformer': transformer
    }
    with open(os.path.join(out_dir, 'setup.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(setup_dict.keys()))
        writer.writeheader()
        writer.writerow(setup_dict)


def test(ds, graph_k, transformer, use_coords, use_features, device, out_dir):
    print('\nTESTING MODEL ...\n')

    img_ds = LungData('../data/')
    in_features = ds[0][0].shape[0]

    # load model
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes, spatial_transformer=transformer)
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(out_dir, 'model.pth')))
    net.eval()

    # metrics
    test_dice = torch.zeros(len(ds), ds.num_classes)
    avg_surface_dist = torch.zeros(len(ds), 2 if ds.exclude_rhf else 3)  # hardcoded number of fissures
    std_surface_dist = torch.zeros_like(avg_surface_dist)
    hd_surface_dist = torch.zeros_like(avg_surface_dist)
    hd95_surface_dist = torch.zeros_like(avg_surface_dist)
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

        labels_pred = out.argmax(1)
        test_dice[i] += batch_dice(labels_pred, lbls, ds.num_classes)

        # convert points back to world coordinates
        case, sequence = ds.ids[i]
        image = img_ds.get_image(next(j for j, fn in enumerate(img_ds.images) if f'{case}_img_{sequence}' in fn))
        spacing = torch.tensor(image.GetSpacing()[::-1], device=device)
        shape = torch.tensor(image.GetSize()[::-1], device=device) * spacing
        pts = kpts_to_world(pts.to(device).transpose(0, 1), shape)  # points in millimeters

        if not ds.lobes:
            # mesh fitting for each fissure
            for j, f in enumerate(labels_pred.unique()[1:]):  # excluding background
                mesh_predict = pointcloud_to_mesh(pts[labels_pred.squeeze() == f].cpu())
                mesh_target = pointcloud_to_mesh(pts[lbls.squeeze() == f].cpu())
                asd, sdsd, hdsd, hd95sd = ssd(mesh_predict, mesh_target)
                avg_surface_dist[i, j] += asd
                std_surface_dist[i, j] += sdsd
                hd_surface_dist[i, j] += hdsd
                hd95_surface_dist[i, j] += hd95sd

        else:
            # TODO: compute fissures from lobes
            pass

    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    mean_assd = avg_surface_dist.mean(0)
    std_assd = avg_surface_dist.std(0)

    mean_sdsd = std_surface_dist.mean(0)
    std_sdsd = std_surface_dist.std(0)

    mean_hd = hd_surface_dist.mean(0)
    std_hd = hd_surface_dist.std(0)

    mean_hd95 = hd95_surface_dist.mean(0)
    std_hd95 = hd95_surface_dist.std(0)

    print(f'Test dice per class: {mean_dice} +- {std_dice}')
    print(f'ASSD per fissure: {mean_assd} +- {std_assd}')

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


def write_results(filepath, mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95):
    with open(filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Class'] + [str(i) for i in range(ds.num_classes)] + ['mean'])
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


def cross_val(ds, split_file, batch_size, graph_k, transformer, use_coords, use_features, device, learn_rate, epochs, show, out_dir):
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
        os.makedirs(fold_dir, exist_ok=True)
        train(train_ds, batch_size, graph_k, transformer, use_coords, use_features, device, learn_rate, epochs, show, fold_dir)

        mean_dice, _, mean_assd, _, mean_sdsd, _, mean_hd, _, mean_hd95, _ = test(val_ds, graph_k, transformer, use_coords, use_features, device, fold_dir)

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
    parser.add_argument('--test_only', const=True, default=False, help='do not train model', nargs='?')
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
    else:
        device = 'cpu'
        print(f'Requested GPU with index {args.gpu} is not available. Only {torch.cuda.device_count()} GPUs detected.')

    # setup directories
    os.makedirs(args.output, exist_ok=True)

    if args.split is None:
        if not args.test_only:
            train(ds, args.batch, args.k, args.transformer, args.coords, args.patch, device, args.lr, args.epochs, args.show, args.output)
        test(ds, args.k, args.transformer, args.coords, args.patch, device, args.output)
    else:
        cross_val(ds, args.split, args.batch, args.k, args.transformer, args.coords, args.patch, device, args.lr, args.epochs, args.show, args.output)
