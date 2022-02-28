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

from data import FaustDataset, PointDataset, load_split_file, save_split_file
from dgcnn import DGCNNSeg


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


def train(ds, batch_size, graph_k, device, learn_rate, epochs, show, out_dir):
    val_split = int(len(ds) * 0.2)
    train_ds, valid_ds = random_split(ds, lengths=[len(ds) - val_split, val_split])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    in_features = train_ds[0][0].shape[0]

    # network
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes)
    net.to(device)

    # training setup
    optimizer = torch.optim.Adam(net.parameters(), lr=learn_rate)
    class_frequency = ds.get_label_frequency()
    class_weights = 1 - class_frequency
    class_weights *= ds.num_classes
    print(f'Class weights: {class_weights.tolist()}')
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

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
        for pts, lbls in train_dl:
            pts = pts.to(device)
            lbls = lbls.to(device)

            # forward & backward pass
            optimizer.zero_grad()
            out = net(pts)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            # statistics
            train_loss[epoch] += loss.item() / len(train_dl)
            train_dice[epoch] += batch_dice(out.argmax(1), lbls, ds.num_classes) / len(train_dl)

        # VALIDATION
        net.eval()
        for pts, lbls in valid_dl:
            # TODO: classify all points?
            pts = pts.to(device)
            lbls = lbls.to(device)

            # forward pass
            with torch.no_grad():
                out = net(pts)

            loss = criterion(out, lbls)

            # statistics
            valid_loss[epoch] += loss.item() / len(valid_dl)
            valid_dice[epoch] += batch_dice(out.argmax(1), lbls, ds.num_classes) / len(valid_dl)

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
        'learn_rate': learn_rate,
        'epochs': epochs
    }
    with open(os.path.join(out_dir, 'setup.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(setup_dict.keys()))
        writer.writeheader()
        writer.writerow(setup_dict)


def test(ds, graph_k, device, out_dir):
    in_features = ds[0][0].shape[0]

    # load model
    net = DGCNNSeg(k=graph_k, in_features=in_features, num_classes=ds.num_classes)
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(out_dir, 'model.pth')))
    net.eval()

    test_dice = torch.zeros(ds.num_classes)
    for pts, lbls in ds:
        pts = pts.unsqueeze(0).to(device)
        lbls = lbls.unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(pts)

        test_dice += batch_dice(out.argmax(1), lbls, ds.num_classes)

    test_dice /= len(ds)

    print(f'Test dice per class: {test_dice}')

    # output file
    with open(os.path.join(out_dir, 'test_results.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Class'] + [str(i) for i in range(ds.num_classes)] + ['mean'])
        writer.writerow(['Test Dice'] + [d.item() for d in test_dice] + [test_dice.mean().item()])


def cross_val(ds, split_file, batch_size, graph_k, device, learn_rate, epochs, show, out_dir):
    print('============ CROSS-VALIDATION ============')
    split = load_split_file(split_file)
    save_split_file(split, os.path.join(out_dir, 'cross_val_split.np.pkl'))
    for fold, tr_val_fold in enumerate(split):
        print(f"------------ FOLD {fold} ----------------------")
        train_ds, val_ds = ds.split_data_set(tr_val_fold)

        fold_dir = os.path.join(out_dir, f'fold{fold}')
        os.makedirs(fold_dir, exist_ok=True)
        train(train_ds, batch_size, graph_k, device, learn_rate, epochs, show, fold_dir)

        test(val_ds, graph_k, device, fold_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCNN for lung fissure segmentation.')
    parser.add_argument('--epochs', default=1000, help='max. number of epochs', type=int)
    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--gpu', default=2, help='gpu index to train on', type=int)
    parser.add_argument('--data', help='data set', default='fissures', type=str, choices=['fissures', 'faust'])
    parser.add_argument('--k', default=20, help='number of neighbors for graph computation', type=int)
    parser.add_argument('--pts', default=1024, help='number of points per forward pass', type=int)
    parser.add_argument('--coords', const=True, default=True, help='use point coords as features', nargs='?')
    parser.add_argument('--patch', const=True, default=True, help='use image patch around points as features', nargs='?')
    parser.add_argument('--batch', default=32, help='batch size', type=int)
    parser.add_argument('--output', default='./results', help='output data path', type=str)
    parser.add_argument('--show', const=True, default=False, help='turn on plots (will only be saved by default)', nargs='?')
    parser.add_argument('--exclude_rhf', const=True, default=False, help='exclude the right horizontal fissure from the model', nargs='?')
    parser.add_argument('--split', default=None, type=str, help='cross validation split file')
    args = parser.parse_args()
    print(args)

    # load data
    if args.data == 'fissures':
        point_dir = f'../point_data/feat_{"coords" if args.coords else ""}{"_mind" if args.coords else ""}'
        print(f'Using point data from {point_dir}')
        ds = PointDataset(args.pts, point_dir, exclude_rhf=args.exclude_rhf)
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
        train(ds, args.batch, args.k, device, args.lr, args.epochs, args.show, args.output)
        test(ds, args.k, device, args.output)
    else:
        cross_val(ds, args.split, args.batch, args.k, device, args.lr, args.epochs, args.show, args.output)
