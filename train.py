import csv
from copy import deepcopy

import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
import torch
import argparse
from torch.utils.data import random_split, DataLoader

from data import FaustDataset
from point_features import PointDataset
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


def train(args):
    # load data
    if args.data == 'fissures':
        ds = PointDataset(args.pts)
    elif args.data == 'faust':
        ds = FaustDataset(args.pts)
    else:
        print(f'No data set named "{args.data}". Exiting.')
        return

    val_split = int(len(ds) * 0.2)
    train_ds, valid_ds = random_split(ds, lengths=[len(ds) - val_split, val_split])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)

    dim = train_ds[0][0].shape[0]

    # network
    in_features = 0
    if args.patch:
        print('Using image patch around points as features')
        print('NOT IMPLEMENTED YET')
        radius = 3
        in_features += radius ** dim

    if args.coords:
        print('Using point coordinates as features')
        in_features += dim

    net = DGCNNSeg(k=args.k, in_features=in_features, num_classes=ds.num_classes)
    net.to(args.device)

    # training setup
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # TODO: weight classes

    train_loss = torch.zeros(args.epochs)
    valid_loss = torch.zeros_like(train_loss)
    train_dice = torch.zeros(args.epochs, ds.num_classes)
    valid_dice = torch.zeros_like(train_dice)
    best_model = None
    best_epoch = 0
    every_n_epochs = 10
    for epoch in range(args.epochs):
        # TRAINING
        net.train()
        for pts, lbls in train_dl:
            pts = pts.to(args.device)
            lbls = lbls.to(args.device)

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
            pts = pts.to(args.device)
            lbls = lbls.to(args.device)

            # forward pass
            with torch.no_grad():
                out = net(pts)

            loss = criterion(out, lbls)

            # statistics
            valid_loss[epoch] += loss.item() / len(valid_dl)
            valid_dice[epoch] += batch_dice(out.argmax(1), lbls, ds.num_classes) / len(valid_dl)

        # status output
        print(f'[{epoch:3}] TRAIN: {train_loss[epoch]:.4f} loss, dice {train_dice[epoch]}, mean {train_dice[epoch, :].mean()}\n'
              f'      VALID: loss {valid_loss[epoch]:.4f}, dice {valid_dice[epoch]}, mean {valid_dice[epoch, :].mean()}')

        # save best snapshot
        if valid_dice[epoch, :].mean() >= valid_dice[best_epoch, :].mean():
            best_model = deepcopy(net.state_dict())
            best_epoch = epoch

        # visualization
        if not (epoch + 1) % every_n_epochs:
            visualize_point_cloud(pts[0], out.argmax(1)[0])

    # training plot
    plt.figure()
    plt.title(f'Training Progression. Best model from epoch {best_epoch}.')
    plt.plot(train_loss, c='b', label='train loss')
    plt.plot(valid_loss, c='r', label='validation loss')
    plt.plot(valid_dice[:, :].mean(1), c='g', label='mean validation dice')
    plt.legend()
    plt.savefig(os.path.join(args.output, f"training_progression.png"))
    plt.close()

    # save best model
    model_path = os.path.join(args.output, 'model.pth')
    print(f'Saving best model from epoch {best_epoch} to path "{model_path}"')
    torch.save(best_model, model_path)

    # save setup
    setup_dict = {key: val for key, val in args._get_kwargs()}
    with open(os.path.join(args.output, 'setup.csv'), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(setup_dict.keys()))
        writer.writeheader()
        writer.writerow(setup_dict)


def test(args):
    # load data
    # TODO: test-split
    if args.data == 'fissures':
        ds = PointDataset(args.pts)
    elif args.data == 'faust':
        ds = FaustDataset(args.pts)
    else:
        print(f'No data set named "{args.data}". Exiting.')
        return

    dim = ds[0][0].shape[0]

    # load model
    in_features = 0
    if args.patch:
        print('Using image patch around points as features')
        print('NOT IMPLEMENTED YET')
        radius = 3
        in_features += radius ** dim

    if args.coords:
        print('Using point coordinates as features')
        in_features += dim

    net = DGCNNSeg(k=args.k, in_features=in_features, num_classes=ds.num_classes)
    net.to(args.device)
    net.load_state_dict(torch.load(os.path.join(args.output, 'model.pth')))
    net.eval()

    test_dice = torch.zeros(ds.num_classes)
    for pts, lbls in ds:
        pts = pts.unsqueeze(0).to(args.device)
        lbls = lbls.unsqueeze(0).to(args.device)

        with torch.no_grad():
            out = net(pts)

        test_dice += batch_dice(out.argmax(1), lbls, ds.num_classes)

    test_dice /= len(ds)

    print(f'Test dice per class: {test_dice}')

    # output file
    with open(os.path.join(args.output, 'test_results.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Class'] + [str(i) for i in range(ds.num_classes)] + ['mean'])
        writer.writerow(['Test Dice'] + [d.item() for d in test_dice] + [test_dice.mean()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCNN for lung fissure segmentation.')
    parser.add_argument('--epochs', default=100, help='max. number of epochs', type=int)
    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--device', default='cuda:2', help='device to train on', type=str)
    parser.add_argument('--data', help='data set', type=str, choices=['fissures', 'faust'])
    parser.add_argument('--k', default=20, help='number of neighbors for graph computation', type=int)
    parser.add_argument('--pts', default=1024, help='number of points per forward pass', type=int)
    parser.add_argument('--coords', const=True, default=False, help='use point coords as features', nargs='?')
    parser.add_argument('--patch', const=True, default=False, help='use image patch around points as features', nargs='?')
    parser.add_argument('--batch', default=4, help='batch size', type=int)
    parser.add_argument('--output', default='./results', help='output data path', type=str)
    args = parser.parse_args()
    print(args)

    train(args)
    test(args)
