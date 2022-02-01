import csv
from copy import deepcopy

import os

from matplotlib import pyplot as plt
from torch import nn
import torch
import argparse
from torch.utils.data import random_split, DataLoader
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


def train(args):
    # load data
    ds = PointDataset(args.data, args.pts)
    val_split = int(len(ds) * 0.2)
    train_ds, valid_ds = random_split(ds, lengths=[len(ds) - val_split, val_split])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch)

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

    net = DGCNNSeg(k=args.k, in_features=in_features, num_classes=args.classes)
    net.to(args.device)

    # training setup
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # TODO: weight classes

    train_loss = torch.zeros(args.epochs)
    valid_loss = torch.zeros_like(train_loss)
    train_dice = torch.zeros(args.epochs, args.classes)
    valid_dice = torch.zeros_like(train_dice)
    best_model = None
    best_epoch = 0
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
            train_dice[epoch] += batch_dice(out.argmax(1), lbls, args.classes) / len(train_dl)

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
            valid_dice[epoch] += batch_dice(out.argmax(1), lbls, args.classes) / len(valid_dl)

        # status output
        print(f'[{epoch:3}] train: {train_loss[epoch]:.4f} loss, {train_dice[epoch]} dice\n'
              f'      valid: {valid_loss[epoch]:.4f} loss, {valid_dice[epoch]} dice')

        # save best snapshot
        if valid_dice[epoch, 1:].mean() >= valid_dice[best_epoch, 1:].mean():
            best_model = deepcopy(net.state_dict())
            best_epoch = epoch

    # training plot
    plt.figure()
    plt.title(f'Training Progression. Best model from epoch {best_epoch}.')
    plt.plot(train_loss, c='b', label='train loss')
    plt.plot(valid_loss, c='r', label='validation loss')
    plt.plot(valid_dice[:, 1:].mean(1), c='g', label='mean validation dice (w/o background)')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DGCNN for lung fissure segmentation.')
    parser.add_argument('--epochs', default=100, help='max. number of epochs')
    parser.add_argument('--lr', default=1e-3, help='learning rate')
    parser.add_argument('--device', default='cuda:2', help='device to train on')
    parser.add_argument('--data', default='/home/kaftan/FissureSegmentation/point_data/', help='data path')
    parser.add_argument('--k', default=20, help='number of neighbors for graph computation')
    parser.add_argument('--pts', default=1024, help='number of points per forward pass')
    parser.add_argument('--coords', const=True, default=False, help='use point coords as features')
    parser.add_argument('--patch', const=True, default=False, help='use image patch around points as features')
    parser.add_argument('--classes', default=4, help='number of classes (including background)')
    parser.add_argument('--batch', default=4, help='batch size')
    parser.add_argument('--output', default='./results', help='output data path')
    args = parser.parse_args()
    print(args)

    train(args)
