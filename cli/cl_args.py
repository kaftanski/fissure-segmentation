import argparse
from data_processing.point_features import KP_MODES


def add_training_parameters(parser):
    parser.add_argument('--epochs', default=1000, help='max. number of epochs', type=int)
    parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
    parser.add_argument('--gpu', default=2, help='gpu index to train on', type=int)
    parser.add_argument('--batch', default=32, help='batch size', type=int)
    parser.add_argument('--test_only', const=True, default=False, help='do not train model', nargs='?')
    parser.add_argument('--output', default='./results', help='output data path', type=str)
    parser.add_argument('--show', const=True, default=False, help='turn on plots (will only be saved by default)',
                        nargs='?')


def add_data_parameters(parser):
    parser.add_argument('--data', help='data set', default='fissures', type=str, choices=['fissures', 'lobes'])
    parser.add_argument('--kp_mode', default='foerstner', help='keypoint extraction mode', type=str, choices=KP_MODES)
    parser.add_argument('--exclude_rhf', const=True, default=False,
                        help='exclude the right horizontal fissure from the model', nargs='?')
    parser.add_argument('--split', default=None, type=str, help='cross validation split file')
    parser.add_argument('--fold', default=None,
                        help='specify if only one fold should be evaluated (needs to be in range of folds in the split file)',
                        type=int)


def get_generic_parser(description):
    parser = argparse.ArgumentParser(description=description)

    add_training_parameters(parser)
    add_data_parameters(parser)

    return parser


def get_dgcnn_train_parser():
    parser = get_generic_parser('Train DGCNN for lung fissure segmentation.')

    parser.add_argument('--k', default=20, help='number of neighbors for graph computation', type=int)
    parser.add_argument('--pts', default=1024, help='number of points per forward pass', type=int)
    parser.add_argument('--coords', const=True, default=False, help='use point coords as features', nargs='?')
    parser.add_argument('--patch', const=True, default=False, help='use image patch around points as features',
                        nargs='?')
    parser.add_argument('--transformer', const=True, default=False, help='use spatial transformer module in DGCNN',
                        nargs='?')
    parser.add_argument('--static', const=True, default=False, help='do not use dynamic graph computation in DGCNN',
                        nargs='?')

    return parser


def get_seg_cnn_train_parser():
    parser = get_generic_parser('Train 3D CNN for lung fissure segmentation.')
    return parser
