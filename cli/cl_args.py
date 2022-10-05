import argparse

from data_processing.keypoint_extraction import KP_MODES
from data_processing.point_features import FEATURE_MODES
from losses.access_losses import Losses


def add_training_parameters(parser):
    group = parser.add_argument_group('Training Parameters')

    group.add_argument('--epochs', default=1000, help='max. number of epochs', type=int)
    group.add_argument('--lr', default=0.001, help='learning rate', type=float)
    group.add_argument('--batch', default=32, help='batch size', type=int)
    group.add_argument('--loss', help='loss function for training. "nnunet" is cross entropy + DICE loss, '
                       '"recall" is weighted cross entropy that promotes recall.', default='nnunet',
                       type=str, choices=Losses.list())
    group.add_argument('--wd', default=1e-5, help='weight decay parameter for Adam optimizer', type=float)


def add_test_parameters(parser):
    group = parser.add_argument_group('Testing Parameters')

    group.add_argument('--test_only', const=True, default=False, help='do not train model', nargs='?')
    group.add_argument('--fold', default=None, type=int,
                       help='specify if only one fold should be evaluated '
                            '(needs to be in range of folds in the split file)')


def add_data_parameters(parser):
    group = parser.add_argument_group('Data Parameters')

    group.add_argument('--data', help='type of data, either fissures or lobes',
                       default='fissures', type=str, choices=['fissures', 'lobes'])
    group.add_argument('--ds', help='dataset to use',
                       default='data', type=str, choices=['data', 'ts'])
    group.add_argument('--kp_mode', default='foerstner', help='keypoint extraction mode', type=str, choices=KP_MODES)
    group.add_argument('--exclude_rhf', const=True, default=False,
                       help='exclude the right horizontal fissure from the model', nargs='?')
    group.add_argument('--split', default=None, type=str, help='cross validation split file')
    group.add_argument('--binary', const=True, default=False, nargs='?',
                       help='make classification problem binary (only train with fissure/non-fissure labels)')


def get_generic_parser(description):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--gpu', default=2, help='gpu index to train on', type=int)
    parser.add_argument('--output', default='./results', help='output data path', type=str)
    parser.add_argument('--show', const=True, default=False, help='turn on plots (will only be saved by default)',
                        nargs='?')
    parser.add_argument('--offline', const=True, default=False,
                        help='Runs the script with nohup and detaches the script. Disables the --show option. '
                             'Output logs will be saved to "./results/logs/<script_name>_<timestamp>.txt"',
                        nargs='?')

    add_training_parameters(parser)
    add_data_parameters(parser)
    add_test_parameters(parser)

    return parser


def get_dgcnn_train_parser():
    parser = get_generic_parser('Train DGCNN for lung fissure segmentation.')

    group = parser.add_argument_group('DGCNN parameters')
    group.add_argument('--k', default=20, help='number of neighbors for graph computation', type=int)
    group.add_argument('--pts', default=1024, help='number of points per forward pass', type=int)
    group.add_argument('--coords', const=True, default=False, help='use point coords as features', nargs='?')
    group.add_argument('--patch', default=None, type=str,
                       help=f'use image patch around points as features, one of {FEATURE_MODES}')
    group.add_argument('--transformer', const=True, default=False, help='use spatial transformer module in DGCNN',
                       nargs='?')
    group.add_argument('--static', const=True, default=False, help='do not use dynamic graph computation in DGCNN',
                       nargs='?')
    group.add_argument('--img_feat_extractor', const=True, default=False,
                       help='use an extra image feature extraction module', nargs='?')

    return parser


def get_seg_cnn_train_parser():
    parser = get_generic_parser('Train 3D CNN for lung fissure segmentation.')

    group = parser.add_argument_group('3D CNN parameters')
    group.add_argument('--patch_size', help='patch size used for each dimension during training', default=96, type=int)

    return parser


def get_dgcnn_ssm_train_parser():
    parser = get_dgcnn_train_parser()
    parser.description = 'Train DGCNN-Shape-Model Regression for lung fissure segmentation'

    group = parser.add_argument_group('SSM parameters')
    group.add_argument('--alpha', default=3., type=float,
                       help='Multiplier for plausible shape range (+-alpha*std.dev.)')
    group.add_argument('--target_variance', default=0.95, type=float,
                       help='Fraction of the dataset variance to be explained by the model')
    group.add_argument('--lssm', const=True, default=False,
                       help='use Localized SSM (Wilms et al., MedIA 2017) instead of standard SSM', nargs='?')

    parser.set_defaults(loss='ssm')
    return parser
