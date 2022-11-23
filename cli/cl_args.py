import argparse
import json

from constants import KP_MODES, FEATURE_MODES
from losses.access_losses import Losses
from models.folding_net import SHAPE_TYPES
from shape_model.generate_corresponding_points import CORRESPONDENCE_MODES


def add_training_parameters(parser):
    group = parser.add_argument_group('Training Parameters')

    group.add_argument('--epochs', default=1000, help='max. number of epochs', type=int)
    group.add_argument('--lr', default=0.001, help='learning rate', type=float)
    group.add_argument('--batch', default=32, help='batch size', type=int)
    group.add_argument('--loss', help='loss function for training. "nnunet" is cross entropy + DICE loss, '
                       '"recall" is weighted cross entropy that promotes recall.', default='nnunet',
                       type=str, choices=Losses.list())
    group.add_argument("--loss_weights", nargs='+', default=None, type=float,
                       help="Weights for the components of loss function.")
    group.add_argument('--wd', default=1e-5, help='weight decay parameter for Adam optimizer', type=float)
    group.add_argument('--scheduler', default='plateau', help='the learn rate scheduler to use', type=str,
                       choices=['cosine', 'plateau', 'none'])


def add_test_parameters(parser):
    group = parser.add_argument_group('Testing Parameters')

    group.add_argument('--test_only', const=True, default=False, help='do not train model', nargs='?')
    group.add_argument('--train_only', const=True, default=False, help='do not test model', nargs='?')
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
    group.add_argument('--split', default=None, type=str,
                       help='cross validation split file. If None: will take the dataset defaults.')
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

    parser.set_defaults(scheduler='cosine')

    return parser


def get_seg_cnn_train_parser():
    parser = get_generic_parser('Train 3D CNN for lung fissure segmentation.')

    group = parser.add_argument_group('3D CNN parameters')
    group.add_argument('--model', choices=['v1', 'v3'], default='v1',
                       help='Choose the model class. Either MobilenetV1 with ASPP or MobilenetV3 with lr-ASPP')
    group.add_argument('--patch_size', help='patch size used for each dimension during training', default=96, type=int)
    group.add_argument('--spacing', help='isotropic resample to this spacing (in mm)', default=1.5, type=float)

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
    group.add_argument('--predict_affine', const=True, default=False, nargs='?',
                       help='predict the affine transformation of the corresponding points')
    group.add_argument('--corr_mode', default='simple', choices=CORRESPONDENCE_MODES, type=str,
                       help='mode of the point correspondence generation')
    group.add_argument('--head_schedule', default={'main': 150, 'translation': 0, 'rotation': 100, 'scaling': 50},
                       type=json.loads, help='json string containing the epoch when the particular head is supposed to be actived during training.')

    parser.set_defaults(loss='ssm')
    return parser


def get_pc_ae_train_parser():
    parser = get_dgcnn_train_parser()
    parser.description = 'Train DGCNN+FoldingNet Encoder+Decoder'

    group = parser.add_argument_group('FoldingNet parameters')
    group.add_argument("--latent", help="Dimensionality of latent shape code (z).", default=512, type=int)
    group.add_argument("--shape", help="Shape type to be folded by the FoldingNet decoder.", choices=SHAPE_TYPES,
                       default='plane')
    group.add_argument("--mesh", default=False, const=True,
                       help="Make the decoder fold a mesh instead of a point cloud.", nargs='?')
    group.add_argument("--deform", default=False, const=True,
                       help="Use deforming decoder instead of folding.", nargs='?')
    group.add_argument("--obj", help="Only use the index of this object (use all objects per default)", type=int,
                       default=None)

    parser.set_defaults(loss='mesh')
    return parser


def get_ae_reg_parser():
    parser = get_generic_parser('Prediction of the segmentation DGCNN to be regularized by the PC-AE (test-only).')

    group = parser.add_argument_group('AE-regularization parameters')
    group.add_argument("--seg_dir", type=str, required=True,
                       help='Directory of the cross-validation of the segmentation DGCNN.')
    group.add_argument("--ae_dir", type=str, required=True,
                       help='Directory of the cross-validation of the PC-AE.')

    parser.set_defaults(test_only=True)
    return parser
