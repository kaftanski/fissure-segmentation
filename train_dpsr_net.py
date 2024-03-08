import os

from cli.cl_args import get_point_segmentation_parser, get_dpsr_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from constants import POINT_DIR, IMG_DIR, POINT_DIR_TS, IMG_DIR_TS
from data import PointToMeshAndLabelDataset
from models.dpsr_net import DPSRNet
from thesis.utils import param_and_op_count
from train import run
from models.access_models import get_point_seg_model_class_from_args
from utils.detached_run import maybe_run_detached_cli
from utils.general_utils import get_device


def test(ds: PointToMeshAndLabelDataset, device, out_dir, show, args):
    pass  # TODO


def speed_test(ds: PointToMeshAndLabelDataset, device, out_dir):
    pass  # TODO


if __name__ == '__main__':
    parser = get_dpsr_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only or args.speed or args.copd:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # load data
    if not args.coords and not args.patch:
        print('No features specified, defaulting to coords as features. '
              'To specify, provide arguments --coords and/or --patch.')

    if args.data in ['fissures', 'lobes']:
        if args.ds == 'data' or args.copd:
            point_dir = POINT_DIR
            img_dir = IMG_DIR
        elif args.ds == 'ts':
            point_dir = POINT_DIR_TS
            img_dir = IMG_DIR_TS
        else:
            raise ValueError(f'No dataset named {args.ds}')

        if args.copd:
            print('Validating with COPD dataset')
            args.test_only = True
            args.speed = False
        else:
            print(f'Using point data from {point_dir}')

        ds = PointToMeshAndLabelDataset(args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                                        folder=point_dir, image_folder=img_dir,
                                        patch_feat=args.patch,
                                        exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes', binary=args.binary,
                                        copd=args.copd)

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

    dpsr_net = DPSRNet(seg_net_class=args.model, in_features=in_features, num_classes=ds.num_classes, k=args.k,
                       spatial_transformer=args.transformer, dynamic=not args.static,
                       dpsr_res=args.res, dpsr_sigma=args.sigma, dpsr_scale=True, dpsr_shift=True)

    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # run the chosen configuration
    run(ds, dpsr_net, test, args)

    # random init network may only produce background label -> no mesh
    # -> count ops in a trained network
    trained = DPSRNet.load(os.path.join(args.output, 'fold0', 'model.pth'), device='cpu')
    param_and_op_count(dpsr_net, (1, *ds[0][0].shape), out_dir=args.output)
