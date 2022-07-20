import os
import warnings

from cli.cl_args import get_dgcnn_ssm_train_parser
from data import CorrespondingPointDataset
from models.dg_ssm import DGSSM
from shape_model.ssm import SSM
from train import run


def test(ds, device, out_dir, show):
    # TODO: compute SSM reconstruction error as baseline
    pass


if __name__ == '__main__':
    parser = get_dgcnn_ssm_train_parser()
    args = parser.parse_args()

    if args.data == 'lobes':
        raise NotImplementedError()
    if args.binary:
        warnings.warn('--binary option has no effect when training the DG-SSM')
    if args.exclude_rhf:
        warnings.warn('--exclude_rhf option has no effect when training the DG-SSM')

    ds = CorrespondingPointDataset(sample_points=args.pts, kp_mode=args.kp_mode, use_coords=args.coords,
                                   patch_feat='mind' if args.patch else None)

    # setup model
    in_features = ds[0][0].shape[0]
    model = DGSSM(k=args.k, in_features=in_features,
                  spatial_transformer=args.transformer, dynamic=not args.static,
                  ssm_alpha=args.alpha, ssm_targ_var=args.target_variance)

    run(ds, model, test, args)
