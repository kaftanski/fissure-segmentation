"""
!!!WARNING: EXPERIMENTAL!!!
Training a point-cloud segmentation network with a differentiable Poisson surface reconstruction (DPSR) layer.
Seems not to work for pulmonary fissure segmentation.
"""
import os

from torchviz import make_dot

from cli.cli_args import get_dpsr_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from constants import POINT_DIR_COPD, IMG_DIR_COPD, POINT_DIR_TS, IMG_DIR_TS_PREPROC
from data_processing.datasets import PointToMeshAndLabelDataset
from models.seg_logits_to_mesh import DPSRNet2
from utils.model_utils import param_and_op_count
from train_point_segmentation import run
from utils.detached_run import maybe_run_detached_cli
from utils.general_utils import get_device, new_dir
from utils.visualization import visualize_trimesh


def test(ds: PointToMeshAndLabelDataset, device, out_dir, show, args):
    pass  # TODO


def speed_test(ds: PointToMeshAndLabelDataset, device, out_dir):
    pass  # TODO


def visualize_prediction(x, y, prediction, epoch, args, out_dir=None):
    pred_seg, pred_mesh = prediction
    pred_seg = pred_seg.detach().cpu()
    pred_mesh = pred_mesh.detach().cpu()
    n_classes = pred_seg.shape[1]
    savepath = os.path.join(new_dir(out_dir, 'validation_plots'), f'meshes_pred_epoch_{epoch}.png') \
        if out_dir is not None else None

    visualize_trimesh(pred_mesh[0:(n_classes-1)].verts_list(), pred_mesh[0:(n_classes-1)].faces_list(),
                      title=f'Predicted meshes (epoch {epoch})', savepath=savepath)


if __name__ == '__main__':
    parser = get_dpsr_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only or args.speed or args.copd:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # load data
    if args.copd:
        point_dir = POINT_DIR_COPD
        img_dir = IMG_DIR_COPD
    else:
        point_dir = POINT_DIR_TS
        img_dir = IMG_DIR_TS_PREPROC

    if args.copd:
        print('Validating with COPD dataset')
        args.test_only = True
        args.speed = False
    else:
        print(f'Using point data from {point_dir}')

    ds = PointToMeshAndLabelDataset(args.pts, kp_mode=args.kp_mode, use_coords=True,
                                    folder=point_dir, image_folder=img_dir,
                                    patch_feat=args.patch,
                                    exclude_rhf=args.exclude_rhf, lobes=args.data == 'lobes', binary=args.binary,
                                    copd=args.copd)

    # setup folder
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # run the speed test
    if args.speed:
        speed_test(ds, get_device(args.gpu), args.output)
        exit(0)

    # setup model
    in_features = ds[0][0].shape[0]

    dpsr_net = DPSRNet2(seg_net_class=args.model, k=args.k, in_features=in_features, num_classes=ds.num_classes,
                        spatial_transformer=args.transformer, dynamic=args.dynamic,
                        normals_smoothing_sigma=args.normals_sigma, dpsr_res=args.res, dpsr_sigma=args.sigma,
                        dpsr_scale=True, dpsr_shift=True)

    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # define the visualization function
    vars(args)['visualize'] = visualize_prediction

    # run the chosen configuration
    run(ds, dpsr_net, test, args)

    # random init network may only produce background label -> no mesh
    # -> count ops in a trained network
    trained = DPSRNet2.load(os.path.join(args.output, 'fold0', 'model.pth'), device='cpu')
    param_and_op_count(dpsr_net, (1, *ds[0][0].shape), out_dir=args.output)

    # visualize the backward graph
    x = ds[0][0].unsqueeze(0)
    y_seg, y_mesh = trained(x)
    backward_graph = make_dot(y_mesh.verts_packed(), params=dict(trained.named_parameters()))
    backward_graph.render(os.path.join(args.output, 'backward_graph'), cleanup=True)
