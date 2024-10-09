import os

import torch
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance

from cli.cli_args import get_pc_ae_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from constants import IMG_DIR, IMG_DIR_TS_PREPROC
from data import SampleFromMeshDS
from metrics import pseudo_symmetric_point_to_mesh_distance, assd
from models.folding_net import DGCNNFoldingNet
from thesis.utils import param_and_op_count
from train import run, write_results
from utils.detached_run import maybe_run_detached_cli
from utils.general_utils import new_dir, pt3d_to_o3d_meshes
from visualization import point_cloud_on_axis, trimesh_on_axis, color_2d_points_bremm, color_2d_mesh_bremm


def normalize_pc_zstd(pc):
    mu = pc.mean(dim=0, keepdim=True)
    sigma = pc.std()  # global stddev so that proportions don't get warped between axes
    pc = (pc - mu) / sigma
    return pc, mu, sigma


def unnormalize_zstd(pc, mu, sigma):
    return pc * sigma + mu


def test(ds: SampleFromMeshDS, device, out_dir, show, args):
    model = DGCNNFoldingNet.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    output_is_mesh = model.decoder.decode_mesh

    pred_dir = new_dir(out_dir, 'test_predictions')
    plot_dir = new_dir(pred_dir, 'plots')

    # visualize folding points
    folding_points = model.decoder.get_folding_points(1).squeeze().cpu()
    if folding_points.shape[1] == 2:  # plane mode
        # add the third coordinate to enable 3d plot
        folding_points = torch.cat([folding_points, torch.zeros(folding_points.shape[0], 1)], dim=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    if output_is_mesh:
        faces = model.decoder.faces.cpu().squeeze()
        color_values = color_2d_mesh_bremm(folding_points[:, :2], faces)
        trimesh_on_axis(ax1, folding_points, faces, facecolors=color_values)
    else:
        color_values = color_2d_points_bremm(folding_points[:, :2])
        point_cloud_on_axis(ax1, folding_points, c=color_values, title="folding points")
    fig.savefig(os.path.join(plot_dir, 'folding_points.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    chamfer_dists = torch.zeros(len(ds.ids), ds.num_objects)
    all_mean_assd = torch.zeros_like(chamfer_dists)
    all_mean_sdsd = torch.zeros_like(chamfer_dists)
    all_hd_assd = torch.zeros_like(chamfer_dists)
    all_hd95_assd = torch.zeros_like(chamfer_dists)
    for i in range(len(ds)):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            x_reconstruct = model(x)

        # unnormalize point cloud
        x = ds.unnormalize_sampled_pc(x.squeeze().transpose(0, 1), i)
        if output_is_mesh:
            x_reconstruct = ds.unnormalize_mesh(x_reconstruct, i)
        else:
            x_reconstruct = ds.unnormalize_sampled_pc(x_reconstruct.transpose(0, 1).squeeze(), i)

        # compute chamfer distance (CAVE: pt3d computes it as mean squared distance and returns d_cham(x,y)+d_cham(y,x))
        cur_pat = ds.continuous_to_pat_index(i)
        cur_obj = ds.continuous_to_obj_index(i)
        chamfer_dists[cur_pat, cur_obj] = chamfer_distance(x_reconstruct.verts_padded(), x.unsqueeze(0),
                                                           point_reduction='mean')[0]

        # compute surface distance
        if output_is_mesh:
            mean, std, hd, hd95 = assd(pt3d_to_o3d_meshes(x_reconstruct)[0], ds.get_obj_mesh(i))
        else:
            mean, std, hd, hd95 = pseudo_symmetric_point_to_mesh_distance(x_reconstruct.cpu(), ds.get_obj_mesh(i))
        all_mean_assd[cur_pat, cur_obj] = mean
        all_mean_sdsd[cur_pat, cur_obj] = std
        all_hd_assd[cur_pat, cur_obj] = hd
        all_hd95_assd[cur_pat, cur_obj] = hd95

        # visualize reconstruction
        fig = plt.figure()
        if output_is_mesh:
            ax1 = fig.add_subplot(111, projection='3d')
            point_cloud_on_axis(ax1, x.cpu(), 'k', label='input', alpha=0.3)
            trimesh_on_axis(ax1, x_reconstruct.verts_padded().cpu().squeeze(), faces, facecolors=color_values, alpha=0.7, label='reconstruction')
        else:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            point_cloud_on_axis(ax1, x.cpu(), 'k', title='input')
            point_cloud_on_axis(ax2, x_reconstruct.cpu(), color_values, title='reconstruction')

        fig.savefig(os.path.join(plot_dir,
                                 f'{"_".join(ds.get_id(i))}_{"fissure" if not ds.lobes else "lobe"}{cur_obj+1}_reconstruction.png'),
                    dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # compute average metrics
    mean_assd = all_mean_assd.mean(0)
    std_assd = all_mean_assd.std(0)

    mean_sdsd = all_mean_sdsd.mean(0)
    std_sdsd = all_mean_sdsd.std(0)

    mean_hd = all_hd_assd.mean(0)
    std_hd = all_hd_assd.std(0)

    mean_hd95 = all_hd95_assd.mean(0)
    std_hd95 = all_hd95_assd.std(0)

    dice_dummy = torch.zeros_like(mean_assd)

    write_results(os.path.join(out_dir, 'test_results.csv'), dice_dummy, dice_dummy, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    # print out results
    print('\n============ RESULTS ============')
    print(f'Mean ASSD per class: {mean_assd} +- {std_assd}')

    return dice_dummy, dice_dummy, \
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, None


if __name__ == '__main__':
    parser = get_pc_ae_train_parser()
    args = parser.parse_args()
    if args.copd:
        raise NotImplementedError('COPD data validation is not applicable to PC-AE')
    maybe_run_detached_cli(args)

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # configure dataset
    if args.ds == 'data':
        mesh_dir = IMG_DIR
    elif args.ds == 'ts':
        mesh_dir = IMG_DIR_TS_PREPROC
    else:
        raise ValueError(f'No dataset named {args.ds}')

    ds = SampleFromMeshDS(mesh_dir, args.pts, fixed_object=args.obj, lobes=args.data == 'lobes', mesh_as_target=args.mesh)

    # configure model
    if not args.mesh and not args.loss == 'mesh':
        raise ValueError('Cannot compute mesh loss for non-mesh reconstructions of the AE.')

    model = DGCNNFoldingNet(k=args.k, n_embedding=args.latent, n_input_points=args.pts, decode_mesh=args.mesh,
                            deform=args.deform, static=args.static)

    # create output directory
    new_dir(args.output)

    param_and_op_count(model, (ds.num_objects, *ds[0][0].shape), out_dir=args.output)

    # store config
    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # run the training
    run(ds, model, test, args)
