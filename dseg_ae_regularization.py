import os

import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance

from cli.cli_args import get_ae_reg_parser
from cli.cli_utils import load_args_for_testing, store_args
from constants import POINT_DIR, POINT_DIR_TS, IMG_DIR_TS, IMG_DIR
from data import load_split_file, save_split_file, ImageDataset, PointToMeshDS
from data_processing.surface_fitting import o3d_mesh_to_labelmap
from metrics import assd, pseudo_symmetric_point_to_mesh_distance
from models.dgcnn import DGCNNSeg
from models.folding_net import DGCNNFoldingNet, FoldingDecoder
from models.modelio import LoadableModel, store_config_args
from train import write_results, run, write_speed_results
from utils.detached_run import maybe_run_detached_cli
from utils.general_utils import new_dir, pt3d_to_o3d_meshes, nanstd, \
    get_device, no_print, knn
from visualization import color_2d_mesh_bremm, trimesh_on_axis, color_2d_points_bremm, point_cloud_on_axis, \
    visualize_o3d_mesh


def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    if N <= num_points:
        if N < num_points:
            print(f'Tried to sample {num_points} from a point cloud with only {N}')
        return kpts, torch.arange(N)
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))

    return kpts[:, ind, :], ind


class RegularizedSegDGCNN(LoadableModel):
    @store_config_args
    def __init__(self, seg_model, ae, n_points_seg, n_points_ae, sample_mode='farthest', random_extend=False):
        super(RegularizedSegDGCNN, self).__init__()
        self.seg_model = DGCNNSeg.load(seg_model, 'cpu')
        self.ae = DGCNNFoldingNet.load(ae, 'cpu')
        self.n_points_seg = n_points_seg
        self.n_points_ae = n_points_ae
        self.sample_mode = sample_mode
        self.random_extend = random_extend

    @torch.no_grad()
    def segment(self, x):
        return self.seg_model.predict_full_pointcloud(x, self.n_points_seg).argmax(1)

    @torch.no_grad()
    def reconstruct(self, x, seg):
        # get refined mesh for each object
        points = []
        meshes = []
        for obj in range(1, self.seg_model.num_classes):
            # get coordinates from the current object point cloud
            pts_per_obj = x[:, :3].transpose(1, 2)[(seg == obj)].view(x.shape[0], -1, 3)
            # points.append(pts_per_obj)

            # skip if less than k points are present
            if pts_per_obj.shape[1] < self.ae.encoder.k:
                meshes.append(None)
                continue

            # extend the points with randomly jittered points
            if self.random_extend:
                pts_per_obj = random_extend_points(pts_per_obj, self.n_points_ae)

            # sample right amount of points from segmentation
            if self.sample_mode == 'farthest':
                sampled = farthest_point_sampling(pts_per_obj, self.n_points_ae)[0].transpose(1, 2)
                mesh = self.ae(sampled)
                points.append(sampled.transpose(1, 2))
            elif self.sample_mode == 'accumulate':
                mesh = self.ae.predict_full_pointcloud(pts_per_obj, self.n_points_ae, n_runs=10)
                points.append(pts_per_obj)
            else:
                raise NotImplementedError(f'Sampling mode {self.sample_mode} not implemented.')

            meshes.append(mesh)

        return meshes, points

    @torch.no_grad()
    def forward(self, x):
        # segmentation of the point cloud
        seg = self.segment(x)
        return self.reconstruct(x, seg)


def random_extend_points(points, desired_n_points):
    n_points = points.shape[1]
    n_points_to_pad = desired_n_points - n_points
    if n_points_to_pad <= 0:
        return points

    # compute average nearest distance
    idx, dist = knn(points.transpose(1, 2), 1, self_loop=False, return_dist=True)
    dist = dist.sqrt()
    avg_dist = dist.mean()
    dist_std = dist.std()

    # randomly choose source points (sample with replacement)
    source_points = points[:, torch.randint(n_points, (n_points_to_pad,))]

    # compute random displacement vectors with their norm having mean of avg_dist / 2
    direction = torch.randn_like(source_points)  # random direction vectors
    direction = direction / direction.norm(dim=2, keepdim=True)  # normalize
    magnitude = torch.randn(points.shape[0], n_points_to_pad, 1, device=points.device) * dist_std + avg_dist  # random magnitudes with mean and std like original point cloud
    displacement = direction * magnitude  # put direction and magnitude together
    jitter_points = source_points + displacement

    # extend points with randomly jittered source points
    return torch.cat([points, jitter_points], dim=1)


def test(ds: PointToMeshDS, device, out_dir, show, args):
    model = RegularizedSegDGCNN.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    output_is_mesh = model.ae.decoder.decode_mesh

    if output_is_mesh:
        image_ds = ImageDataset(ds.image_folder, do_augmentation=False)
        label_dir = new_dir(out_dir, 'test_predictions', 'labelmaps')

    # pred_dir = new_dir(out_dir, 'test_predictions')
    plot_dir = new_dir(out_dir, 'plots')

    # visualize folding points
    folding_points = model.ae.decoder.get_folding_points(1).squeeze().cpu()
    if folding_points.shape[1] == 2:  # plane mode
        # add the third coordinate to enable 3d plot
        folding_points = torch.cat([folding_points, torch.zeros(folding_points.shape[0], 1)], dim=1)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    if output_is_mesh:
        faces = model.ae.decoder.faces.cpu().squeeze()
        color_values = color_2d_mesh_bremm(folding_points[:, :2], faces)
        trimesh_on_axis(ax1, folding_points, faces, facecolors=color_values)
    else:
        color_values = color_2d_points_bremm(folding_points[:, :2])
        point_cloud_on_axis(ax1, folding_points, c=color_values, title="folding points")
    fig.savefig(os.path.join(plot_dir, 'folding_points.png'), dpi=300, bbox_inches='tight')
    # again without axes
    ax1.axis('off')
    fig.savefig(os.path.join(plot_dir, 'folding_points_no_ax.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    if isinstance(model.ae.decoder, FoldingDecoder):
        # show only the points from now on
        color_values = color_2d_points_bremm(folding_points[:, :2])

    chamfer_dists = torch.zeros(len(ds.ids), ds.num_classes - 1)
    all_mean_assd = torch.zeros_like(chamfer_dists)
    all_mean_sdsd = torch.zeros_like(chamfer_dists)
    all_hd_assd = torch.zeros_like(chamfer_dists)
    all_hd95_assd = torch.zeros_like(chamfer_dists)
    for i in range(len(ds)):
        x, lbl = ds.get_full_pointcloud(i)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            reconstruct_meshes, sampled_points_per_obj = model(x)

        meshes_pred_o3d = []
        for cur_obj, reconstruct_obj in enumerate(reconstruct_meshes):
            if reconstruct_obj is None:
                all_mean_assd[i, cur_obj] = float('NaN')
                all_mean_sdsd[i, cur_obj] = float('NaN')
                all_hd_assd[i, cur_obj] = float('NaN')
                all_hd95_assd[i, cur_obj] = float('NaN')
                continue

            # unnormalize point cloud
            input_coords = x[0, :3, lbl==cur_obj+1].transpose(0, 1)
            input_coords = ds.unnormalize_pc(input_coords, i)
            segmented_sampled_coords = ds.unnormalize_pc(sampled_points_per_obj[cur_obj], i)
            if output_is_mesh:
                reconstruct_obj = ds.unnormalize_mesh(reconstruct_obj, i)
            else:
                reconstruct_obj = ds.unnormalize_pc(reconstruct_obj.transpose(0, 1).squeeze(), i)

            # compute chamfer distance (CAVE: pt3d computes it as mean squared distance and returns d_cham(x,y)+d_cham(y,x))
            # currently no GPU support
            reconstruct_obj = reconstruct_obj.cpu()
            input_coords = input_coords.cpu()
            chamfer_dists[i, cur_obj] = chamfer_distance(reconstruct_obj.verts_padded(), input_coords.unsqueeze(0),
                                                         point_reduction='mean')[0]

            # compute surface distance
            if output_is_mesh:
                reconstruct_obj_o3d = pt3d_to_o3d_meshes(reconstruct_obj)[0]
                meshes_pred_o3d.append(reconstruct_obj_o3d)
                mean, std, hd, hd95 = assd(reconstruct_obj_o3d, ds.meshes[i][cur_obj])
            else:
                mean, std, hd, hd95 = pseudo_symmetric_point_to_mesh_distance(reconstruct_obj.cpu(), ds.meshes[i][cur_obj])
            all_mean_assd[i, cur_obj] = mean
            all_mean_sdsd[i, cur_obj] = std
            all_hd_assd[i, cur_obj] = hd
            all_hd95_assd[i, cur_obj] = hd95

            # visualize reconstruction
            fig = plt.figure()
            if output_is_mesh and not isinstance(model.ae.decoder, FoldingDecoder):
                ax1 = fig.add_subplot(111, projection='3d')
                # point_cloud_on_axis(ax1, input_coords.cpu(), 'b', label='input', alpha=0.3)
                point_cloud_on_axis(ax1, segmented_sampled_coords.cpu(), 'k', label='segmented points', alpha=0.3)
                trimesh_on_axis(ax1, reconstruct_obj.verts_padded().cpu().squeeze(), faces, facecolors=color_values, alpha=0.7, label='reconstruction')

                # # again only mesh with no axis
                # fig = plt.figure()
                # ax1 = fig.add_subplot(111, projection='3d')
                # trimesh_on_axis(ax1, reconstruct_obj.verts_padded().cpu().squeeze(), faces, facecolors=color_values,
                #                 alpha=0.7, )
                # ax1.axis("off")
                # fig.savefig(os.path.join(plot_dir,
                #                          f'{"_".join(ds.ids[i])}_{"fissure" if not ds.lobes else "lobe"}{cur_obj + 1}_reconstruction_no_ax.png'),
                #             dpi=300, bbox_inches='tight')

                # # only points
                # fig = plt.figure()
                # ax1 = fig.add_subplot(111, projection='3d')
                # ax1.axis('off')
                # point_cloud_on_axis(ax1, segmented_sampled_coords.cpu(), 'k', alpha=0.3)
                # fig.savefig(os.path.join(plot_dir, f'{"_".join(ds.ids[i])}_{"fissure" if not ds.lobes else "lobe"}{cur_obj + 1}_reconstruction_pts.png'),
                #             dpi=300, bbox_inches='tight')

            else:
                if isinstance(model.ae.decoder, FoldingDecoder):
                    points = reconstruct_obj.verts_padded().cpu().squeeze()
                else:
                    points = reconstruct_obj.cpu()
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                point_cloud_on_axis(ax1, input_coords.cpu(), 'k', title='input')
                point_cloud_on_axis(ax2, points, color_values, title='reconstruction')

            fig.savefig(os.path.join(plot_dir,
                                     f'{"_".join(ds.ids[i])}_{"fissure" if not ds.lobes else "lobe"}{cur_obj+1}_reconstruction.png'),
                        dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        if output_is_mesh:
            case, sequence = ds.ids[i]

            # visualize all objects in one plot
            title_prefix = f'{case}_{sequence}'
            visualize_o3d_mesh(meshes_pred_o3d, title=title_prefix + ' surface prediction', show=show,
                               savepath=os.path.join(plot_dir, f'{title_prefix}_mesh_pred.png'))

            # voxelize meshes to label maps
            label_map_predict = o3d_mesh_to_labelmap(meshes_pred_o3d, shape=ds.img_sizes_index[i][::-1], spacing=ds.spacings[i])
            label_image_predict = sitk.GetImageFromArray(label_map_predict.numpy().astype(np.uint8))
            label_image_predict.CopyInformation(image_ds.get_lung_mask(image_ds.get_index(case, sequence)))
            sitk.WriteImage(label_image_predict, os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))

    # compute average metrics
    mean_assd = all_mean_assd.nanmean(0)
    std_assd = nanstd(all_mean_assd, 0)

    mean_sdsd = all_mean_sdsd.nanmean(0)
    std_sdsd = nanstd(all_mean_sdsd, 0)

    mean_hd = all_hd_assd.nanmean(0)
    std_hd = nanstd(all_hd_assd, 0)

    mean_hd95 = all_hd95_assd.nanmean(0)
    std_hd95 = nanstd(all_hd95_assd, 0)

    dice_dummy = torch.zeros_like(mean_assd)

    percent_missing = all_mean_assd.isnan().float().mean(0) * 100

    write_results(os.path.join(out_dir, f'test_results{"_copd" if args.copd else ""}.csv'), dice_dummy, dice_dummy, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing)

    # print out results
    print('\n============ RESULTS ============')
    print(f'Mean ASSD per class: {mean_assd} +- {std_assd}')

    return dice_dummy, dice_dummy, \
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing


def speed_test(ds: PointToMeshDS, device, out_dir):
    model = RegularizedSegDGCNN.load(os.path.join(out_dir, 'fold0', 'model.pth'), device=device)
    model.to(device)
    model.eval()

    torch.manual_seed(42)

    # prepare measuring inference time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_inference_times = []
    all_post_proc_times = []
    all_points_per_fissure = []
    for i in range(len(ds)):
        x, lbl = ds.get_full_pointcloud(i)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad(), no_print():
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))

            seg = model.segment(x)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))

            reconstruct_meshes, sampled_points_per_obj = model.reconstruct(x, seg)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_post_proc_times.append(curr_time)

        all_points_per_fissure.append(torch.tensor([len(o.squeeze()) for o in sampled_points_per_obj]))

    write_speed_results(out_dir, all_inference_times, all_post_proc_times=all_post_proc_times,
                        points_per_fissure=all_points_per_fissure if not model.random_extend else None)


if __name__ == '__main__':
    parser = get_ae_reg_parser()
    args = parser.parse_args()

    if args.copd:
        raise NotImplementedError('COPD data validation is not yet implemented for DGCNN + PC-AE')

    maybe_run_detached_cli(args)

    dgcnn_args = load_args_for_testing(args.seg_dir, args)
    pc_ae_args = load_args_for_testing(args.ae_dir, args)

    assert dgcnn_args.data == pc_ae_args.data
    # assert dgcnn_args.ds == pc_ae_args.ds
    # assert dgcnn_args.exclude_rhf == pc_ae_args.exclude_rhf
    # assert dgcnn_args.split == pc_ae_args.split
    assert not dgcnn_args.binary

    # if dgcnn_args.split is None:
    #     dgcnn_args.split = DEFAULT_SPLIT if dgcnn_args.ds == 'data' else DEFAULT_SPLIT_TS
    split = load_split_file(os.path.join(dgcnn_args.output, "cross_val_split.np.pkl"))
    new_dir(args.output)
    save_split_file(split, os.path.join(args.output, "cross_val_split.np.pkl"))
    for fold in range(len(split)):
        model = RegularizedSegDGCNN(os.path.join(dgcnn_args.output, f'fold{fold}', 'model.pth'),
                                    os.path.join(pc_ae_args.output, f'fold{fold}', 'model.pth'),
                                    n_points_seg=dgcnn_args.pts, n_points_ae=pc_ae_args.pts,
                                    sample_mode=args.sampling, random_extend=args.pad_with_random_offsets)
        fold_dir = new_dir(args.output, f"fold{fold}")
        model.save(os.path.join(fold_dir, 'model.pth'))

    if args.ds == 'data':
        point_dir = POINT_DIR
        img_dir = IMG_DIR
    elif args.ds == 'ts':
        point_dir = POINT_DIR_TS
        img_dir = IMG_DIR_TS
    else:
        raise ValueError(f'No dataset named {args.ds}')

    print(f'Using point data from {point_dir}')
    ds = PointToMeshDS(dgcnn_args.pts, kp_mode=dgcnn_args.kp_mode, use_coords=dgcnn_args.coords, folder=point_dir,
                       image_folder=img_dir,
                       patch_feat=dgcnn_args.patch, exclude_rhf=dgcnn_args.exclude_rhf, lobes=dgcnn_args.data == 'lobes',
                       binary=dgcnn_args.binary, do_augmentation=False)

    if args.speed:
        speed_test(ds, get_device(args.gpu), args.output)
        exit()

    store_args(args=args, out_dir=args.output)

    run(ds, model, test, args)
