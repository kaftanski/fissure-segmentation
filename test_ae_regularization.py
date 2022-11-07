import os

import torch
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

from cli.cl_args import get_ae_reg_parser
from cli.cli_utils import load_args_for_testing
from data import PointDataset, load_split_file, save_split_file
from data_processing.keypoint_extraction import POINT_DIR_TS, POINT_DIR
from metrics import assd, pseudo_symmetric_point_to_mesh_distance
from models.dgcnn import DGCNNSeg
from models.folding_net import DGCNNFoldingNet
from models.modelio import LoadableModel, store_config_args
from train import write_results, run
from utils.detached_run import maybe_run_detached_cli
from utils.image_ops import load_image_metadata
from utils.utils import new_dir, kpts_to_grid, kpts_to_world, ALIGN_CORNERS, pt3d_to_o3d_meshes, load_meshes
from visualization import color_2d_mesh_bremm, trimesh_on_axis, color_2d_points_bremm, point_cloud_on_axis


def farthest_point_sampling(kpts, num_points):
    _, N, _ = kpts.size()
    ind = torch.zeros(num_points).long()
    ind[0] = torch.randint(N, (1,))
    dist = torch.sum((kpts - kpts[:, ind[0], :]) ** 2, dim=2)
    for i in range(1, num_points):
        ind[i] = torch.argmax(dist)
        dist = torch.min(dist, torch.sum((kpts - kpts[:, ind[i], :]) ** 2, dim=2))

    return kpts[:, ind, :], ind


class RegularizedSegDGCNN(LoadableModel):
    @store_config_args
    def __init__(self, seg_model, ae, n_points_seg, n_points_ae, sample_mode='farthest'):
        super(RegularizedSegDGCNN, self).__init__()
        self.seg_model = DGCNNSeg.load(seg_model, 'cpu')
        self.ae = DGCNNFoldingNet.load(ae, 'cpu')
        self.n_points_seg = n_points_seg
        self.n_points_ae = n_points_ae
        self.sample_mode = sample_mode

    @torch.no_grad()
    def forward(self, x):
        # segmentation of the point cloud
        seg = self.seg_model.predict_full_pointcloud(x, self.n_points_seg).argmax(1)

        # get refined mesh for each object
        meshes = []
        for obj in seg.unique()[1:]:
            # get coordinates from the current object point cloud
            pts_per_obj = x[:, :3].transpose(1, 2)[(seg == obj)].view(x.shape[0], -1, 3)
            # sample right amount of points from segmentation
            if self.sample_mode == 'farthest':
                sampled = farthest_point_sampling(pts_per_obj, self.n_points_ae)[0].transpose(1, 2)
            else:
                raise NotImplementedError()

            mesh = self.ae(sampled)
            meshes.append(mesh)

        return meshes


class PointToMeshDS(PointDataset):
    def __init__(self, sample_points, kp_mode, folder, image_folder, use_coords=True,
                 patch_feat=None, exclude_rhf=False, lobes=False, binary=False, do_augmentation=True):
        super(PointToMeshDS, self).__init__(sample_points, kp_mode, folder, use_coords, patch_feat, exclude_rhf, lobes, binary, do_augmentation)
        self.meshes = []
        self.img_sizes = []
        for case, sequence in self.ids:
            meshes = load_meshes(image_folder, case, sequence, obj_name='fissure' if not lobes else 'lobe')
            if not lobes and exclude_rhf:
                meshes = meshes[:2]
            self.meshes.append(meshes)

            size, spacing = load_image_metadata(os.path.join(image_folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

    def normalize_sampled_pc(self, samples, index):
        return kpts_to_grid(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_sampled_pc(self, samples, index):
        return kpts_to_world(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def normalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.normalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def unnormalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.unnormalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())


def test(ds: PointToMeshDS, device, out_dir, show):
    model = RegularizedSegDGCNN.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    output_is_mesh = model.ae.decoder.decode_mesh

    pred_dir = new_dir(out_dir, 'test_predictions')
    plot_dir = new_dir(pred_dir, 'plots')

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
    if show:
        plt.show()
    else:
        plt.close(fig)

    chamfer_dists = torch.zeros(len(ds.ids), ds.num_classes)
    all_mean_assd = torch.zeros_like(chamfer_dists)
    all_mean_sdsd = torch.zeros_like(chamfer_dists)
    all_hd_assd = torch.zeros_like(chamfer_dists)
    all_hd95_assd = torch.zeros_like(chamfer_dists)
    for i in range(len(ds)):
        x, lbl = ds.get_full_pointcloud(i)
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            reconstruct_meshes = model(x)

        for cur_obj, reconstruct_obj in enumerate(reconstruct_meshes):
            # unnormalize point cloud
            coords = x[0, :3, lbl==cur_obj+1].transpose(0, 1)
            coords = ds.unnormalize_sampled_pc(coords, i)
            if output_is_mesh:
                reconstruct_obj = ds.unnormalize_mesh(reconstruct_obj, i)
            else:
                reconstruct_obj = ds.unnormalize_sampled_pc(reconstruct_obj.transpose(0, 1).squeeze(), i)

            # compute chamfer distance (CAVE: pt3d computes it as mean squared distance and returns d_cham(x,y)+d_cham(y,x))
            chamfer_dists[i, cur_obj] = chamfer_distance(reconstruct_obj.verts_padded(), coords.unsqueeze(0),
                                                         point_reduction='mean')[0]

            # compute surface distance
            if output_is_mesh:
                mean, std, hd, hd95 = assd(pt3d_to_o3d_meshes(reconstruct_obj)[0], ds.meshes[i][cur_obj])
            else:
                mean, std, hd, hd95 = pseudo_symmetric_point_to_mesh_distance(reconstruct_obj.cpu(), ds.meshes[i][cur_obj])
            all_mean_assd[i, cur_obj] = mean
            all_mean_sdsd[i, cur_obj] = std
            all_hd_assd[i, cur_obj] = hd
            all_hd95_assd[i, cur_obj] = hd95

            # visualize reconstruction
            fig = plt.figure()
            if output_is_mesh:
                ax1 = fig.add_subplot(111, projection='3d')
                point_cloud_on_axis(ax1, x.cpu(), 'k', label='input', alpha=0.3)
                trimesh_on_axis(ax1, reconstruct_obj.verts_padded().cpu().squeeze(), faces, facecolors=color_values, alpha=0.7, label='reconstruction')
            else:
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                point_cloud_on_axis(ax1, x.cpu(), 'k', title='input')
                point_cloud_on_axis(ax2, reconstruct_obj.cpu(), color_values, title='reconstruction')

            fig.savefig(os.path.join(plot_dir,
                                     f'{"_".join(ds.ids[i])}_{"fissure" if not ds.lobes else "lobe"}{cur_obj+1}_reconstruction.png'),
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
    parser = get_ae_reg_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    dgcnn_args = load_args_for_testing(args.seg_dir, args)
    pc_ae_args = load_args_for_testing(args.ae_dir, args)

    assert dgcnn_args.data == pc_ae_args.data
    # assert dgcnn_args.ds == pc_ae_args.ds
    assert dgcnn_args.data == pc_ae_args.data
    # assert dgcnn_args.exclude_rhf == pc_ae_args.exclude_rhf
    # assert dgcnn_args.split == pc_ae_args.split
    assert not dgcnn_args.binary

    split = load_split_file(dgcnn_args.split)
    save_split_file(split, os.path.join(args.output, "cross_val_split.np.pkl"))
    for fold in range(len(split)):
        model = RegularizedSegDGCNN(os.path.join(dgcnn_args.output, f'fold{fold}', 'model.pth'),
                                    os.path.join(pc_ae_args.output, f'fold{fold}', 'model.pth'),
                                    n_points_seg=dgcnn_args.pts, n_points_ae=pc_ae_args.pts)
        fold_dir = new_dir(args.output, f"fold{fold}")
        model.save(os.path.join(fold_dir, 'model.pth'))

    if args.ds == 'data':
        point_dir = POINT_DIR
        img_dir = '../data'
    elif args.ds == 'ts':
        point_dir = POINT_DIR_TS
        img_dir = '../TotalSegmentator/ThoraxCrop'
    else:
        raise ValueError(f'No dataset named {args.ds}')

    print(f'Using point data from {point_dir}')
    ds = PointToMeshDS(dgcnn_args.pts, kp_mode=dgcnn_args.kp_mode, use_coords=dgcnn_args.coords, folder=point_dir,
                       image_folder=img_dir,
                       patch_feat=dgcnn_args.patch, exclude_rhf=dgcnn_args.exclude_rhf, lobes=dgcnn_args.data == 'lobes',
                       binary=dgcnn_args.binary)

    run(ds, model, test, args)
