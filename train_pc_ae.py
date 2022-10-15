import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes, join_meshes_as_batch

from cli.cl_args import get_pc_ae_train_parser
from cli.cli_utils import load_args_for_testing, store_args
from data import CustomDataset
from metrics import pseudo_symmetric_point_to_mesh_distance, assd
from models.folding_net import DGCNNFoldingNet
from train import run, write_results
from utils.detached_run import maybe_run_detached_cli
from utils.image_ops import load_image_metadata
from utils.utils import load_meshes, new_dir, kpts_to_grid, ALIGN_CORNERS, kpts_to_world, o3d_to_pt3d_meshes, \
    pt3d_to_o3d_meshes
from visualization import point_cloud_on_axis, trimesh_on_axis, color_2d_points_bremm, color_2d_mesh_bremm


class SampleFromMeshDS(CustomDataset):
    def __init__(self, folder, sample_points, fixed_object: int = None, exclude_rhf=False, lobes=False, mesh_as_target=True):
        super(SampleFromMeshDS, self).__init__(exclude_rhf=exclude_rhf, binary=False, do_augmentation=True)

        self.sample_points = sample_points
        self.fixed_object = fixed_object
        self.mesh_as_target = mesh_as_target
        self.lobes = lobes

        mesh_dirs = sorted(glob.glob(os.path.join(folder, "*_mesh_*")))

        self.ids = []
        self.meshes = []
        self.img_sizes = []
        for mesh_dir in mesh_dirs:
            case, sequence = os.path.basename(mesh_dir).split("_mesh_")
            meshes = load_meshes(folder, case, sequence, obj_name='fissure' if not lobes else 'lobe')
            if not lobes and exclude_rhf:
                meshes = meshes[:2]
            self.meshes.append(meshes)
            self.ids.append((case, sequence))

            size, spacing = load_image_metadata(os.path.join(folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

        assert all(len(self.meshes[0]) == len(m) for m in self.meshes)
        self.num_objects = len(self.meshes[0])

    def __len__(self):
        return len(self.ids) * self.num_objects if self.fixed_object is None else len(self.ids)

    def __getitem__(self, item):
        meshes = self.meshes[self.continuous_to_pat_index(item)]
        obj_index = self.continuous_to_obj_index(item)
        current_mesh = meshes[obj_index]

        # sample point cloud from meshes
        samples = current_mesh.sample_points_uniformly(number_of_points=self.sample_points)
        samples = torch.from_numpy(np.asarray(samples.points)).float()

        # normalize to pytorch grid coordinates
        samples = self.normalize_sampled_pc(samples, item).transpose(0, 1)

        # get the target: either the PC itself or the GT mesh
        if self.mesh_as_target:
            target = self.normalize_mesh(o3d_to_pt3d_meshes([current_mesh]), item)
        else:
            target = samples

        return samples, target

    def normalize_sampled_pc(self, samples, index):
        return kpts_to_grid(samples, self.get_img_size(index)[::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_sampled_pc(self, samples, index):
        return kpts_to_world(samples, self.get_img_size(index)[::-1], align_corners=ALIGN_CORNERS)

    def normalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.normalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def unnormalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.unnormalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def get_batch_collate_fn(self):
        if self.mesh_as_target:
            def mesh_collate_fn(list_of_samples):
                pcs = torch.stack([pc for pc, _ in list_of_samples], dim=0)
                meshes = join_meshes_as_batch([mesh for _, mesh in list_of_samples])
                return pcs, meshes

            return mesh_collate_fn

        else:
            # no special collation function needed
            return None

    def continuous_to_pat_index(self, item):
        return item // self.num_objects if self.fixed_object is None else item

    def continuous_to_obj_index(self, item):
        return item % self.num_objects if self.fixed_object is None else self.fixed_object

    def get_id(self, item):
        return self.ids[self.continuous_to_pat_index(item)]

    def get_img_size(self, item):
        return self.img_sizes[self.continuous_to_pat_index(item)]

    def get_obj_mesh(self, item):
        return self.meshes[self.continuous_to_pat_index(item)][self.continuous_to_obj_index(item)]


def normalize_pc_zstd(pc):
    mu = pc.mean(dim=0, keepdim=True)
    sigma = pc.std()  # global stddev so that proportions don't get warped between axes
    pc = (pc - mu) / sigma
    return pc, mu, sigma


def unnormalize_zstd(pc, mu, sigma):
    return pc * sigma + mu


def test(ds: SampleFromMeshDS, device, out_dir, show):
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
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        point_cloud_on_axis(ax1, x.cpu(), 'b', title='input', label='input')
        if output_is_mesh:
            trimesh_on_axis(ax2, x_reconstruct.verts_padded().cpu().squeeze(), faces, facecolors=color_values, title='reconstruction', alpha=0.5, label='reconstruction')
        else:
            point_cloud_on_axis(ax2, x_reconstruct.cpu(), color_values, title='reconstruction')
        fig.savefig(os.path.join(plot_dir,
            f'{"_".join(ds.get_id(i))}_{"fissure" if not ds.lobes else "lobe"}{cur_obj+1}_reconstruction.png'),
            dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # compute average metrics
    mean_assd = all_mean_assd.mean(0, keepdim=True)
    std_assd = all_mean_assd.std(0, keepdim=True)

    mean_sdsd = all_mean_sdsd.mean(0, keepdim=True)
    std_sdsd = all_mean_sdsd.std(0, keepdim=True)

    mean_hd = all_hd_assd.mean(0, keepdim=True)
    std_hd = all_hd_assd.std(0, keepdim=True)

    mean_hd95 = all_hd95_assd.mean(0, keepdim=True)
    std_hd95 = all_hd95_assd.std(0, keepdim=True)

    dice_dummy = torch.zeros_like(all_mean_assd)

    write_results(os.path.join(out_dir, 'test_results.csv'), dice_dummy, dice_dummy,
                  all_mean_assd, std_assd, all_mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    return dice_dummy, dice_dummy, \
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


if __name__ == '__main__':
    parser = get_pc_ae_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    # configure dataset
    if args.ds == 'data':
        mesh_dir = '../data'
    elif args.ds == 'ts':
        mesh_dir = '../TotalSegmentator/ThoraxCrop'
    else:
        raise ValueError(f'No dataset named {args.ds}')

    ds = SampleFromMeshDS(mesh_dir, args.pts, fixed_object=args.obj, lobes=args.data == 'lobes', mesh_as_target=args.mesh)

    # configure model
    if not args.mesh and not args.loss == 'mesh':
        raise ValueError('Cannot compute mesh loss for non-mesh reconstructions of the AE.')

    model = DGCNNFoldingNet(k=args.k, n_embedding=args.latent, shape_type=args.shape, decode_mesh=args.mesh,
                            deform=args.deform)

    # store config
    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    # run the training
    run(ds, model, test, args)
