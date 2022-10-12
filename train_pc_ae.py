import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from cli.cl_args import get_pc_ae_train_parser
from data import CustomDataset
from losses.chamfer_loss import ChamferLoss
from metrics import pseudo_symmetric_point_to_mesh_distance
from models.folding_net import DGCNNFoldingNet
from train import run, write_results
from utils.image_ops import load_image_metadata
from utils.utils import load_meshes, new_dir, kpts_to_grid, ALIGN_CORNERS, kpts_to_world
from visualization import point_cloud_on_axis


class SampleFromMeshDS(CustomDataset):
    def __init__(self, folder, sample_points, fixed_object: int=0, lobes=False, return_normalization_params=False):
        super(SampleFromMeshDS, self).__init__(exclude_rhf=False, binary=False, do_augmentation=True)

        self.sample_points = sample_points
        self.fixed_object = fixed_object
        self.return_norm_params = return_normalization_params

        mesh_dirs = sorted(glob.glob(os.path.join(folder, "*_mesh_*")))

        self.ids = []
        self.meshes = []
        self.img_sizes = []
        for mesh_dir in mesh_dirs:
            case, sequence = os.path.basename(mesh_dir).split("_mesh_")
            meshes = load_meshes(folder, case, sequence, obj_name='fissure' if not lobes else 'lobe')
            self.meshes.append(meshes)
            self.ids.append((case, sequence))

            size, spacing = load_image_metadata(os.path.join(folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        meshes = self.meshes[item]

        if self.fixed_object is not None:
            meshes = [meshes[self.fixed_object]]

        # sample point cloud from meshes
        samples_per_component = []
        for m in meshes:
            samples = m.sample_points_uniformly(number_of_points=self.sample_points)
            samples = torch.from_numpy(np.asarray(samples.points)).float()

            # normalize to pytorch grid coordinates
            samples = self.normalize_sampled_pc(samples, item)
            samples_per_component.append(samples.transpose(0, 1))

        return samples_per_component[self.fixed_object], samples_per_component[self.fixed_object]

    def normalize_sampled_pc(self, samples, index):
        return kpts_to_grid(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_sampled_pc(self, samples, index):
        return kpts_to_world(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)


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

    ds.return_norm_params = True

    pred_dir = new_dir(out_dir, 'test_predictions')
    plot_dir = new_dir(pred_dir, 'plots')

    # visualize folding points
    folding_points = model.decoder.get_folding_points(1).squeeze()
    if folding_points.shape[1] == 2:  # plane mode
        # add the third coordinate to enable 3d plot
        folding_points = torch.cat([folding_points, torch.zeros(folding_points.shape[0], 1)], dim=1)
    color_values = folding_points.sum(-1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    point_cloud_on_axis(ax1, folding_points, c=color_values, cmap='viridis', title="folding points")
    fig.savefig(os.path.join(plot_dir, 'folding_points.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)

    chamfer_dists = []
    mean_assd = []
    mean_sdsd = []
    hd_assd = []
    hd95_assd = []
    criterion = ChamferLoss()
    for i in range(len(ds)):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)

        with torch.no_grad():
            x_reconstruct = model(x)

        # unnormalize point cloud
        x = ds.unnormalize_sampled_pc(x, i)
        x_reconstruct = ds.unnormalize_sampled_pc(x_reconstruct, i)

        # compute chamfer distance
        chamfer_dists.append(criterion(x_reconstruct, x).item())

        # reorder points to (N x 3)
        x = x.squeeze().transpose(0, 1)
        x_reconstruct = x_reconstruct.squeeze().transpose(0, 1)

        # compute surface distance
        mean, std, hd, hd95 = pseudo_symmetric_point_to_mesh_distance(x_reconstruct.cpu(), ds.meshes[i][ds.fixed_object])
        mean_assd.append(mean)
        mean_sdsd.append(std)
        hd_assd.append(hd)
        hd95_assd.append(hd95)

        # TODO: make prediction as a mesh

        # visualize reconstruction
        fig = plt.figure()
        ax1 = fig.add_subplot(121, projection='3d')
        point_cloud_on_axis(ax1, x.cpu(), 'b', title='input')
        ax2 = fig.add_subplot(122, projection='3d')
        point_cloud_on_axis(ax2, x_reconstruct.cpu(), color_values, cmap='viridis', title='reconstruction')
        fig.savefig(os.path.join(plot_dir, f'{"_".join(ds.ids[i])}_reconstruction.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # compute average metrics  # TODO compute for multiple labels
    mean_assd = torch.tensor(mean_assd).unsqueeze(1)
    mean_sdsd = torch.tensor(mean_sdsd).unsqueeze(1)
    hd_assd = torch.tensor(hd_assd).unsqueeze(1)
    hd95_assd = torch.tensor(hd95_assd).unsqueeze(1)

    mean_assd = mean_assd.mean(0, keepdim=True)
    std_assd = mean_assd.std(0, keepdim=True)

    mean_sdsd = mean_sdsd.mean(0, keepdim=True)
    std_sdsd = mean_sdsd.std(0, keepdim=True)

    mean_hd = hd_assd.mean(0, keepdim=True)
    std_hd = hd_assd.std(0, keepdim=True)

    mean_hd95 = hd95_assd.mean(0, keepdim=True)
    std_hd95 = hd95_assd.std(0, keepdim=True)

    dice_dummy = torch.zeros_like(mean_assd)

    write_results(os.path.join(out_dir, 'test_results.csv'), dice_dummy, dice_dummy,
                  mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    return dice_dummy, dice_dummy, \
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


if __name__ == '__main__':
    parser = get_pc_ae_train_parser()
    args = parser.parse_args()

    model = DGCNNFoldingNet(k=args.k, n_embedding=args.latent, shape_type=args.shape)

    ds = SampleFromMeshDS('../data', args.pts, fixed_object=0,  # TODO: just one fissure for now
                          lobes=args.data == 'lobes')

    run(ds, model, test, args)
