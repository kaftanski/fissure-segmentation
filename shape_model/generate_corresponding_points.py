import argparse
import csv
import glob
import os
import pickle
from typing import Sequence

import numpy as np
import open3d as o3d
import pytorch3d.loss
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import k_means, OPTICS

from data import ImageDataset
from metrics import point_surface_distance
from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
from shape_model.point_cloud_registration import inverse_transformation_at_sampled_points, \
    inverse_affine_transform
from shape_model.ssm import save_shape
from utils.detached_run import maybe_run_detached_cli
from utils.tqdm_utils import tqdm_redirect
from utils.utils import new_dir
from visualization import trimesh_on_axis, point_cloud_on_axis


def data_set_correspondences(fixed_pcs: np.ndarray,
                             all_moving_meshes: Sequence[Sequence[o3d.geometry.TriangleMesh]],
                             all_moving_pcs: np.ndarray, all_affine_transforms: np.ndarray, all_moved_pcs: np.ndarray,
                             all_displacements: np.ndarray, fixed_img_shape, plot_dir,
                             mode='cluster', undo_affine_reg=True, show=True, optics_minsamples_divisor=-1):

    corresponding_pcs = []
    corresponding_fixed_pc = []

    mean_p2m = []
    std_p2m = []
    hd_p2m = []
    cf_dists = []

    for obj_i in range(all_moved_pcs.shape[1]):
        all_points = np.concatenate(all_moved_pcs[:, obj_i], axis=0)

        if mode == 'kmeans':
            # k-means to determine sampling locations
            centroids, label, inertia = k_means(all_points, n_clusters=all_moved_pcs.shape[2])

        elif mode == 'cluster':
            eps_heuristic = (all_points.max() - all_points.min()) * 0.05
            cluster_estimator = OPTICS(min_samples=len(all_moving_meshes) // optics_minsamples_divisor, max_eps=eps_heuristic)
            clustering = cluster_estimator.fit_predict(all_points)

            # compute cluster centroids
            clusters = np.unique(clustering)
            centroids = np.zeros((len(clusters)-1, 3))
            for c in np.unique(clustering):
                # -1 values are considered outliers
                if c == -1:
                    continue
                centroids[c] = np.mean(all_points[clustering == c], axis=0)

        elif mode == 'parzen':
            raise NotImplementedError('Parzen mode is unfinished')
            # kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(all_points)  # TODO: check different bandwidths
            # density = kde.score_samples(all_points)
            #
            # if show:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     point_cloud_on_axis(ax, all_points, c=density, cmap='Blues', title='Point Cloud Density', label='')
            #     plt.show()
            #
            # def normalize(grid, img_shape):
            #     D, H, W = img_shape
            #     return (grid / torch.tensor([W - 1, H - 1, D - 1], device=grid.device)) * 2  # no -1 for displacements!

            # dense_density = thin_plate_dense(
            #     normalize(torch.from_numpy(all_points).unsqueeze(0), img_shape).float() - 1,
            #     torch.from_numpy(density).view(1, -1, 1),
            #     shape=img_shape[::-1], step=4, lambd=0.1)
            # dense_density = dense_density[..., 0].unsqueeze(1)
            # dense_density_nms = nms(dense_density, kernel_size=3)
            # top_k_val, top_k_ind = torch.topk(dense_density_nms, n_sample_points)
        elif mode == 'simple':
            # corresponding points by inverse transformation sampled at fixed points
            # -> define the fixed points as centroids
            centroids = fixed_pcs[obj_i]

        else:
            raise ValueError(f'No mode named {mode}.')

        # centroids are the new fixed PC
        corresponding_fixed_pc.append(centroids)

        # visualize clustering result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_cloud_on_axis(ax, fixed_pcs[obj_i], c='b', title='fixed vs. centroids', label='fixed pc')
        point_cloud_on_axis(ax, centroids, c='r', label='centroids')
        fig.savefig(os.path.join(plot_dir, f'clustering_obj{obj_i + 1}_centroids'), bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

        # compute inverse transformations at centroids for each moved point cloud which yields corresponding points
        corresponding_pcs.append([])
        mean_p2m.append([])
        std_p2m.append([])
        hd_p2m.append([])
        cf_dists.append([])
        for instance in tqdm_redirect(range(len(all_moved_pcs)), desc=f'inverse transform on centroids, object {obj_i+1}'):
            sampled_pc = inverse_transformation_at_sampled_points(
                all_moved_pcs[instance, obj_i], all_displacements[instance, obj_i],
                sample_point_cloud=centroids, img_shape=fixed_img_shape)

            # optionally undo affine transformation
            sampled_pc_not_affine = inverse_affine_transform(sampled_pc,
                                                             all_affine_transforms[instance, obj_i, :, :3].T,
                                                             all_affine_transforms[instance, obj_i, :, 3])
            if undo_affine_reg:
                corresponding_pcs[obj_i].append(sampled_pc_not_affine)
                # affine transform was undone, just the identity is left
                all_affine_transforms[instance, obj_i] = np.eye(3, 4)
            else:
                corresponding_pcs[obj_i].append(sampled_pc)

            # EVALUATION: compute surface distance between sampled moving and GT mesh!
            dist_moved = point_surface_distance(query_points=sampled_pc_not_affine,
                                                trg_points=all_moving_meshes[instance][obj_i].vertices,
                                                trg_tris=all_moving_meshes[instance][obj_i].triangles)
            p2m = dist_moved.mean()
            std = dist_moved.std()
            hd = dist_moved.max()
            print(f'Point Cloud distance to GT mesh: {p2m.item():.4f} +- {std.item():.4f} (Hausdorff: {hd.item():.4f})')
            cf, _ = pytorch3d.loss.chamfer_distance(torch.from_numpy(sampled_pc_not_affine[None]).float(),
                                                 torch.from_numpy(all_moving_pcs[instance, obj_i, None]).float())
            print(f'Chamfer Distance: {cf:.4f}')

            mean_p2m[obj_i].append(p2m)
            std_p2m[obj_i].append(std)
            hd_p2m[obj_i].append(hd)
            cf_dists[obj_i].append(cf)

            # visualize sampling and back-transformation
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            point_cloud_on_axis(ax, sampled_pc_not_affine, c='r', title='backtransformation', label='centroids-sampled_disp')
            point_cloud_on_axis(ax, all_moving_pcs[instance, obj_i], c='b', label='original moving')
            fig.savefig(os.path.join(plot_dir, f'instance{instance}_obj{obj_i+1}_backtrans'), bbox_inches='tight', dpi=300)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            point_cloud_on_axis(ax, sampled_pc_not_affine, c='r', cmap=None, label='New PC')
            trimesh_on_axis(ax, np.asarray(all_moving_meshes[instance][obj_i].vertices),
                            np.asarray(all_moving_meshes[instance][obj_i].triangles), color='b',
                            title='centroids - sampled displacements', alpha=0.5, label='GT mesh')
            fig.savefig(os.path.join(plot_dir, f'instance{instance}_obj{obj_i+1}_backtrans_gtmesh'), bbox_inches='tight', dpi=300)

            if show:
                plt.show()
            else:
                plt.close()

    # return np.stack(corresponding_pcs).swapaxes(0, 1), np.stack(all_affine_transforms).swapaxes(0, 1), \
    #     {'mean': torch.tensor(mean_p2m).T, 'std': torch.tensor(std_p2m).T, 'hd': torch.tensor(hd_p2m).T,
    #      'cf': torch.tensor(cf_dists).T}

    # concat because clustering with optics/dbscan yields different amount of points per object
    corresponding_fixed_pc = np.concatenate(corresponding_fixed_pc)
    labels = np.concatenate([np.full(len(corresponding_pcs[i][0]), i+1) for i in range(len(corresponding_pcs))])
    print('Corresponding point labels (counts):', np.unique(labels, return_counts=True))

    return np.concatenate(corresponding_pcs, axis=1), all_affine_transforms, corresponding_fixed_pc, labels, \
        {'mean': torch.tensor(mean_p2m).T, 'std': torch.tensor(std_p2m).T, 'hd': torch.tensor(hd_p2m).T,
         'cf': torch.tensor(cf_dists).T}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kmeans", "cluster", "simple"], default="simple",
                        help="method for determining correspondences")
    parser.add_argument("--ts", const=True, default=False, nargs="?", help="use total segmentator dataset")
    parser.add_argument("--undo_affine", const=True, default=False, nargs="?",
                        help="undo affine transformation of the point cloud registration")
    parser.add_argument("--show", const=True, default=False, nargs="?", help="show pyplot figures")
    parser.add_argument("--lobes", const=True, default=False, nargs="?", help="use lobes instead of fissure objects")
    parser.add_argument("--optics_divisor", type=int, default=16, help="divisor for OPTICS' min-samples parameter")
    parser.add_argument("--offline", const=True, default=False, nargs="?", help="run detached from pycharm")
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    ###### SETUP ######

    # mode = 'kmeans'
    # mode = 'cluster'
    # mode = 'simple'
    mode = args.mode

    lobes = args.lobes
    total_segmentator = args.ts
    undo_affine = args.undo_affine
    show = args.show

    optics_min_samples_divisor = args.optics_divisor

    # data set
    if total_segmentator:
        ds = TotalSegmentatorDataset()
    else:
        ds = ImageDataset("../data", do_augmentation=False, resample_spacing=1.)
    get_meshes = ds.get_fissure_meshes if not lobes else ds.get_lobe_meshes

    # set output path
    base_path = f'results/corresponding_points{"_ts" if total_segmentator else ""}/{"lobes" if lobes else "fissures"}'
    reg_dir = os.path.join(base_path, 'registrations')
    out_path = new_dir(base_path, mode)
    if mode == "cluster":
        out_path = new_dir(out_path, str(optics_min_samples_divisor))

    # load fixed point cloud
    fixed_fn = glob.glob(os.path.join(base_path, '*.npz'))
    assert len(fixed_fn) == 1
    fixed_pcs = np.load(fixed_fn[0], allow_pickle=True)['shape']
    f = ds.get_index(*os.path.split(fixed_fn[0])[1].replace('.npz', '').split('_'))

    # load all precomputed registrations
    def load(fname):
        return np.load(os.path.join(reg_dir, fname), allow_pickle=True)

    all_displacements = load('displacements.npz')
    all_moved_pcs = load('moved_pcs.npz')
    all_moving_pcs = load('moving_pcs.npz')
    all_affine_transforms = load('transforms.npz')
    ids = load('ids.npz')

    # remove any ids that might have been removed after the registration
    mask = np.ones(len(ids), dtype=bool)
    for i, loaded_id in enumerate(ids):
        try:
            ds.get_index(*loaded_id)
        except ValueError:
            mask[i] = False
    all_displacements = all_displacements[mask, ...]
    all_moved_pcs = all_moved_pcs[mask, ...]
    all_moving_pcs = all_moving_pcs[mask, ...]
    all_affine_transforms = all_affine_transforms[mask, ...]
    ids = ids[mask, ...]

    # load meshes
    moving_ids = [ds.get_id(m) for m in range(len(ds)) if m != f]  # prevent id mismatch
    assert np.all(np.array(moving_ids)==ids), 'Mismatch between current data set and registrations loaded.'
    moving_meshes = [get_meshes(m) for m in range(len(ds)) if m != f]

    corr_points, transforms, corr_fixed, labels, evaluation = data_set_correspondences(
        fixed_pcs, moving_meshes, all_moving_pcs, all_affine_transforms, all_moved_pcs, all_displacements,
        plot_dir=new_dir(out_path, 'plots'), fixed_img_shape=ds.get_fissures(f).GetSize()[::-1],
        mode=mode, show=show, undo_affine_reg=undo_affine, optics_minsamples_divisor=optics_min_samples_divisor)

    # output corresponding point clouds
    save_shape(corr_fixed, os.path.join(out_path, f'{"_".join(ds.get_id(f))}_corr_pts.npz'))

    for m in range(len(corr_points)):
        save_shape(corr_points[m], os.path.join(out_path, f'{"_".join(moving_ids[m])}_corr_pts.npz'), transforms=transforms[m])

    with open(os.path.join(out_path, 'labels.npz'), 'wb') as label_file:
        pickle.dump(labels, label_file)

    # evaluation results
    mean_p2m = evaluation['mean']
    std_p2m = evaluation['std']
    hd_p2m = evaluation['hd']
    cf_dists = evaluation['cf']

    print('\n====== RESULTS ======')
    print(f'P2M distance: {mean_p2m.mean(0)} +- {std_p2m.mean(0)} (Hausdorff: {hd_p2m.mean(0)}')

    with open(os.path.join(out_path, 'results.csv'), 'w') as csv_results_file:
        writer = csv.writer(csv_results_file)
        writer.writerow(['Affine-Pre-Reg', str(not undo_affine)])
        writer.writerow(['Object'] + [str(i+1) for i in range(mean_p2m.shape[1])] + ['mean'])
        writer.writerow(['mean p2m'] + [v.item() for v in mean_p2m.mean(0)] + [mean_p2m.mean().item()])
        writer.writerow(['std p2m'] + [v.item() for v in std_p2m.mean(0)] + [std_p2m.mean().item()])
        writer.writerow(['hd p2m'] + [v.item() for v in hd_p2m.mean(0)] + [hd_p2m.mean().item()])
        writer.writerow(['chamfer'] + [v.item() for v in cf_dists.mean(0)] + [cf_dists.mean().item()])
