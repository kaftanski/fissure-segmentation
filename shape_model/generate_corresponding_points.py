import csv
import os
from typing import Sequence

import numpy as np
import open3d as o3d
import pytorch3d.loss
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import k_means, OPTICS

from data import ImageDataset
from metrics import point_surface_distance
from shape_model.point_cloud_registration import register_cpd, visualize, inverse_transformation_at_sampled_points, \
    inverse_affine_transform
from shape_model.ssm import save_shape
from visualization import trimesh_on_axis, point_cloud_on_axis


def data_set_correspondences(fixed_pcs: Sequence[o3d.geometry.PointCloud],
                             all_moving_meshes: Sequence[Sequence[o3d.geometry.TriangleMesh]],
                             img_shape, n_sample_points=1024, mode='cluster', undo_affine_reg=True, show=True):
    all_affine_transforms = []
    all_moving_pcs = []
    corresponding_pcs = []

    mean_p2m = []
    std_p2m = []
    hd_p2m = []
    cf_dists = []

    # register objects separately (TODO: joint registration?)
    for obj_i, fixed in enumerate(fixed_pcs):
        moved_pcs = []
        moving_displacements = []
        all_affine_transforms.append([])
        all_moving_pcs.append([])
        fixed_pc_np = np.asarray(fixed.points)

        # register all moving objects into fixed space
        for moving in all_moving_meshes:
            # sample points evenly from mesh
            moving_pc = moving[obj_i].sample_points_poisson_disk(number_of_points=n_sample_points)
            moving_pc_np = np.asarray(moving_pc.points)
            all_moving_pcs[obj_i].append(moving_pc_np)

            # register and check result
            affine_prereg, affine_mat, affine_translation, deformed_pc_np, displacements = register_cpd(
                fixed_pc_np, moving_pc_np)
            deformed_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(deformed_pc_np.astype(np.float32)))
            reg_result = o3d.pipelines.registration.evaluate_registration(source=deformed_pc, target=fixed,
                                                                          max_correspondence_distance=10)
            print(reg_result)

            moved_pcs.append(deformed_pc_np)
            moving_displacements.append(displacements)

            # remember affine transformation for later
            # (pycpd defines affine transformation as x*A -> we transpose the matrix so we can use A^T*x)
            all_affine_transforms[obj_i].append(np.concatenate([affine_mat.T, affine_translation[:, np.newaxis]], axis=1))

            if show:
                # VISUALIZATION
                # show registration steps
                fig = plt.figure()
                ax1 = fig.add_subplot(131, projection='3d')
                ax2 = fig.add_subplot(132, projection='3d')
                ax3 = fig.add_subplot(133, projection='3d')

                visualize('Initial PC', fixed_pc_np, moving_pc_np, ax1)
                visualize('Affine', fixed_pc_np, affine_prereg, ax2)
                visualize('Deformable', fixed_pc_np, deformed_pc_np, ax3)

                plt.show()

        all_points = np.concatenate(moved_pcs, axis=0)
        if mode == 'kmeans':
            # k-means to determine sampling locations
            centroids, label, inertia = k_means(all_points, n_clusters=n_sample_points)

        elif mode == 'cluster':
            eps_heuristic = (all_points.max() - all_points.min()) * 0.05
            cluster_estimator = OPTICS(min_samples=len(all_moving_meshes), max_eps=eps_heuristic)
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
        else:
            raise ValueError(f'No mode named {mode}.')

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            point_cloud_on_axis(ax, fixed_pc_np, c='g', title='fixed vs. centroids', label='fixed pc')
            point_cloud_on_axis(ax, centroids, c='b', label='centroids')
            plt.show()

        # compute inverse transformations at centroids for each moved point cloud which yields corresponding points
        corresponding_pcs.append([])
        mean_p2m.append([])
        std_p2m.append([])
        hd_p2m.append([])
        cf_dists.append([])
        for instance in range(len(moved_pcs)):
            sampled_pc = inverse_transformation_at_sampled_points(moved_pcs[instance],
                                                                  moving_displacements[instance],
                                                                  sample_point_cloud=centroids, img_shape=img_shape)

            # optionally undo affine transformation
            sampled_pc_not_affine = inverse_affine_transform(sampled_pc, all_affine_transforms[obj_i][instance][:, :3].T,
                                                             all_affine_transforms[obj_i][instance][:, 3])
            if undo_affine_reg:
                corresponding_pcs[obj_i].append(sampled_pc_not_affine)
                # affine transform was undone, just the identity is left
                all_affine_transforms[obj_i][instance] = np.eye(3, 4)
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
                                                 torch.from_numpy(all_moving_pcs[obj_i][instance][None]).float())
            print(f'Chamfer Distance: {cf:.4f}')

            mean_p2m[obj_i].append(p2m)
            std_p2m[obj_i].append(std)
            hd_p2m[obj_i].append(hd)
            cf_dists[obj_i].append(cf)

            if show:
                # visualize sampling and back-transformation
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                point_cloud_on_axis(ax, sampled_pc_not_affine, c='r', title='backtransformation', label='centroids-sampled_disp')
                point_cloud_on_axis(ax, all_moving_pcs[obj_i][instance], c='b', label='original moving')

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                point_cloud_on_axis(ax, sampled_pc_not_affine, c='r', cmap=None, label='New PC')
                trimesh_on_axis(ax, np.asarray(all_moving_meshes[instance][obj_i].vertices),
                                np.asarray(all_moving_meshes[instance][obj_i].triangles), color='b',
                                title='centroids - sampled displacements', alpha=0.5, label='GT mesh')

                plt.show()

        # compile all transformations
        all_affine_transforms[obj_i] = np.stack(all_affine_transforms[obj_i], axis=0)

    labels = np.array([1] * len(corresponding_pcs[0]) + [2] * len(corresponding_pcs[1]))
    print(len(corresponding_pcs[0]), len(corresponding_pcs[1]))
    return np.concatenate(corresponding_pcs, axis=1), np.stack(all_affine_transforms).swapaxes(0, 1), labels, \
        {'mean': torch.tensor(mean_p2m).T, 'std': torch.tensor(std_p2m).T, 'hd': torch.tensor(hd_p2m).T,
         'cf': torch.tensor(cf_dists).T}
    # return np.stack(corresponding_pcs).swapaxes(0, 1), np.stack(all_affine_transforms).swapaxes(0, 1), \


if __name__ == '__main__':
    # mode = 'kmeans'
    mode = 'cluster'
    # mode = 'simple'

    out_path = f'results/corresponding_points_{mode}'
    os.makedirs(out_path, exist_ok=True)
    undo_affine = False
    n_sample_points = 1024

    ds = ImageDataset("../data", do_augmentation=False, resample_spacing=1.)

    f = 0
    sequence = ds.get_id(f)[1]
    # assert sequence == 'fixed'
    fixed_meshes = ds.get_fissure_meshes(f)
    img_fixed, _ = ds[f]

    # sample points from fixed
    fixed_pcs = [mesh.sample_points_poisson_disk(number_of_points=n_sample_points) for mesh in fixed_meshes]
    fixed_pcs_np = np.stack([pc.points for pc in fixed_pcs])
    save_shape(fixed_pcs_np, os.path.join(out_path, f'{"_".join(ds.get_id(f))}_corr_pts.npz'), transforms=None)

    # register each image onto fixed
    moving_meshes = [ds.get_fissure_meshes(m) for m in range(len(ds)) if m != f][-10:-5]
    corr_points, transforms, labels, evaluation = data_set_correspondences(
        fixed_pcs, moving_meshes, img_shape=img_fixed.shape, mode=mode, show=True, undo_affine_reg=undo_affine)

    mean_p2m = evaluation['mean']
    std_p2m = evaluation['std']
    hd_p2m = evaluation['hd']
    cf_dists = evaluation['cf']

    # output results
    for m in range(len(corr_points)):
        save_shape(corr_points[m], os.path.join(out_path, f'{"_".join(ds.get_id(m))}_corr_pts.npz'), transforms=transforms[m])

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
