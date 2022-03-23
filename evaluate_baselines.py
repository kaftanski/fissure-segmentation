import os
from glob import glob

import numpy as np
import open3d as o3d
import torch

from data import LungData
from metrics import assd
from train import compute_mesh_metrics, write_results


def evaluate_voxel2mesh(experiment_dir="/home/kaftan/FissureSegmentation/voxel2mesh-master/resultsExperiment_000"):
    def extract_id_from_fn(fn):
        case, sequence = fn.split('_')[-4:-2]
        sequence = sequence.replace('fix', 'fixed').replace('mov', 'moving')
        return case, sequence

    def _box_in_bounds(box, image_shape):
        """ from voxel2mesh utils.utils_common.py"""
        newbox = []
        pad_width = []

        for box_i, shape_i in zip(box, image_shape):
            pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
            newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

            newbox.append(newbox_i)
            pad_width.append(pad_width_i)

        needs_padding = any(i != (0, 0) for i in pad_width)

        return newbox, pad_width, needs_padding

    def crop_indices(image_shape, patch_shape, center):
        """ from voxel2mesh utils.utils_common.py"""
        box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
        box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
        slices = tuple(slice(i[0], i[1]) for i in box)
        return slices, pad_width, needs_padding

    ds = LungData('../data/')
    n_fissures = 2
    n_folds = 5

    # TODO: always make sure this is the same as in voxel2mesh config.py
    # patch_shape = (64, 64, 64)  # cfg.patch_shape
    largest_image_shape = (352, 352, 352)  # cfg.largest_image_shape

    test_assd = torch.zeros(n_folds, n_fissures)
    test_sdsd = torch.zeros_like(test_assd)
    test_hd = torch.zeros_like(test_assd)
    test_hd95 = torch.zeros_like(test_assd)

    for fold in range(n_folds):
        all_pred_meshes = []
        all_targ_meshes = []
        ids = []

        fold_dir = os.path.join(experiment_dir, f'trial_{fold+1}')
        mesh_dir = os.path.join(fold_dir, 'best_performance3', 'mesh')

        files_per_fissure = []
        for f in range(n_fissures):
            files_per_fissure.append(sorted(glob(os.path.join(mesh_dir, f'testing_pred_*_part_{f}.obj'))))

        for files in zip(*files_per_fissure):
            case, sequence = extract_id_from_fn(files[0])
            ids.append((case, sequence))

            # load the target meshes
            img_index = ds.get_index(case, sequence)
            target_meshes = ds.get_meshes(img_index)[:n_fissures]

            # load v2m predictions
            pred_meshes = [o3d.io.read_triangle_mesh(fn) for fn in files]

            # prepare undoing of normalization and padding
            img = ds.get_image(img_index)
            shape_unit_spacing = [int(sz * sp) for sz, sp in zip(img.GetSize()[::-1], img.GetSpacing()[::-1])]
            _, pad_width, _ = crop_indices(shape_unit_spacing, largest_image_shape, (s//2 for s in shape_unit_spacing))
            unpad_z, unpad_y, unpad_x = -pad_width[0][0], -pad_width[1][0], -pad_width[2][0]
            for i, (prediction, target) in enumerate(zip(pred_meshes, target_meshes)):
                # restore the xyz order for vertices
                verts_xyz = np.flip(np.asarray(prediction.vertices), axis=-1)

                # undo normalization and padding
                verts_xyz = (0.5 * (verts_xyz + 1)) * (max(largest_image_shape)-1) + np.asarray([unpad_z, unpad_y, unpad_x])
                prediction = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts_xyz),
                                                       triangles=prediction.triangles)

                pred_meshes[i] = prediction

            all_pred_meshes.append(pred_meshes)
            all_targ_meshes.append(target_meshes)

        # compute surface distances
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics(all_pred_meshes, all_targ_meshes, ids=ids, show=True)
        write_results(os.path.join(fold_dir, 'test_results.csv'), None, None, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

        test_assd[fold] += mean_assd
        test_sdsd[fold] += mean_sdsd
        test_hd[fold] += mean_hd
        test_hd95[fold] += mean_hd95

    # compute averages over cross-validation folds
    mean_assd = test_assd.mean(0)
    std_assd = test_assd.std(0)

    mean_sdsd = test_sdsd.mean(0)
    std_sdsd = test_sdsd.std(0)

    mean_hd = test_hd.mean(0)
    std_hd = test_hd.std(0)

    mean_hd95 = test_hd95.mean(0)
    std_hd95 = test_hd95.std(0)

    # print out results
    print('\n============ RESULTS ============')
    print(f'Mean ASSD per class: {mean_assd} +- {std_assd}')

    # output file
    write_results(os.path.join(experiment_dir, 'cv_results.csv'), None, None, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)


if __name__ == '__main__':
    evaluate_voxel2mesh()
