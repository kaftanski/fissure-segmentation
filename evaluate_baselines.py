import os
from glob import glob

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from skimage.morphology import skeletonize_3d

from data import LungData
from data_processing.surface_fitting import poisson_reconstruction
from train import compute_mesh_metrics, write_results
from utils import remove_all_but_biggest_component


def evaluate_voxel2mesh(experiment_dir="/home/kaftan/FissureSegmentation/voxel2mesh-master/resultsExperiment_000", show=True):
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
        mesh_dir = os.path.join(fold_dir, 'best_performance', 'mesh')

        files_per_fissure = []
        for f in range(n_fissures):
            meshes = sorted(glob(os.path.join(mesh_dir, f'testing_pred_*_part_{f}.obj')))
            if not meshes:
                files_per_fissure.append(sorted(glob(os.path.join(mesh_dir.replace('best_performance', 'best_performance3'), f'testing_pred_*_part_{f}.obj'))))
            else:
                files_per_fissure.append(meshes)

        for files in zip(*files_per_fissure):
            case, sequence = files[0].split('_')[-4:-2]
            sequence = sequence.replace('fix', 'fixed').replace('mov', 'moving')
            ids.append((case, sequence))

            # load the target meshes
            img_index = ds.get_index(case, sequence)
            target_meshes = ds.get_fissure_meshes(img_index)[:n_fissures]

            # load v2m predictions
            pred_meshes = [o3d.io.read_triangle_mesh(fn) for fn in files]

            # prepare undoing of normalization and padding
            img = ds.get_image(img_index)
            shape_unit_spacing = [int(sz * sp) for sz, sp in zip(img.GetSize()[::-1], img.GetSpacing()[::-1])]
            _, pad_width, _ = crop_indices(shape_unit_spacing, largest_image_shape, (s//2 for s in shape_unit_spacing))
            unpad_z, unpad_y, unpad_x = -pad_width[0][0], -pad_width[1][0], -pad_width[2][0]
            for i, (prediction, target) in enumerate(zip(pred_meshes, target_meshes)):
                verts = np.asarray(prediction.vertices)

                # undo normalization and padding
                verts = (0.5 * (verts + 1)) * (max(largest_image_shape)-1) + np.asarray([unpad_z, unpad_y, unpad_x])
                prediction = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts),
                                                       triangles=prediction.triangles)

                pred_meshes[i] = prediction

            all_pred_meshes.append(pred_meshes)
            all_targ_meshes.append(target_meshes)

        # compute surface distances
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics(all_pred_meshes, all_targ_meshes, ids=ids, show=show)
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


def evaluate_nnunet(result_dir='/home/kaftan/FissureSegmentation/nnUNet_baseline/nnu_results/nnUNet/3d_fullres/Task501_FissureCOPDEMPIRE/nnUNetTrainerV2__nnUNetPlansv2.1',
                    mode='surface', show=True):
    assert mode in ['surface', 'voxels']

    ds = LungData('../data')
    n_folds = 5
    n_fissures = 2

    test_assd = torch.zeros(n_folds, n_fissures)
    test_sdsd = torch.zeros_like(test_assd)
    test_hd = torch.zeros_like(test_assd)
    test_hd95 = torch.zeros_like(test_assd)
    for fold in range(n_folds):
        fold_dir = os.path.join(result_dir, f'fold_{fold}')
        files = sorted(glob(os.path.join(fold_dir, 'validation_raw_postprocessed', '*.nii.gz')))

        mesh_dir = os.path.join(fold_dir, 'validation_mesh_reconstructions')
        os.makedirs(mesh_dir, exist_ok=True)

        all_predictions = []
        all_targ_meshes = []
        ids = []
        spacings = []
        for f in files:
            case, sequence = f.split(os.sep)[-1].split('_')[-2:]
            sequence = sequence.replace('fix', 'fixed').replace('mov', 'moving').replace('.nii.gz', '')
            ids.append((case, sequence))

            img_index = ds.get_index(case, sequence)
            target_meshes = ds.get_fissure_meshes(img_index)[:n_fissures]
            all_targ_meshes.append(target_meshes)

            fissures_predict = sitk.ReadImage(f)
            if mode == 'surface':
                _, predicted_meshes = poisson_reconstruction(fissures_predict, ds.get_lung_mask(img_index))
                # TODO: compare Poisson to Marching Cubes mesh generation
                # TODO: try skimage skeletonize3d instead of sitk.BinaryThinning
                for i, m in enumerate(predicted_meshes):
                    # post-process
                    remove_all_but_biggest_component(m)

                    # save reconstructed mesh
                    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, f'{case}_fissure{i+1}_{sequence}.obj'), m)

                all_predictions.append(predicted_meshes)
            else:
                fissure_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures_predict).astype(int))
                predicted_fissures = [fissure_tensor == f for f in fissure_tensor.unique()[1:]]
                # predicted_fissures = [torch.from_numpy(skeletonize_3d(fissure_tensor == f) > 0) for f in fissure_tensor.unique()[1:]]
                all_predictions.append(predicted_fissures)
                spacings.append(fissures_predict.GetSpacing())

                # # re-assemble labelmap
                # fissure_skeleton = torch.zeros_like(predicted_fissures[0], dtype=torch.long)
                # for skel in predicted_fissures:
                #     fissure_skeleton += fissure_tensor * skel
                # img = sitk.GetImageFromArray(fissure_skeleton.numpy().astype(np.uint8))
                # img.CopyInformation(fissures_predict)
                # sitk.WriteImage(img, f'./results/nnunet_pred_skeletonized_{case}_{sequence}.nii.gz')

        # compute surface distances
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics(
            all_predictions, all_targ_meshes, ids=ids, show=show, spacings=spacings)
        write_results(os.path.join(mesh_dir, f'test_results_{mode}.csv'), None, None, mean_assd, std_assd, mean_sdsd,
                      std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

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
    write_results(os.path.join(result_dir, f'cv_results_{mode}.csv'), None, None, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)


if __name__ == '__main__':
    # evaluate_voxel2mesh(show=False)
    # evaluate_nnunet(mode='surface', show=False)
    evaluate_nnunet(mode='voxels', show=False)
