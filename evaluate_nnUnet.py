import csv
import os
import re
import time
from glob import glob

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

import constants
from data_processing.datasets import LungData, load_split_file
from data_processing.find_lobes import lobes_to_fissures
from data_processing.surface_fitting import poisson_reconstruction, o3d_mesh_to_labelmap, pointcloud_surface_fitting
from train_point_segmentation import compute_mesh_metrics, write_results
from utils.general_utils import find_test_fold_for_id, create_o3d_mesh, mask_out_verts_from_mesh, \
    remove_all_but_biggest_component
from utils.sitk_image_ops import sitk_image_to_tensor


def evaluate_nnunet(result_dir, my_data_dir, mode='surface', pts_subsample=10000, show=True, copd=False):
    assert mode in ['surface', 'voxels', 'subsample']

    ds = LungData(my_data_dir)
    n_folds = 5

    if copd:
        n_fissures = 2
    else:
        n_fissures = 3

    if mode == 'subsample':
        # rename mode for output filenames
        mode = mode + str(pts_subsample)
        torch.manual_seed(0)

    test_assd = torch.zeros(n_folds, n_fissures)
    test_sdsd = torch.zeros_like(test_assd)
    test_hd = torch.zeros_like(test_assd)
    test_hd95 = torch.zeros_like(test_assd)
    test_missing_percent = torch.zeros_like(test_assd)
    all_times = []
    all_pre_proc_times = []

    all_files = sorted(glob(os.path.join(result_dir, 'cv_niftis_postprocessed' if not copd else '*', '*.nii.gz')))
    ids_per_fold = {fold: [] for fold in range(n_folds) }
    files_per_fold = {fold: [] for fold in range(n_folds)}
    for f in all_files:
        filename = f.split(os.sep)[-1]
        if not copd:
            case, sequence = filename.split('_')[-2:]
            sequence = sequence.replace('fix', 'fixed').replace('mov', 'moving').replace('.nii.gz', '')

            fold = find_test_fold_for_id(case, sequence, load_split_file(constants.DEFAULT_SPLIT_TS))
            ids_per_fold[fold].append((case, sequence))
            files_per_fold[fold].append(f)

        else:
            match = re.match('COPD[0-1][0-9]', filename)
            case = match.group(0)
            sequence = filename.replace(case, "")[0]
            if sequence == "f":
                sequence = "fixed"
            elif sequence == "m":
                sequence = "moving"
            else:
                raise ValueError(f'No sequence for char "{sequence}"')

            for fold in range(n_folds):
                ids_per_fold[fold].append((case, sequence))
                files_per_fold[fold].append(f)

    for fold in range(n_folds):
        print('\n' + '='*30)
        print(f'Fold {fold}')
        print('='*30)
        fold_dir = os.path.join(result_dir, f'fold_{fold}')
        # files = sorted(glob(os.path.join(fold_dir, 'validation_raw_postprocessed' if not copd else '', '*.nii.gz')))
        files = files_per_fold[fold]

        mesh_dir = os.path.join(fold_dir, 'validation_mesh_reconstructions')
        plot_dir = os.path.join(fold_dir, f'plots_{mode}')
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        all_predictions = []
        all_targ_meshes = []

        all_times_fold = []
        all_pre_proc_times_fold = []

        ids = ids_per_fold[fold]
        spacings = []
        for i, f in enumerate(files):
            case, sequence = ids[i]
            print(case, sequence)

            img_index = ds.get_index(case, sequence)

            labelmap_predict = sitk.ReadImage(f)
            if 'Lobes' in result_dir:  # results are lobes -> convert them to fissures first
                labelmap_predict, _ = lobes_to_fissures(labelmap_predict, mask=ds.get_lung_mask(img_index))

            if mode == 'surface':
                _, predicted_meshes, times = poisson_reconstruction(labelmap_predict, ds.get_lung_mask(img_index), return_times=True, mask_dilate_radius=1)
                all_times_fold.append(torch.tensor(times))
                for i, m in enumerate(predicted_meshes):
                    # save reconstructed mesh
                    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, f'{case}_fissure{i+1}_{sequence}.obj'), m)

                all_predictions.append(predicted_meshes)
            elif mode == 'voxels':
                fissure_tensor = torch.from_numpy(sitk.GetArrayFromImage(labelmap_predict).astype(int))
                predicted_fissures = [fissure_tensor == f for f in fissure_tensor.unique()[1:]]
                # predicted_fissures = [torch.from_numpy(skeletonize_3d(fissure_tensor == f) > 0) for f in fissure_tensor.unique()[1:]]
                all_predictions.append(predicted_fissures)
                spacings.append(labelmap_predict.GetSpacing())
                all_times_fold.append(torch.tensor([-1., -1.]))

                # # re-assemble labelmap
                # fissure_skeleton = torch.zeros_like(predicted_fissures[0], dtype=torch.long)
                # for skel in predicted_fissures:
                #     fissure_skeleton += fissure_tensor * skel
                # img = sitk.GetImageFromArray(fissure_skeleton.numpy().astype(np.uint8))
                # img.CopyInformation(fissures_predict)
                # sitk.WriteImage(img, f'./results/nnunet_pred_skeletonized_{case}_{sequence}.nii.gz')
            elif mode == 'subsample' + str(pts_subsample):
                labelmap_predict_tensor = sitk_image_to_tensor(labelmap_predict)

                # get mask
                mask_img = ds.get_lung_mask(img_index)
                mask_img = sitk.BinaryDilate(mask_img, (1,)*3)  # no difference in results
                mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask_img).astype(bool))

                # get spacing and world-shape
                spacing = torch.tensor(labelmap_predict.GetSpacing())
                shape = torch.tensor(labelmap_predict.GetSize()[::-1]) * spacing.flip(0)

                # preprocess labelmap
                start = time.time()
                indices_foreground = torch.nonzero(labelmap_predict_tensor)
                labelled_points = indices_foreground.flip(-1).float() * spacing  # in mm
                point_labels = labelmap_predict_tensor[indices_foreground.split(1,1)].squeeze()

                # subsample points
                n_points = min(pts_subsample, labelled_points.shape[0])
                random_selection = torch.randperm(labelled_points.shape[0])[:n_points]
                subsampled_points = labelled_points[random_selection]
                subsampled_labels = point_labels[random_selection]
                preproc_time = time.time() - start
                all_pre_proc_times_fold.append(preproc_time)

                # reconstruct fissure surface from subsapmled points
                predicted_fissures = []
                times = []
                for i in range(n_fissures):
                    start = time.time()

                    # get points for current fissure
                    points_per_fissure = subsampled_points[subsampled_labels==i+1]

                    # fit surface to points (with post-processing)
                    try:
                        mesh_predict = pointcloud_surface_fitting(points_per_fissure, crop_to_bbox=True)

                        mask_out_verts_from_mesh(mesh_predict, mask_tensor, spacing)  # apply lung mask
                        right = i > 1  # right fissure(s) are label 2 and 3
                        remove_all_but_biggest_component(mesh_predict, right=right,
                                                         center_x=shape[2] / 2)  # keep only largest component

                        predicted_fissures.append(mesh_predict)

                    except ValueError as e:
                        # no points have been predicted to be in this class
                        print(e)
                        predicted_fissures.append(create_o3d_mesh(verts=np.array([]), tris=np.array([])))

                    times.append(time.time() - start)

                all_times_fold.append(torch.tensor(times))
                all_predictions.append(predicted_fissures)

                # get the number of points per fissure and write into a csv-file
                n_per_fissure = subsampled_labels.unique(return_counts=True)[1].tolist()
                n_per_fissure_orig = point_labels.unique(return_counts=True)[1].tolist()
                n_points_file = os.path.join(mesh_dir, f'n_points_per_fissure_{mode}.csv')
                if not os.path.isfile(n_points_file):
                    with open(n_points_file, 'w') as f:
                        writer = csv.writer(f)

                        # write header
                        writer.writerow(['case', 'sequence'] + [f'fissure_{i+1}_subsampled' for i in range(n_fissures)] + [f'fissure_{i+1}_total' for i in range(n_fissures)])

                        # write data
                        writer.writerow([case, sequence] + list(map(str, n_per_fissure)) + list(map(str, n_per_fissure_orig)))
                else:
                    with open(n_points_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([case, sequence] + list(map(str, n_per_fissure)) + list(map(str, n_per_fissure_orig)))

            else:
                raise ValueError(f'Unknown mode "{mode}"')

            target_meshes = ds.get_fissure_meshes(img_index)[:n_fissures]
            all_targ_meshes.append(target_meshes)

            if all_pre_proc_times_fold:
                print(f'Current mean time for pre-processing: {torch.tensor(all_pre_proc_times_fold).mean():4f}s +- {torch.tensor(all_pre_proc_times_fold).std():4f}s')
            print(f'Current mean time per image: {torch.stack(all_times_fold).sum(1).mean():4f}s +- {torch.stack(all_times_fold).sum(1).std():4f}s')

        # compute surface distances
        mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing = compute_mesh_metrics(
            all_predictions, all_targ_meshes, ids=ids, show=show, spacings=spacings, plot_folder=plot_dir)
        write_results(os.path.join(mesh_dir, f'test_results_{mode}.csv'), None, None, mean_assd, std_assd, mean_sdsd,
                      std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing)

        test_assd[fold] += mean_assd
        test_sdsd[fold] += mean_sdsd
        test_hd[fold] += mean_hd
        test_hd95[fold] += mean_hd95
        test_missing_percent[fold] += percent_missing

        # write times
        with open(os.path.join(mesh_dir, f'inference_time_node2_{mode}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Pre-proc', 'Pre-proc_std', 'Mesh-rec', 'Mesh-rec_std'])
            writer.writerow([torch.tensor(all_pre_proc_times_fold).mean().item(), torch.tensor(all_pre_proc_times_fold).std().item(),
                             torch.stack(all_times_fold).sum(1).mean().item(), torch.stack(all_times_fold).sum(1).std().item()])

        all_times.extend(all_times_fold)
        all_pre_proc_times.extend(all_pre_proc_times_fold)

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
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, test_missing_percent.mean(0))

    # write times
    with open(os.path.join(result_dir, f'inference_time_node2_{mode}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Pre-proc', 'Pre-proc_std', 'Mesh-rec', 'Mesh-rec_std'])
        writer.writerow(
            [torch.tensor(all_pre_proc_times).mean().item(), torch.tensor(all_pre_proc_times).std().item(),
             torch.stack(all_times).sum(1).mean().item(), torch.stack(all_times).sum(1).std().item()])


if __name__ == '__main__':
    n_fissures = 3

    data_dir = constants.IMG_DIR_TS_PREPROC

    # trained nnU-net paths
    nnu_results_dir = '../nnUNet/output/nnu_results'
    nnu_task = "Task503_FissuresTotalSeg"
    nnu_trainer = "nnUNetTrainerV2_200ep"
    nnu_result_dir = f'{nnu_results_dir}/nnUNet/3d_fullres/{nnu_task}/{nnu_trainer}__nnUNetPlansv2.1'

    # run evaluation
    # evaluate_nnunet(nnu_result_dir, my_data_dir=data_dir, mode='voxels', show=False)
    evaluate_nnunet(nnu_result_dir, my_data_dir=data_dir, mode='surface', pts_subsample=10000, show=False)
    evaluate_nnunet(nnu_result_dir, my_data_dir=data_dir, mode='subsample', pts_subsample=10000, show=False)

    copd_data_dir = constants.IMG_DIR_COPD
    nnu_copd_result_dir = "../nnUNet/output/copd_pred"
    evaluate_nnunet(nnu_copd_result_dir, my_data_dir=copd_data_dir, mode='surface', show=False, copd=True)
    # evaluate_nnunet(nnu_copd_result_dir, my_data_dir=copd_data_dir, mode='voxels', show=False, copd=True)
