import os.path

import SimpleITK as sitk
import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt

from constants import DEFAULT_SPLIT_TS, IMG_DIR_TS, IMG_DIR_TSv1, CLASS_COLORS, CLASSES
from data import load_split_file, ImageDataset
from preprocess_totalsegmentator_dataset import find_non_zero_ranges
from thesis.utils import legend_figure, save_fig, textwidth_to_figsize
from utils.general_utils import find_test_fold_for_id
from visualization import visualize_with_overlay

slices = {'s0070': [70, 180]}
result_folder = 'results/plots/qualitative'


def create_image_figure(img):
    return plt.figure(figsize=textwidth_to_figsize(2, aspect=img.shape[0] / img.shape[1]))


def slice_image(img_3d, slice_num, slice_dim):
    index = [slice(None)] * slice_dim + [slice(slice_num, slice_num + 1)]
    return img_3d[index].squeeze(), index


def fissure_window_level_and_mask(img: np.ndarray, mask: np.ndarray, high=-600):
    low = -1024
    img[img < low] = low
    img[img > high] = high
    img[mask == 0] = high + 1
    return img


def crop_to_lung_indices(img):
    max_val = img.max()
    ranges = find_non_zero_ranges((img != max_val)[None])
    return slice(ranges[0, 0], ranges[0, 1]+1), slice(ranges[1, 0], ranges[1, 1]+1), slice(ranges[2, 0], ranges[2, 1]+1)


def multi_model_overlay(img, label_maps, slice_num, slice_dim=2, fig_name='keypoint_qualitative_comparison', patid='s0070'):
    """

    :param img:
    :param label_maps: viewed as binary label maps
    :param slice_num:
    :param slice_dim:
    :return:
    """
    # slice index
    index = [slice(None)] * slice_dim + [slice(slice_num, slice_num + 1)]
    img_slice = img[index]

    # assemble label map
    combined_label = np.zeros_like(img_slice, dtype=int)
    models = []
    for i, (model, label) in enumerate(label_maps.items()):
        label = sitk.GetArrayFromImage(label)
        combined_label[label[index] != 0] = i + 1
        models.append(model)

    fig = create_image_figure(img_slice)

    colors = matplotlib.cm.get_cmap('tab10').colors
    visualize_with_overlay(img_slice.squeeze(), combined_label.squeeze(), alpha=0.6, colors=colors, ax=fig.gca())

    save_fig(fig, 'results/plots/qualitative', f'{fig_name}_{patid}_slice{slice_num}', pdf=False)
    legend_figure(labels=list(label_maps.keys()), colors=colors,
                  outdir='results/plots/qualitative', basename=f'{fig_name}_legend')

    # plot the slice without labels
    fig = create_image_figure(img_slice)
    visualize_with_overlay(img_slice.squeeze(), np.zeros_like(combined_label.squeeze()), ax=fig.gca())
    save_fig(fig, 'results/plots/qualitative', f'{patid}_slice{slice_num}', pdf=False)


def multi_class_overlay(img, label_map, model_name, patid, slice_dim=2):
    """

    :param img:
    :param label_maps: viewed as binary label maps
    :param slice_num:
    :param slice_dim:
    :return:
    """

    # assemble label map
    label_map = sitk.GetArrayFromImage(label_map)

    # crop the interesting image region
    crop_indices = crop_to_lung_indices(img)
    img = img[crop_indices]
    label_map = label_map[crop_indices]
    for slice_num in slices[patid]:
        # slice index
        slice_num_cropped = slice_num - crop_indices[slice_dim].start
        img_slice, index = slice_image(img, slice_num_cropped, slice_dim)

        label_slice = label_map[index].squeeze()

        # figure
        fig = create_image_figure(img_slice)
        colors = [c for i, c in enumerate(CLASS_COLORS) if i+1 in np.unique(label_slice)]
        visualize_with_overlay(img_slice, label_slice, alpha=0.6, ax=fig.gca(), colors=colors)

        save_fig(fig, 'results/plots/qualitative', f'{model_name}_{patid}_slice{slice_num}', pdf=False)
        legend_figure(labels=[CLASSES[i] for i in sorted(list(CLASSES.keys()))], colors=CLASS_COLORS,
                      outdir='results/plots/qualitative', basename=f'classes_legend')

        # plot the slice without labels
        fig = create_image_figure(img_slice)
        visualize_with_overlay(img_slice, np.zeros_like(label_slice), ax=fig.gca())
        save_fig(fig, 'results/plots/qualitative', f'{patid}_slice{slice_num}', pdf=False)


def kp_comparison_figure(patid='s0070', ae=False):
    model = 'DGCNN_seg' if not ae else 'DSEGAE_reg_aug_1024'
    model_folders = {
        'cnn': f'results/{model}_cnn_image',
        'hessian': f'results/{model}_enhancement_image',
        'foerstner': f'results/{model}_foerstner_image',
    }

    img_windowed, fold = get_image_and_fold(patid)
    result_folders = {model: os.path.join(folder, f'fold{fold}', 'test_predictions', 'labelmaps') for model, folder in model_folders.items()}

    target = sitk.ReadImage(os.path.join(result_folders['cnn'].replace(model, 'DGCNN_seg'), f'{patid}_fissures_targ_fixed.nii.gz'))
    label_maps = {model: sitk.ReadImage(os.path.join(path, f'{patid}_fissures_pred_fixed.nii.gz')) for model, path in result_folders.items()}
    label_maps['ground-truth'] = target

    for kp in label_maps.keys():
        model_name = f'{model}_{kp}_image' if kp != 'ground-truth' else 'ground-truth'
        multi_class_overlay(img_windowed, label_maps[kp], slice_dim=2, model_name=model_name, patid=patid)


def get_image_and_fold(patid, ds_path=IMG_DIR_TS):
    ds = ImageDataset(ds_path, do_augmentation=False)
    img_index = ds.get_index(patid, 'fixed')
    split = load_split_file(DEFAULT_SPLIT_TS)
    fold = find_test_fold_for_id(patid, sequence='fixed', split=split)
    img = ds.get_image(img_index)
    mask = ds.get_lung_mask(img_index)
    img_windowed = fissure_window_level_and_mask(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(mask))
    return img_windowed, fold


def comparative_figures(patid='s0070'):
    img_windowed, fold = get_image_and_fold(patid, IMG_DIR_TSv1)
    label_map_nnu = sitk.ReadImage(os.path.join(
        f'/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/nnUNet_baseline/nnu_results/nnUNet/3d_fullres/Task503_FissuresTotalSeg/nnUNetTrainerV2_200ep__nnUNetPlansv2.1/fold_{fold}/validation_raw_postprocessed',
        f'{patid}_fix.nii.gz'))
    multi_class_overlay(img_windowed, label_map_nnu, slice_dim=2, model_name='nnU-Net', patid=patid)

    label_map_v2m = sitk.ReadImage(os.path.join(
        f'/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/voxel2mesh-master/resultsExperiment_003/trial_{fold+1}/best_performance/voxels',
        f'{patid}_fissures_pred_fixed.nii.gz'))
    multi_class_overlay(img_windowed, label_map_v2m, slice_dim=2, model_name='v2m', patid=patid)


if __name__ == '__main__':
    # comparative_figures()
    kp_comparison_figure(ae=True)
