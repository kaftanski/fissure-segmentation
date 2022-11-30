import os.path

import SimpleITK as sitk
import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt

from constants import DEFAULT_SPLIT_TS
from data import load_split_file
from preprocess_totalsegmentator_dataset import TotalSegmentatorDataset
from thesis.utils import legend_figure, save_fig, textwidth_to_figsize
from utils.general_utils import find_test_fold_for_id
from visualization import visualize_with_overlay


def fissure_window_level(img: np.ndarray):
    low = -1024
    high = 500
    img[img < low] = low
    img[img > high] = high
    return img


def multi_model_overlay(img, label_maps, slice_num, slice_dim=2):
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

    fig = plt.figure(figsize=textwidth_to_figsize(0.3, 1/2))

    colors = matplotlib.cm.get_cmap('tab10').colors
    visualize_with_overlay(img_slice.squeeze(), combined_label.squeeze(), alpha=0.5, colors=colors)

    save_fig(fig, 'results/plots', 'keypoint_qualitative_comparison', pdf=False)
    legend_figure(labels=list(label_maps.keys()), colors=colors,
                  outdir='results/plots', basename='keypoint_qualitative_comparison')


def kp_comparison_figure(patid='s0070'):
    model_folders = {
        'cnn': 'results/DGCNN_seg_cnn_image',
        'hessian': 'results/DGCNN_seg_enhancement_image',
        'foerstner': 'results/DGCNN_seg_foerstner_image',
    }

    ds = TotalSegmentatorDataset()
    img_index = ds.get_index(patid, 'fixed')
    split = load_split_file(DEFAULT_SPLIT_TS)
    fold = find_test_fold_for_id(patid, sequence='fixed', split=split)
    result_folders = {model: os.path.join(folder, f'fold{fold}', 'test_predictions', 'labelmaps') for model, folder in model_folders.items()}

    img = ds.get_image(img_index)
    img_windowed = fissure_window_level(sitk.GetArrayFromImage(img))

    target = sitk.ReadImage(os.path.join(result_folders['cnn'], f'{patid}_fissures_targ_fixed.nii.gz'))
    label_maps = {model: sitk.ReadImage(os.path.join(path, f'{patid}_fissures_pred_fixed.nii.gz')) for model, path in result_folders.items()}
    label_maps['ground-truth'] = target

    multi_model_overlay(img_windowed, label_maps, slice_dim=2, slice_num=180)


if __name__ == '__main__':
    kp_comparison_figure()
