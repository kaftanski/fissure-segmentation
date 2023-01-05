import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

from constants import POINT_DIR_TS, IMG_DIR_TS, KP_MODES
from data import ImageDataset
from data_processing.fissure_enhancement import get_enhanced_fissure_image, HessianEnhancementFilter, \
    load_fissure_stats, FISSURE_STATS_FILE
from preprocess_totalsegmentator_dataset import find_non_zero_ranges
from qualitative import fissure_window_level_and_mask, crop_to_lung_indices, create_image_figure, slice_image
from thesis.utils import textwidth_to_figsize, save_fig
from utils.general_utils import load_points
from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor

result_folder = 'results/plots/keypoints'


def plot_keypoints(image, mode, patid, slice_num, slice_dim=1, spacing=1.5, crop_indices=None):
    pts, _, _, _ = load_points(os.path.join(POINT_DIR_TS, mode), patid)
    pts = (pts.T + 1)/2 * np.flip(np.array(image.shape))[None]

    if crop_indices is None:
        crop_indices = crop_to_lung_indices(image)
    slice_num_cropped = slice_num - crop_indices[slice_dim].start
    image = image[crop_indices]
    index = [slice(None)] * slice_dim + [slice(slice_num_cropped, slice_num_cropped + 1)]
    img_slice = image[index].squeeze()

    pts = pts - np.array([crop_indices[2].start, crop_indices[1].start, crop_indices[0].start])
    pts_index = [0, 1, 2]
    pts_index.remove(slice_dim)
    pts_slice = pts[np.abs(pts[:, slice_dim] - slice_num_cropped) < spacing/2]

    fig = plt.figure(figsize=textwidth_to_figsize(2, aspect=img_slice.shape[0] / img_slice.shape[1]))
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(img_slice, cmap='gray')
    ax.scatter(pts_slice[:, pts_index[0]], pts_slice[:, pts_index[1]], marker='+', s=300, c='orangered', linewidths=3)
    save_fig(fig, result_folder, f'{mode}_keypoints_{patid}_slice{slice_num}_largemarker', pdf=False)

    fig = plt.figure(figsize=textwidth_to_figsize(2, aspect=img_slice.shape[0] / img_slice.shape[1]))
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(img_slice, cmap='gray')
    save_fig(fig, result_folder, f'{patid}_slice{slice_num}', pdf=False)


def plot_all(patid='s0070'):
    ds = ImageDataset(IMG_DIR_TS)
    img_index = ds.get_index(patid, 'fixed')
    img = ds.get_image(img_index)
    img_window = fissure_window_level_and_mask(sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(ds.get_lung_mask(img_index)))

    for mode in KP_MODES:
        plot_keypoints(img_window, mode, slice_num=118, patid=patid, slice_dim=1)


def plot_bad_example_enhancement():
    patid = 's1024'
    slice_num = 287# 316
    ds = ImageDataset(IMG_DIR_TS)
    img_index = ds.get_index(patid, 'fixed')
    enhanced = ds.get_enhanced_fissures(img_index)
    enhanced = sitk.GetArrayFromImage(enhanced)
    mask = resample_equal_spacing(ds.get_lung_mask(img_index), use_nearest_neighbor=True)
    mask = sitk.GetArrayFromImage(mask)
    enhanced[mask == 0], _ = slice_image(enhanced, slice_num=slice_num, slice_dim=1).max()
    crop_indices = (slice(15, 200), slice(0, enhanced.shape[1]), slice(130, 400))
    plot_keypoints(enhanced, 'enhancement', slice_num=slice_num, patid=patid, slice_dim=1, crop_indices=crop_indices)

    img = ds.get_image(img_index)
    img = resample_equal_spacing(img)
    img_window = fissure_window_level_and_mask(sitk.GetArrayFromImage(img), mask, high=0)
    img_crop = img_window[crop_indices]
    img_slice, _ = slice_image(img_crop, slice_num=slice_num, slice_dim=1)
    fig = create_image_figure(img_window)
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(img_slice, cmap='gray')
    save_fig(fig, result_folder, f'{patid}_slice{slice_num}', pdf=False)


def plot_enhancement(patid='s0070'):
    ds = ImageDataset(IMG_DIR_TS)
    img_index = ds.get_index(patid, 'fixed')
    img = ds.get_image(img_index)
    img = resample_equal_spacing(img)
    mask = resample_equal_spacing(ds.get_lung_mask(img_index))
    fissure_mu, fissure_sigma = load_fissure_stats(FISSURE_STATS_FILE)
    fissure_filter = HessianEnhancementFilter(fissure_mu, fissure_sigma)
    enhanced, planeness, hu_weights = fissure_filter(sitk_image_to_tensor(img).float().unsqueeze(0).unsqueeze(0), return_intermediate=True)
    enhanced = enhanced.squeeze()

    # apply mask
    enhanced[sitk_image_to_tensor(mask) == 0] = 0
    planeness[sitk_image_to_tensor(mask) == 0] = 0
    hu_weights[sitk_image_to_tensor(mask) == 0] = 0

    ranges = find_non_zero_ranges(enhanced.numpy()[None])
    crop_indices = slice(ranges[0, 0], ranges[0, 1] + 1), slice(ranges[1, 0], ranges[1, 1] + 1), slice(ranges[2, 0], ranges[2, 1] + 1)

    enhanced = enhanced[crop_indices]
    planeness = planeness[crop_indices]
    hu_weights = hu_weights[crop_indices]

    slice_dim = 1
    slice_num = 118
    slice_num_cropped = slice_num - crop_indices[slice_dim].start

    enhanced, _ = slice_image(enhanced, slice_num_cropped, slice_dim)
    fig = create_image_figure(enhanced)
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(enhanced)
    save_fig(fig, result_folder, 'enhancement_result', pdf=False)

    planeness, _ = slice_image(planeness, slice_num_cropped, slice_dim)
    fig = create_image_figure(planeness)
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(planeness)
    save_fig(fig, result_folder, 'enhancement_planeness', pdf=False)

    hu_weights, _ = slice_image(hu_weights, slice_num_cropped, slice_dim)
    fig = create_image_figure(hu_weights)
    ax = fig.gca()
    ax.set_axis_off()
    ax.imshow(hu_weights)
    save_fig(fig, result_folder, 'enhancement_hu_weights', pdf=False)


if __name__ == '__main__':
    KP_MODES.remove('noisy')
    plot_all()
    # # plot_enhancement()
    # plot_bad_example_enhancement()
