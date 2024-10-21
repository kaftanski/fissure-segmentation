import argparse
import os.path

import SimpleITK as sitk

from constants import IMG_DIR_COPD, POINT_DIR_COPD
from data_processing.datasets import LungData
from data_processing.find_lobes import find_lobes
from data_processing.keypoint_extraction import compute_keypoints
from data_processing.surface_fitting import poisson_reconstruction
from utils.sitk_image_ops import apply_mask
from utils.general_utils import save_meshes


def main(ds, index):
    exclude_right_horizontal_fissure = True

    fissures = ds.get_fissures(index)
    if fissures is None:
        print(f'No fissure segmentation for {ds.get_filename(index)} available ... Skipping.')
        return

    img = ds.get_image(index)
    mask = ds.get_lung_mask(index)
    case, sequence = ds.get_id(index)
    print(f'Running data processing for fissures of {case}, {sequence}\n')

    # 1. surface fitting (poisson)
    regularized_fissures, fissure_meshes = poisson_reconstruction(fissures, mask)
    poisson_path = os.path.join(IMG_DIR_COPD, f'{case}_fissures_poisson_{sequence}.nii.gz')
    sitk.WriteImage(regularized_fissures, poisson_path)
    save_meshes(fissure_meshes, IMG_DIR_COPD, case, sequence, obj_name='fissure')

    # 2. Lung-Masking of fissure labels
    fissures_masked = apply_mask(regularized_fissures, mask)
    sitk.WriteImage(fissures_masked, poisson_path)
    print()

    # 3. Lobe Generation
    lobes, lobe_meshes, success = find_lobes(fissures_masked, mask, exclude_rhf=exclude_right_horizontal_fissure)
    sitk.WriteImage(lobes, os.path.join(IMG_DIR_COPD, f'{case}_lobes_{sequence}.nii.gz'))
    if not success:
        print('Not enough lobes found, please check if the fissure segmentation is complete. '
              'Skipping point feature computation.')
        return
    save_meshes(lobe_meshes, IMG_DIR_COPD, case, sequence, obj_name='lobe')


def run_all():
    ds = LungData(IMG_DIR_COPD)
    for i in range(len(ds)):
        main(ds, i)


def run_one(case, sequence):
    ds = LungData(IMG_DIR_COPD)
    img_index = ds.get_index(case, sequence)
    main(ds, img_index)


if __name__ == '__main__':
    run_all()
