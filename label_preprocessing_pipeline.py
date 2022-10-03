import argparse
import os.path

import SimpleITK as sitk

from data import LungData
from data_processing.find_lobes import find_lobes
from data_processing.keypoint_extraction import compute_keypoints
from data_processing.surface_fitting import poisson_reconstruction, save_meshes
from utils.image_ops import apply_mask

IMG_DATA_DIR = '/home/kaftan/FissureSegmentation/data'
POINT_DATA_DIR = '/home/kaftan/FissureSegmentation/point_data'


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
    poisson_path = os.path.join(IMG_DATA_DIR, f'{case}_fissures_poisson_{sequence}.nii.gz')
    sitk.WriteImage(regularized_fissures, poisson_path)
    save_meshes(fissure_meshes, IMG_DATA_DIR, case, sequence, obj_name='fissure')

    # 2. Lung-Masking of fissure labels
    fissures_masked = apply_mask(regularized_fissures, mask)
    sitk.WriteImage(fissures_masked, poisson_path)
    print()

    # 3. Lobe Generation
    lobes, lobe_meshes, success = find_lobes(fissures_masked, mask, exclude_rhf=exclude_right_horizontal_fissure)
    sitk.WriteImage(lobes, os.path.join(IMG_DATA_DIR, f'{case}_lobes_{sequence}.nii.gz'))
    if not success:
        print('Not enough lobes found, please check if the fissure segmentation is complete. '
              'Skipping point feature computation.')
        return
    save_meshes(lobe_meshes, IMG_DATA_DIR, case, sequence, obj_name='lobe')

    # 4. Point Features
    compute_keypoints(img, fissures_masked, lobes, mask, POINT_DATA_DIR, case, sequence)
    # TODO: compute features


def run_all():
    ds = LungData(IMG_DATA_DIR)
    for i in range(len(ds)):
        main(ds, i)


def run_one(case, sequence):
    ds = LungData(IMG_DATA_DIR)
    img_index = ds.get_index(case, sequence)
    main(ds, img_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, help='case ID, e.g. "EMPIRE01"')
    parser.add_argument('--sequence', type=str, choices=['fixed', 'moving'], help='case sequence: either fixed or moving')
    args = parser.parse_args()

    if args.case and args.sequence:
        run_one(args.case, args.sequence)
    else:
        run_all()
