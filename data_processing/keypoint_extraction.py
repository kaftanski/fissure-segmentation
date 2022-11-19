import os.path
import time

import SimpleITK as sitk
import numpy as np
import torch

from cli.cl_args import get_seg_cnn_train_parser
from cli.cli_utils import load_args_for_testing
from constants import KP_MODES, POINT_DIR, POINT_DIR_TS
from data import ImageDataset, load_split_file, LungData
from data_processing import foerstner
from models.lraspp_3d import LRASPP_MobileNetv3_large_3d
from models.seg_cnn import MobileNetASPP
from utils.detached_run import run_detached_from_pycharm
from utils.image_ops import resample_equal_spacing, multiple_objects_morphology, sitk_image_to_tensor
from utils.utils import kpts_to_grid, ALIGN_CORNERS, sample_patches_at_kpts, topk_alldims

MAX_KPTS = 20000  # point clouds shouldn't be bigger for memory concerns


def get_foerstner_keypoints(device, img_tensor, mask, sigma=0.5, threshold=1e-8, nms_kernel=7):
    # compute förstner keypoints
    start = time.time()
    kp = foerstner.foerstner_kpts(img_tensor,
                                  mask=torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool)).unsqueeze(
                                      0).unsqueeze(0).to(device),
                                  sigma=sigma, thresh=threshold, d=nms_kernel)
    print(f'\tFound {kp.shape[0]} keypoints (took {time.time() - start:.4f})')
    return kp


def get_noisy_keypoints(fissures_tensor, device):
    """ compute keypoints as noisy fissure labels (for testing DGCNN)

    :param fissures_tensor: fissure labels tensor
    :param device: torch device
    :return: keypoints
    """
    # take all fissure points (limit to 20.000 for computational reasons)
    kp = torch.nonzero(fissures_tensor).float()
    kp = kp[torch.randperm(len(kp))[:MAX_KPTS]].to(device)
    # add some noise to them
    noise = torch.randn_like(kp, dtype=torch.float) * 3
    kp += noise.to(device)
    kp = kp.long()
    # prevent index out of bounds
    for d in range(kp.shape[1]):
        kp[:, d] = torch.clamp(kp[:, d], min=0, max=fissures_tensor.squeeze().shape[d] - 1)
    return kp


def get_cnn_keypoints(cv_dir, case, sequence, device, out_path, softmax_threshold=0.3, feat_patch=5):
    """ also computes CNN features (the softmax scores patch)

    :param cv_dir:
    :param case:
    :param sequence:
    :param device:
    :param out_path:
    :param softmax_threshold:
    :param feat_patch:
    :return:
    """
    default_parser = get_seg_cnn_train_parser()
    args, _ = default_parser.parse_known_args()
    args = load_args_for_testing(cv_dir, args)

    ds = ImageDataset(folder=data_dir, do_augmentation=False, patch_size=(args.patch_size,)*3,
                      resample_spacing=args.spacing)
    cross_val_split = load_split_file(os.path.join(cv_dir, "cross_val_split.np.pkl"))
    sequence_temp = sequence.replace('moving', 'mov').replace('fixed', 'fix')

    # find the fold, where this image has been in the test-split
    fold_nr = None
    for i, fold in enumerate(cross_val_split):
        if any(case in name and sequence_temp in name for name in fold['val']):
            fold_nr = i

    if fold_nr is None:
        raise ValueError(f'ID {case}_{sequence} is not present in any cross-validation test split (directory: {cv_dir})')

    if args.model == 'v1':
        model_class = MobileNetASPP
    elif args.model == 'v3':
        model_class = LRASPP_MobileNetv3_large_3d
    else:
        raise NotImplementedError()

    model = model_class.load(os.path.join(cv_dir, f'fold{fold_nr}', 'model.pth'), device=device)
    model.eval()
    model.to(device)

    img_index = ds.get_index(case, sequence)
    input_img = ds.get_batch_collate_fn()([ds[img_index]])[0].to(device)
    with torch.no_grad():
        softmax_pred = model.predict_all_patches(input_img)

    # threshold the softmax scores
    # fissure_points = torch.zeros(softmax_pred.shape[2:], device=device)
    # for lbl in range(1, model.num_classes):  # TODO: take argmax?
    #     fissure_points = torch.logical_or(fissure_points, softmax_pred[0, lbl] > softmax_threshold)
    fissure_points = softmax_pred.argmax(1).squeeze() != 0

    # apply lung mask
    lung_mask = resample_equal_spacing(ds.get_lung_mask(ds.get_index(case, sequence)),
                                       ds.resample_spacing, use_nearest_neighbor=True)
    lung_mask = sitk_image_to_tensor(lung_mask).to(device)
    fissure_points = torch.logical_and(fissure_points, lung_mask)

    # nonzero voxels to points
    kp = torch.nonzero(fissure_points) * ds.resample_spacing

    # compute cnn features: sum of foreground softmax scores)
    kp_grid = kpts_to_grid(kp.flip(-1),
                           shape=torch.tensor(fissure_points.shape) * ds.resample_spacing, align_corners=ALIGN_CORNERS)
    features = sample_patches_at_kpts(softmax_pred[:, 1:].sum(1, keepdim=True), kp_grid, feat_patch).squeeze().flatten(start_dim=1).transpose(0, 1)
    # torch.save(features.cpu(), os.path.join(out_path, f'{case}_cnn_{sequence}.pth'))

    return kp.long(), features


def get_hessian_fissure_enhancement_kpts(enhanced_img, device, min_threshold=0.2):
    enhanced_img = sitk.DiscreteGaussian(enhanced_img, variance=(1, 1, 1), useImageSpacing=True)
    enhanced_img_tensor = sitk_image_to_tensor(enhanced_img).to(device)

    top_vals, top_idx = topk_alldims(enhanced_img_tensor, MAX_KPTS)
    top_idx = torch.stack(top_idx, dim=1)
    kp = top_idx[top_vals > min_threshold]
    return kp


def compute_keypoints(img, fissures, lobes, mask, out_dir, case, sequence, kp_mode='foerstner',
                      enhanced_img_path: str=None, cnn_dir: str=None, device='cuda:2'):
    if kp_mode == 'cnn':
        assert cnn_dir is not None

    print(f'Computing {kp_mode} keypoints for case {case}, {sequence}...')
    torch.cuda.empty_cache()

    out_dir = os.path.join(out_dir, kp_mode)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # resample all images to unit spacing
    img = resample_equal_spacing(img, target_spacing=1)
    mask = resample_equal_spacing(mask, target_spacing=1, use_nearest_neighbor=True)
    fissures = resample_equal_spacing(fissures, target_spacing=1, use_nearest_neighbor=True)
    lobes = resample_equal_spacing(lobes, target_spacing=1, use_nearest_neighbor=True)

    img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0).float().to(device)

    # dilate fissures so that more keypoints get assigned foreground labels
    fissures_dilated = multiple_objects_morphology(fissures, radius=2, mode='dilate')  # TODO: problem?
    fissures_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures_dilated).astype(int))

    # # dilate lobes to fill gaps from the fissures
    # lobes_dilated = multiple_objects_morphology(lobes, radius=2, mode='dilate')

    if kp_mode == 'foerstner':
        kp = get_foerstner_keypoints(device, img_tensor, mask, sigma=0.5, threshold=1e-8, nms_kernel=5)

    elif kp_mode == 'noisy':
        kp = get_noisy_keypoints(fissures_tensor, device)

    elif kp_mode == 'cnn':
        kp, cnn_feat = get_cnn_keypoints(cv_dir=cnn_dir, case=case, sequence=sequence, device=device, out_path=out_dir)

    elif kp_mode == 'enhancement':
        assert enhanced_img_path is not None, \
            'Tried to use fissure enhancement for keypoint extraction but no path to enhanced image given.'
        enhanced_img = sitk.ReadImage(enhanced_img_path)
        kp = get_hessian_fissure_enhancement_kpts(enhanced_img, device, min_threshold=0.2)

    else:
        raise ValueError(f'No keypoint-mode named "{kp_mode}".')

    # limit number of keypoints
    if len(kp) > MAX_KPTS:
        perm = torch.randperm(len(kp), device=kp.device)[:MAX_KPTS]
        kp = kp[perm]
        if kp_mode == 'cnn':
            torch.save(cnn_feat.cpu()[:, perm], os.path.join(out_dir, f'{case}_cnn_{sequence}.pth'))
    elif len(kp) < 2048:
        print(case, sequence, "has less than minimum of 2048 kpts!")

    # get label for each point
    kp_cpu = kp.cpu()
    labels = fissures_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    torch.save(labels.cpu(), os.path.join(out_dir, f'{case}_fissures_{sequence}.pth'))
    print(f'\tkeypoints per fissure: {labels.unique(return_counts=True)[1].tolist()}')

    # lobes_tensor = torch.from_numpy(sitk.GetArrayFromImage(lobes_dilated).astype(int))
    # lobes = lobes_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    # torch.save(lobes.cpu(), os.path.join(out_dir, f'{case}_lobes_{sequence}.pth'))
    # print(f'\tkeypoints per lobe: {lobes.unique(return_counts=True)[1].tolist()}')

    # coordinate features: transform indices into physical points
    spacing = torch.tensor(img.GetSpacing()[::-1]).unsqueeze(0).to(device)
    points = kpts_to_grid((kp * spacing).flip(-1), torch.tensor(img_tensor.shape[2:], device=device) * spacing.squeeze(),
                          align_corners=ALIGN_CORNERS).transpose(0, 1)
    torch.save(points.cpu(), os.path.join(out_dir, f'{case}_coords_{sequence}.pth'))

    # compute_point_features(img_tensor, kp, case, sequence, out_dir, use_mind)

    # # VISUALIZATION
    # for i in range(-5, 5):
    #     chosen_slice = img_tensor.squeeze().shape[1] // 2 + i
    #     plt.imshow(img_tensor.squeeze()[:, chosen_slice].cpu(), 'gray')
    #     keypoints_slice = kp_cpu[kp_cpu[:, 1] == chosen_slice]
    #     plt.plot(keypoints_slice[:, 2], keypoints_slice[:, 0], '+')
    #     plt.gca().invert_yaxis()
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'results/EMPIRE02_fixed_keypoints_{i+5}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    #     plt.show()


if __name__ == '__main__':
    run_detached_from_pycharm()

    ts = True

    if ts:
        data_dir = '../TotalSegmentator/ThoraxCrop_v2'
        point_dir = POINT_DIR_TS
        cnn_dir = 'results/lraspp_recall_loss'
    else:
        data_dir = '../data'
        point_dir = POINT_DIR
        cnn_dir = 'results/recall_loss'

    ds = LungData(data_dir)

    for mode in KP_MODES:
        if mode != 'cnn':
            continue

        print('MODE:', mode)
        for i in range(len(ds)):
            case, _, sequence = ds.get_filename(i).split('/')[-1].split('_')
            sequence = sequence.replace('.nii.gz', '')

            print(f'Computing points for case {case}, {sequence}...')
            if ds.fissures[i] is None:
                print('\tNo fissure segmentation found.')
                continue

            img = ds.get_image(i)
            fissures = ds.get_regularized_fissures(i)
            lobes = ds.get_lobes(i)
            mask = ds.get_lung_mask(i)

            if mode == 'foerstner' and np.prod(img.GetSize()) > 26.5 * 1e6:
                device = 'cpu'
            else:
                device = 'cuda:0'

            compute_keypoints(img, fissures, lobes, mask, point_dir, case, sequence, kp_mode=mode,
                              enhanced_img_path=ds.fissures_enhanced[i], device=device, cnn_dir=cnn_dir)
