import os.path
import time

import SimpleITK as sitk
import torch

from data import ImageDataset, load_split_file, LungData
from data_processing import foerstner
from models.seg_cnn import MobileNetASPP
from utils.detached_run import run_detached_from_pycharm
from utils.image_ops import resample_equal_spacing, multiple_objects_morphology, sitk_image_to_tensor
from utils.utils import kpts_to_grid, ALIGN_CORNERS

KP_MODES = ['foerstner', 'noisy', 'cnn', 'enhancement']
MAX_KPTS = 20000  # point clouds shouldn't be bigger for memory concerns


POINT_DIR = '/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/point_data'
POINT_DIR_TS = os.path.join(POINT_DIR, 'ts')


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


def get_cnn_keypoints(cv_dir, case, sequence, device, softmax_threshold=0.3):
    ds = ImageDataset(folder='../data', do_augmentation=False)
    cross_val_split = load_split_file(os.path.join(cv_dir, "cross_val_split.np.pkl"))
    sequence_temp = sequence.replace('moving', 'mov').replace('fixed', 'fix')

    # find the fold, where this image has been in the test-split
    fold_nr = None
    for i, fold in enumerate(cross_val_split):
        if any(case in name and sequence_temp in name for name in fold['val']):
            fold_nr = i

    if fold_nr is None:
        raise ValueError(f'ID {case}_{sequence} is not present in any cross-validation test split (directory: {cv_dir})')

    model = MobileNetASPP.load(os.path.join(cv_dir, f'fold{fold_nr}', 'model.pth'), device=device)
    model.eval()
    model.to(device)

    input_img = ds.get_batch_collate_fn()([ds[ds.get_index(case, sequence)]])[0].to(device)
    with torch.no_grad():
        softmax_pred = model.predict_all_patches(input_img)

    # threshold the softmax scores
    fissure_points = torch.zeros(softmax_pred.shape[2:], device=device)
    for lbl in range(1, model.num_classes):
        fissure_points = torch.logical_or(fissure_points, softmax_pred[0, lbl] > softmax_threshold)
    # TODO: apply lung mask
    kp = torch.nonzero(fissure_points) * torch.tensor((ds.resample_spacing,)*3, device=fissure_points.device)
    kp = kp.long()

    return kp


def get_hessian_fissure_enhancement_kpts(enhanced_img, device, threshold=0.3):
    enhanced_img = sitk.DiscreteGaussian(enhanced_img, variance=(1, 1, 1), useImageSpacing=True)
    enhanced_img_tensor = sitk_image_to_tensor(enhanced_img).to(device)
    kp = torch.nonzero(enhanced_img_tensor > threshold).long()
    return kp


def compute_keypoints(img, fissures, lobes, mask, out_dir, case, sequence, kp_mode='foerstner', enhanced_img_path: str=None, device='cuda:2'):
    print(f'Computing keypoints and point features for case {case}, {sequence}...')
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
    fissures_dilated = multiple_objects_morphology(fissures, radius=2, mode='dilate')
    fissures_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissures_dilated).astype(int))

    # dilate lobes to fill gaps from the fissures  # TODO: use lobe filling?
    lobes_dilated = multiple_objects_morphology(lobes, radius=2, mode='dilate')

    if kp_mode == 'foerstner':
        kp = get_foerstner_keypoints(device, img_tensor, mask, sigma=0.5, threshold=1e-8, nms_kernel=7)

    elif kp_mode == 'noisy':
        kp = get_noisy_keypoints(fissures_tensor, device)

    elif kp_mode == 'cnn':
        kp = get_cnn_keypoints(cv_dir='results/recall_loss', case=case, sequence=sequence, device=device)

    elif kp_mode == 'enhancement':
        assert enhanced_img_path is not None, \
            'Tried to use fissure enhancement for keypoint extraction but no path to enhanced image given.'
        enhanced_img = sitk.ReadImage(enhanced_img_path)
        kp = get_hessian_fissure_enhancement_kpts(enhanced_img, device, threshold=0.25)

    else:
        raise ValueError(f'No keypoint-mode named "{kp_mode}".')

    # limit number of keypoints
    if len(kp) > MAX_KPTS:
        kp = kp[torch.randperm(len(kp), device=kp.device)[:MAX_KPTS]]

    # get label for each point
    kp_cpu = kp.cpu()
    labels = fissures_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    torch.save(labels.cpu(), os.path.join(out_dir, f'{case}_fissures_{sequence}.pth'))
    print(f'\tkeypoints per fissure: {labels.unique(return_counts=True)[1].tolist()}')

    lobes_tensor = torch.from_numpy(sitk.GetArrayFromImage(lobes_dilated).astype(int))
    lobes = lobes_tensor[kp_cpu[:, 0], kp_cpu[:, 1], kp_cpu[:, 2]]
    torch.save(lobes.cpu(), os.path.join(out_dir, f'{case}_lobes_{sequence}.pth'))
    print(f'\tkeypoints per lobe: {lobes.unique(return_counts=True)[1].tolist()}')

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

    # data_dir = '../TotalSegmentator/ThoraxCrop'
    data_dir = '../data'
    ds = LungData(data_dir)

    for mode in KP_MODES:
        if mode == 'noisy':
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

            compute_keypoints(img, fissures, lobes, mask, POINT_DIR_TS, case, sequence, kp_mode='enhancement',
                              enhanced_img_path=ds.fissures_enhanced[i], device='cuda:3')
