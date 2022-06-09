import os.path
import time

import SimpleITK as sitk
import torch

from data import ImageDataset, load_split_file
from data_processing import foerstner
from models.seg_cnn import MobileNetASPP


KP_MODES = ['foerstner', 'noisy', 'cnn']


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
    kp = kp[torch.randperm(len(kp))[:20000]].to(device)
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

    kp = torch.nonzero(fissure_points) * torch.tensor((ds.resample_spacing,)*3, device=fissure_points.device)
    kp = kp.long()
    return kp


if __name__ == '__main__':
    print(get_cnn_keypoints('../results/binary_3DCNN_cv', 'COPD01', 'fixed', device='cuda:1'))
