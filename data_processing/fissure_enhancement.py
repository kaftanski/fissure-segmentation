import os
import time

import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import _gaussian_kernel1d
from skimage.filters.ridges import compute_hessian_eigenvalues
from sklearn.metrics import RocCurveDisplay

from utils.image_ops import resample_equal_spacing, apply_mask
from utils.image_utils import filter_1d, smooth
from visualization import plot_slice


def gaussian_kernel_1d(sigma, order=0, truncate=4.0):
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel1d(sigma, order, radius)
    return kernel


def hessian_matrix(img: torch.Tensor, sigma: float):
    kernel_1st_deriv = gaussian_kernel_1d(sigma, order=2)
    kernel_1st_deriv = torch.from_numpy(kernel_1st_deriv).float().to(img.device)

    kernel_2nd_deriv = gaussian_kernel_1d(sigma, order=2)
    kernel_2nd_deriv = torch.from_numpy(kernel_2nd_deriv).float().to(img.device)

    img = img.float()
    H = torch.zeros(*img.squeeze().shape, 3, 3, device=img.device)
    for first_dim in range(len(img.squeeze().shape)):
        for second_dim in range(len(img.squeeze().shape)):
            if first_dim == second_dim:
                # filter dimension with 2nd derivation of gaussian kernel
                H[..., first_dim, second_dim] = filter_1d(img, kernel_2nd_deriv, dim=first_dim).squeeze()

            elif second_dim > first_dim:
                # filter with 1st derivation of gaussian kernel in both dimensions
                img_deriv = filter_1d(
                    filter_1d(img, kernel_1st_deriv, dim=first_dim),
                    kernel_1st_deriv, dim=second_dim).squeeze()

                # differentiation is linear -> sequence does not matter
                H[..., first_dim, second_dim] = img_deriv
                H[..., second_dim, first_dim] = img_deriv

    return H


def hessian_based_enhancement_torch(img: torch.Tensor, fissure_mu: float, fissure_sigma: float, device='cuda:0', show=False):
    # ensure batch and channel dimensions
    img = img.squeeze()
    img = img.view(1, 1, *img.shape)
    img = img.float().to(device)

    # smooth image with gaussian kernel
    img_smooth = smooth(img, sigma=1.)

    # compute hessian matrix
    H = hessian_matrix(img_smooth, sigma=1.)

    # get hessian eigenvalues
    eigenvalues = torch.linalg.eigvalsh(H)

    # sort by absolute value in descending order
    abs_sorting = torch.argsort(torch.abs(eigenvalues), dim=-1, descending=True)
    eigenvalues = torch.gather(eigenvalues, dim=-1, index=abs_sorting)

    # compute the fissure enhanced image
    fissure_enhanced = fissure_filter(img.squeeze(), eigenvalues[..., 0], eigenvalues[..., 1], fissure_mu, fissure_sigma, show=show)
    return fissure_enhanced


def fissure_filter(img, hessian_lambda1, hessian_lambda2, fissure_mu, fissure_sigma, show=False):
    backend = torch if isinstance(hessian_lambda1, torch.Tensor) else np
    # compute planeness value
    P = backend.zeros_like(hessian_lambda1)
    abs_lambda1 = backend.abs(hessian_lambda1)
    abs_lambda2 = backend.abs(hessian_lambda2)
    P[hessian_lambda1 < 0] = (abs_lambda1[hessian_lambda1 < 0] - abs_lambda2[hessian_lambda1 < 0]) \
                             / (abs_lambda1[hessian_lambda1 < 0] + abs_lambda2[hessian_lambda1 < 0])
    # P[lambda1 >= 0] = 0

    # compute fissure likeliness based on Hounsfield units
    hu_weights = backend.exp(-((img - fissure_mu)**2) / (2 * fissure_sigma**2))

    # full fissure filter
    F = hu_weights * P

    # visualize
    if show:
        def maybe_to_cpu(maybe_tensor):
            if backend == torch:
                maybe_tensor = maybe_tensor.cpu()
            return maybe_tensor

        dim = 1
        plot_slice(maybe_to_cpu(P[None, None]), s=P.shape[dim]//2, dim=dim, title='plane-ness')
        plot_slice(maybe_to_cpu(F[None, None]), s=F.shape[dim]//2, dim=dim, title='fissure-ness')
        plot_slice(maybe_to_cpu(hu_weights[None, None]), s=hu_weights.shape[dim] // 2, dim=dim, title='HU weights')

    return F


def hessian_based_enhancement(img: np.ndarray, fissure_mu: float, fissure_sigma: float, show=False):
    """ R. Wiemker et al.:
    "Unsupervised extraction of the pulmonary interlobar fissures from high resolution thoracic CT data"

    :return
    """
    # smooth image
    img_smooth = gaussian_filter(img, sigma=1)

    # eigenvalues are sorted in absolute ascending order, we want descending
    lambda3, lambda2, lambda1 = compute_hessian_eigenvalues(img_smooth, sigma=1.,  # TODO: more scales (more sigmas)
                                                            sorting='abs', mode='reflect')

    F = fissure_filter(img, lambda1, lambda2, fissure_mu, fissure_sigma, show)

    return F


def get_enhanced_fissure_image(image: sitk.Image, fissures: sitk.Image, lung_mask: sitk.Image, resample_spacing: float = None, device='cuda:2', show=False):
    if resample_spacing is not None:
        image = resample_equal_spacing(image, target_spacing=resample_spacing)
        fissures = resample_equal_spacing(fissures, target_spacing=resample_spacing, use_nearest_neighbor=True)
        lung_mask = resample_equal_spacing(lung_mask, target_spacing=resample_spacing, use_nearest_neighbor=True)

    img_arr = sitk.GetArrayFromImage(image)
    fissures_arr = sitk.GetArrayFromImage(fissures)

    # compute fissure HU statistics
    fissure_mu = img_arr[fissures_arr != 0].mean()
    fissure_sigma = img_arr[fissures_arr != 0].std()

    # fissure enhancement
    start = time.time()
    F = hessian_based_enhancement_torch(torch.from_numpy(img_arr), fissure_mu, fissure_sigma,
                                        show=show, device=device).cpu()
    print(f'{time.time() - start:.4f} s')
    enhanced_img = sitk.GetImageFromArray(F)
    enhanced_img.CopyInformation(image)

    # apply lung mask
    enhanced_img = apply_mask(enhanced_img, lung_mask)

    if show:
        plot_slice(sitk.GetArrayViewFromImage(enhanced_img)[None, None],
                   s=sitk.GetArrayViewFromImage(enhanced_img).shape[1] // 2, dim=1, title='lung-masked')

    roc_auc = threshold_curves(F.numpy(), fissures_arr, show=True)

    return enhanced_img, roc_auc


def threshold_curves(pred_values: np.ndarray, labels: np.ndarray, out_dir=None, show=False):
    label_names = np.unique(labels)[1:]
    label_names = label_names.tolist() + ['all']

    # flatten all arrays
    labels = labels.flatten()
    pred_values = pred_values.flatten()

    roc_auc = {}
    avg_prec = {}
    roc_display = None
    prc_display = None
    for lbl in label_names:
        if lbl != 'all':
            gt = labels == lbl
            name = f'label {lbl}'
        else:
            gt = labels != 0
            name = 'all labels'

        roc_display = RocCurveDisplay.from_predictions(y_true=gt, y_pred=pred_values, name=name,
                                                       ax=None if roc_display is None else roc_display.ax_)

        # prc_display = PrecisionRecallDisplay.from_predictions(y_true=gt, y_pred=pred_values, name=name,
        #                                                       ax=None if prc_display is None else prc_display.ax_)

        # get area under ROC curve as measurement
        roc_auc[lbl] = roc_display.roc_auc

        # get average precision score
        avg_prec[lbl] = None

    if out_dir is not None:
        roc_display.figure_.savefig(os.path.join(out_dir, 'roc.png'), dpi=300)
        # prc_display.figure_.savefig(os.path.join(out_dir, 'prc.png'), dpi=300)

    if show:
        plt.show()
    else:
        plt.close(roc_display.figure_)
        # plt.close(prc_display.figure_)

    return roc_auc  #, avg_prec


if __name__ == '__main__':
    test_img = sitk.ReadImage('../data/EMPIRE01_img_fixed.nii.gz')
    test_fissures = sitk.ReadImage('../data/EMPIRE01_fissures_poisson_fixed.nii.gz')
    test_mask = sitk.ReadImage('../data/EMPIRE01_mask_fixed.nii.gz')

    # enhanced_img = get_enhanced_fissure_image(test_img, test_fissures, test_mask, resample_spacing=1.5,
    #                                           device='cuda:2', show=False)
    enhanced_img, roc_auc = get_enhanced_fissure_image(test_img, test_fissures, test_mask, resample_spacing=1,
                                                       device='cpu', show=False)
    print(roc_auc)
    sitk.WriteImage(enhanced_img, 'results/EMPIRE01_fixed_fissures_enhanced_GPU.nii.gz')
