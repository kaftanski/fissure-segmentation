import csv
import os
import time
import warnings

import SimpleITK as sitk
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import _gaussian_kernel1d
from skimage.filters.ridges import compute_hessian_eigenvalues
from sklearn.metrics import RocCurveDisplay
from torch import nn

from data import ImageDataset
from metrics import binary_recall, batch_dice
from models.seg_cnn import PatchBasedModule
from utils.image_ops import resample_equal_spacing, apply_mask
from utils.image_utils import filter_1d
from utils.tqdm_utils import tqdm_redirect
from utils.utils import new_dir
from visualization import plot_slice


class HessianEnhancementFilter(PatchBasedModule):
    def __init__(self, fissure_mu, fissure_sigma, gaussian_smoothing_sigma=1., gaussian_derivation_sigma=1., show=False):
        super(HessianEnhancementFilter, self).__init__(1)

        self.fissure_mu = fissure_mu
        self.fissure_sigma = fissure_sigma
        self.show = show

        self.register_parameter('kernel_1st_deriv',
                                nn.Parameter(
                                    torch.from_numpy(gaussian_kernel_1d(gaussian_derivation_sigma, order=1)).float(),
                                    requires_grad=False))

        self.register_parameter('kernel_2nd_deriv',
                                nn.Parameter(
                                    torch.from_numpy(gaussian_kernel_1d(gaussian_derivation_sigma, order=2)).float(),
                                    requires_grad=False))

        self.register_parameter('kernel_gaussian_smoothing',
                                nn.Parameter(
                                    torch.from_numpy(gaussian_kernel_1d(gaussian_smoothing_sigma)).float(),
                                    requires_grad=False))

    def forward(self, img):
        # smooth image with gaussian kernel
        img_smooth = img
        for dim in range(len(img.shape)-2):
            img_smooth = filter_1d(img_smooth, self.kernel_gaussian_smoothing, dim)

        # compute hessian matrix
        H = self.compute_hessian_matrix(img)

        # get hessian eigenvalues
        eigenvalues = torch.linalg.eigvalsh(H)

        # sort by absolute value in descending order
        abs_sorting = torch.argsort(torch.abs(eigenvalues), dim=-1, descending=True)
        eigenvalues = torch.gather(eigenvalues, dim=-1, index=abs_sorting)

        # compute the fissure enhanced image
        fissure_enhanced = fissure_filter(img.squeeze(), eigenvalues[..., 0], eigenvalues[..., 1], self.fissure_mu,
                                          self.fissure_sigma, show=self.show).unsqueeze(0).unsqueeze(0)
        if torch.any(torch.isnan(fissure_enhanced)) or torch.any(torch.isinf(fissure_enhanced)):
            warnings.warn('NaN or inf value in fissure enhancement image')

        return fissure_enhanced

    def compute_hessian_matrix(self, img):
        H = torch.zeros(*img.squeeze().shape, 3, 3, device=img.device)
        for first_dim in range(len(img.squeeze().shape)):
            for second_dim in range(len(img.squeeze().shape)):
                if first_dim == second_dim:
                    # filter dimension with 2nd derivation of gaussian kernel
                    H[..., first_dim, second_dim] = filter_1d(img, self.kernel_2nd_deriv, dim=first_dim).squeeze()

                elif second_dim > first_dim:
                    # filter with 1st derivation of gaussian kernel in both dimensions
                    img_deriv = filter_1d(
                        filter_1d(img, self.kernel_1st_deriv, dim=first_dim),
                        self.kernel_1st_deriv, dim=second_dim).squeeze()

                    # differentiation is linear -> sequence does not matter
                    H[..., first_dim, second_dim] = img_deriv
                    H[..., second_dim, first_dim] = img_deriv

        return H


def gaussian_kernel_1d(sigma, order=0, truncate=4.0):
    sigma = float(sigma)
    radius = int(truncate * sigma + 0.5)
    kernel = _gaussian_kernel1d(sigma, order, radius)
    return kernel


def hessian_matrix(img: torch.Tensor, sigma: float):
    kernel_1st_deriv = gaussian_kernel_1d(sigma, order=1)
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

    hessian_filter = HessianEnhancementFilter(fissure_mu, fissure_sigma,
                                              gaussian_smoothing_sigma=1., gaussian_derivation_sigma=1., show=False)
    hessian_filter.to(device)

    if device == 'cpu':
        # no patch-based prediction needed
        fissures_enhanced = hessian_filter(img)
    else:
        fissures_enhanced = hessian_filter.predict_all_patches(img, min_overlap=0.25, patch_size=(64, 64, 64))

    return fissures_enhanced.squeeze()


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


def get_enhanced_fissure_image(image: sitk.Image, fissures: sitk.Image, lung_mask: sitk.Image, device='cuda:2', show=False):
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

    return enhanced_img


def fissure_candidates(enhanced_img: sitk.Image, gt_fissures: sitk.Image, fixed_thresh: float = None, show=False,
                       img_dir: str = None, img_prefix = ''):
    enhanced_img_arr = sitk.GetArrayFromImage(enhanced_img)
    gt_fissures_arr = sitk.GetArrayFromImage(gt_fissures)
    gt_fissures_binary = torch.from_numpy(gt_fissures_arr != 0)

    # receiver operator characteristics curve
    roc_auc = threshold_curves(enhanced_img_arr, gt_fissures_arr, show=show)

    # evaluate result based on different thresholds
    thresholds = torch.linspace(0, 1, steps=21) if fixed_thresh is None else [fixed_thresh]
    dices = []
    recalls = []
    for t in thresholds:
        fissure_prediction = sitk.BinaryThreshold(enhanced_img, upperThreshold=t.item(), insideValue=0, outsideValue=1)
        fissure_prediction_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissure_prediction).astype(bool))

        dices.append(batch_dice(fissure_prediction_tensor[None], gt_fissures_binary[None], n_labels=2)[1])  # foreground
        recalls.append(binary_recall(fissure_prediction_tensor[None], gt_fissures_binary[None])[0])

    fig = plt.figure()
    plt.plot(thresholds, recalls, label='recall')
    plt.plot(thresholds, dices, label='dice')
    plt.title('thresholding fissure-enhanced image')
    plt.xlabel('threshold')
    plt.legend()

    if img_dir is not None:
        fig.savefig(os.path.join(img_dir, f'{img_prefix}metrics_per_threshold.png'), dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return roc_auc, thresholds.numpy(), torch.stack(dices, dim=0).numpy(), torch.stack(recalls, dim=0).numpy()


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


def enhance_full_dataset(ds: ImageDataset, out_dir: str, eval_dir: str, resample_spacing: float = None, show=False,
                         device='cuda:0'):
    new_dir(out_dir)
    new_dir(eval_dir)
    all_roc_aucs = []
    all_dsc = []
    all_rec = []
    for i in tqdm_redirect(range(len(ds))):
        # load data
        patid = ds.get_id(i)
        img = ds.get_image(i)
        fissures = ds.get_regularized_fissures(i)
        lung_mask = ds.get_lung_mask(i)

        # preprocess
        if resample_spacing is not None:
            img = resample_equal_spacing(img, target_spacing=resample_spacing)
            fissures = resample_equal_spacing(fissures, target_spacing=resample_spacing, use_nearest_neighbor=True)
            lung_mask = resample_equal_spacing(lung_mask, target_spacing=resample_spacing, use_nearest_neighbor=True)

        # fissure enhancement
        enhanced_img = get_enhanced_fissure_image(img, fissures, lung_mask, device=device, show=show)
        sitk.WriteImage(enhanced_img, os.path.join(out_dir, f'{patid[0]}_fissures_enhanced_{patid[1]}.nii.gz'))

        # evaluation
        roc_auc, thresh, dsc, rec = fissure_candidates(enhanced_img, fissures, show=show,
                                                       img_dir=eval_dir, img_prefix=f'{patid[0]}_{patid[1]}_')
        all_roc_aucs.append(roc_auc)
        all_dsc.append(dsc)
        all_rec.append(rec)

    all_rec = np.stack(all_rec, axis=0)
    all_dsc = np.stack(all_dsc, axis=0)
    with open(os.path.join(eval_dir, 'results.csv'), 'w') as result_csv_file:
        writer = csv.writer(result_csv_file)
        writer.writerow(['Mean ROC-AUC'] + [np.array([roc_auc['all'] for roc_auc in all_roc_aucs]).mean()])
        writer.writerow(['Per-Threshold'] + thresh.tolist())
        writer.writerow(['Recall'] + all_rec.mean(0).tolist())
        writer.writerow(['Dice'] + all_dsc.mean(0).tolist())


if __name__ == '__main__':
    # test_img = sitk.ReadImage('../data/EMPIRE01_img_fixed.nii.gz')
    # test_fissures = sitk.ReadImage('../data/EMPIRE01_fissures_poisson_fixed.nii.gz')
    # test_mask = sitk.ReadImage('../data/EMPIRE01_mask_fixed.nii.gz')
    #
    # resample_spacing = None
    # if resample_spacing is not None:
    #     image = resample_equal_spacing(test_img, target_spacing=resample_spacing)
    #     fissures = resample_equal_spacing(test_fissures, target_spacing=resample_spacing, use_nearest_neighbor=True)
    #     lung_mask = resample_equal_spacing(test_mask, target_spacing=resample_spacing, use_nearest_neighbor=True)
    #
    # enhanced_img = get_enhanced_fissure_image(
    #     test_img, test_fissures, test_mask, device='cuda:2', show=False)
    # roc_auc, thresh, dsc, rec = fissure_candidates(enhanced_img, test_fissures, show=True)
    # print(roc_auc)
    # sitk.WriteImage(enhanced_img, 'results/EMPIRE01_fixed_fissures_enhanced_patch.nii.gz')
    #
    # enhanced_img = get_enhanced_fissure_image(
    #     test_img, test_fissures, test_mask, device='cpu', show=False)
    # roc_auc, thresh, dsc, rec = fissure_candidates(enhanced_img, test_fissures, show=True)
    # print(roc_auc)
    # sitk.WriteImage(enhanced_img, 'results/EMPIRE01_fixed_fissures_enhanced_torch.nii.gz')

    ds = ImageDataset('../TotalSegmentator/ThoraxCrop', do_augmentation=False)
    out_dir = new_dir('results', 'hessian_fissure_enhancement', 'TotalSegmentator')
    eval_dir = new_dir(out_dir, 'eval')
    enhance_full_dataset(ds, out_dir=out_dir, eval_dir=eval_dir, resample_spacing=1.5, show=True, device='cuda:1')
