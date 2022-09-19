import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters.ridges import compute_hessian_eigenvalues

from visualization import plot_slice


def hessian_based_enhancement(img: np.ndarray, fissure_mu: float, fissure_sigma: float, show=False):
    """ R. Wiemker et al.:
    "Unsupervised extraction of the pulmonary interlobar fissures from high resolution thoracic CT data"

    :return
    """
    # smooth image
    img_smooth = gaussian_filter(img, sigma=1)

    # normalize image
    img_norm = (img_smooth.astype(float) - img_smooth.min()) / (img_smooth.max() - img_smooth.min())

    # eigenvalues are sorted in absolute ascending order, we want descending
    lambda3, lambda2, lambda1 = compute_hessian_eigenvalues(img_norm, sigma=1.,  # TODO: more scales (more sigmas)
                                                            sorting='abs', mode='reflect')

    # compute planeness value
    P = np.zeros_like(lambda1)
    abs_lambda1 = np.abs(lambda1)
    abs_lambda2 = np.abs(lambda2)
    P[lambda1 < 0] = (abs_lambda1[lambda1 < 0] - abs_lambda2[lambda1 < 0]) \
                     / (abs_lambda1[lambda1 < 0] + abs_lambda2[lambda1 < 0])
    # P[lambda1 >= 0] = 0

    # compute fissure likeliness based on Hounsfield units
    hu_weights = np.exp(-((img - fissure_mu)**2) / (2 * fissure_sigma**2))

    # full fissure filter
    F = hu_weights * P

    # visualize
    if show:
        dim = 1
        plot_slice(P[None, None], s=P.shape[dim]//2, dim=dim, title='plane-ness')
        plot_slice(F[None, None], s=F.shape[dim]//2, dim=dim, title='fissure-ness')
        plot_slice(hu_weights[None, None], s=hu_weights.shape[dim] // 2, dim=dim, title='HU weights')

    return F


if __name__ == '__main__':
    test_img = sitk.ReadImage('../data/EMPIRE01_img_fixed.nii.gz')
    test_img_arr = sitk.GetArrayFromImage(test_img)
    test_fissures = sitk.GetArrayFromImage(sitk.ReadImage('../data/EMPIRE01_fissures_poisson_fixed.nii.gz'))

    # compute fissure HU statistics
    fissure_mu = test_img_arr[test_fissures != 0].mean()
    fissure_sigma = test_img_arr[test_fissures != 0].std()

    # fissure enhancement
    F = hessian_based_enhancement(test_img_arr, fissure_mu, fissure_sigma, show=True)
    enhanced_img = sitk.GetImageFromArray(F)
    enhanced_img.CopyInformation(test_img)
    sitk.WriteImage(enhanced_img, 'results/EMPIRE01_fixed_fissures_enhanced.nii.gz')
