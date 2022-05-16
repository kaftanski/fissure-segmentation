from matplotlib import pyplot as plt

from data import ImageDataset
from image_ops import sitk_image_to_tensor, resample_equal_spacing
from visualization import visualize_with_overlay

ds = ImageDataset('../data', patch_scaling=0.5)
test_img = 0
for i in range(20):
    img, label = ds[test_img]
    img = img.squeeze()
    label = label.squeeze()

    img_orig_resample = sitk_image_to_tensor(resample_equal_spacing(ds.get_image(test_img), target_spacing=1.5))

    fig, ax = plt.subplots(2, 3)

    for slc in range(3):
        ax[0, slc].imshow(img_orig_resample[[slice(None)] * slc + [img_orig_resample.shape[slc] // 2]].squeeze(), cmap='gray')
        visualize_with_overlay(img[[slice(None)] * slc + [img.shape[slc] // 2]].squeeze(),
                               label[[slice(None)] * slc + [img.shape[slc] // 2]].squeeze(),
                               ax=ax[1, slc])

    plt.show()
