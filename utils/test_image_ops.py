from unittest import TestCase

import SimpleITK as sitk
import numpy as np

from utils.image_ops import apply_mask


class Test(TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        self.im_size = (25, 25, 25)

        self.mask_arr = np.zeros(self.im_size, dtype=bool)
        self.mask_arr[3:15, 3:15, 3:15] = 1
        self.mask = sitk.GetImageFromArray(self.mask_arr.astype(int))

    def test_apply_mask_short(self):
        short_image_arr = np.random.randint(0, 255, size=self.im_size)
        self.asserts_for_apply_mask(short_image_arr)

    def test_apply_mask_float_img(self):
        float_img_arr = np.random.randn(*self.im_size)
        self.asserts_for_apply_mask(float_img_arr)

    def asserts_for_apply_mask(self, img_arr):
        img = sitk.GetImageFromArray(img_arr)

        masked = apply_mask(img, self.mask)
        masked_arr = sitk.GetArrayFromImage(masked)
        self.assertTrue(np.all(masked_arr[self.mask_arr] == img_arr[self.mask_arr]))

        inv_mask = np.logical_not(self.mask_arr)
        self.assertTrue(np.all(masked_arr[inv_mask] == 0))

        self.assertEqual(masked.GetPixelID(), img.GetPixelID())
