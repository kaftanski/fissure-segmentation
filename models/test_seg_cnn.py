from unittest import TestCase

from models.patch_based_model import get_patch_starts


class Test(TestCase):
    def test_get_patch_starts(self):
        self.get_and_assert_patch_starts(img_size=(128, 128, 128), patch_size=(64, 64, 64), min_overlap=0.3)

    def test_get_patch_starts_half_overlap(self):
        self.get_and_assert_patch_starts(img_size=(300, 158, 113), patch_size=(53, 72, 15), min_overlap=0.5)

    def test_get_patch_starts_no_overlap(self):
        self.get_and_assert_patch_starts(img_size=(268, 483, 126), patch_size=(123, 45, 18), min_overlap=0)

    def test_get_patch_starts_small_overlap(self):
        self.get_and_assert_patch_starts(img_size=(268, 483, 126), patch_size=(123, 45, 18), min_overlap=0.001)

    def test_get_patch_starts_fissure_enhancement_use_case(self):
        self.get_and_assert_patch_starts(img_size=(315, 334, 334), patch_size=(64, 64, 64), min_overlap=0.01)

    def test_get_patch_starts_3d_cnn_use_case(self):
        self.get_and_assert_patch_starts(img_size=(259, 342, 256), patch_size=(128, 128, 128), min_overlap=0.01)

    def get_and_assert_patch_starts(self, img_size, patch_size, min_overlap):
        patch_starts = get_patch_starts(img_size, min_overlap, patch_size)
        for d in range(len(patch_starts)):
            self.assertEqual(img_size[d], patch_starts[d][-1] + patch_size[d])
            #print(patch_starts[d], patch_size[d], img_size[d])
            for i in range(1, len(patch_starts[d])):
                real_overlap = patch_starts[d][i-1] + patch_size[d] - patch_starts[d][i]
                #print(patch_size[d] * min_overlap, real_overlap)
                self.assertGreaterEqual(real_overlap, patch_size[d] * min_overlap,
                                        msg=f'Real overlap smaller than the minimum [dim {d}, patch #{i+1}/{len(patch_starts[d])}. \n\tAll patch starts: {patch_starts[d]}')
