from unittest import TestCase

import torch

from utils.utils import sample_patches_at_kpts, kpts_to_grid, ALIGN_CORNERS


class Test(TestCase):
    def setUp(self) -> None:
        # torch.manual_seed(42)
        # self.example_image = torch.randint(low=-1024, high=2000, size=(128, 128, 128), dtype=torch.float)
        # self.kpts_world = torch.nonzero(self.example_image)[::10000]
        # self.kpts_grid = kpts_to_grid(self.kpts_world, self.example_image.shape, align_corners=True)
        self.example_image = torch.arange(5**3, dtype=torch.float).view(1, 1, 5, 5, 5)
        self.all_points_world = torch.nonzero(torch.ones(5, 5, 5))
        self.all_points_grid = kpts_to_grid(self.all_points_world, self.example_image.shape[2:], align_corners=ALIGN_CORNERS)
        self.center_point_grid = self.all_points_grid[None, self.all_points_grid.shape[0] // 2]

    def test_sample_patches_at_kpts(self):
        center_patch = sample_patches_at_kpts(self.example_image, self.center_point_grid, patch_size=5)
        self.assertListEqual(self.example_image.tolist(), center_patch.tolist())

    def test_sample_patches_at_kpts_exception(self):
        with self.assertRaises(ValueError):
            sample_patches_at_kpts(self.example_image, self.all_points_world, patch_size=5)

    def test_sample_patches_at_all_points(self):
        all_patches = sample_patches_at_kpts(self.example_image, self.all_points_grid, patch_size=5)
        target_shape = list(self.example_image.shape)
        target_shape[1] = self.all_points_grid.shape[0]
        self.assertEqual(torch.Size(target_shape), all_patches.shape)

        center_patch = all_patches[:, all_patches.shape[1]//2]
        self.assertTrue(torch.all(center_patch == self.example_image),
                        f'difference tensor: {center_patch-self.example_image}')

    def test_sample_patches_at_kpts_big_img(self):
        torch.manual_seed(42)
        im_size = 128
        radius = 3
        example_image = torch.randint(low=-1024, high=2000, size=(1, 1, im_size, im_size+20, im_size+10), dtype=torch.float)
        kpts_world = torch.nonzero(example_image.squeeze())[::10000]
        kpts_world = torch.clamp(kpts_world, radius, im_size-radius-1)  # no padding required
        kpts_grid = kpts_to_grid(kpts_world, example_image.shape[2:], align_corners=ALIGN_CORNERS)

        patches = sample_patches_at_kpts(example_image, kpts_grid, patch_size=2*radius + 1)

        for p in range(10, 110):
            point = kpts_world[p]
            patch = patches[0, p]
            self.assertListEqual(example_image[0, 0,
                                               point[2]-radius:point[2]+radius+1,
                                               point[1]-radius:point[1]+radius+1,
                                               point[0]-radius:point[0]+radius+1].tolist(), patch.tolist())

