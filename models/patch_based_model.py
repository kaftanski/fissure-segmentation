import math

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D
from models.modelio import LoadableModel, store_config_args


class PatchBasedModule(nn.Module):
    def __init__(self, num_classes, activation=lambda x: x):
        super().__init__()
        self.num_classes = num_classes
        self.gaussian_weight_map = None
        self.activation = activation

    def predict_all_patches(self, img, patch_size=(128, 128, 128), min_overlap=0.5, use_gaussian=True):
        assert len(img.shape)-2 == len(patch_size)
        assert 0 <= min_overlap < 1

        patch_starts = get_patch_starts(img.shape[2:], min_overlap, patch_size)

        print(img.shape[2:])
        print(patch_starts)

        # extract the patches and forward them through the network
        output = torch.zeros(img.shape[0], self.num_classes, *img.shape[2:], device=img.device)
        output_normalization_map = torch.zeros_like(output, device=img.device)
        for start_x in patch_starts[0]:
            for start_y in patch_starts[1]:
                for start_z in patch_starts[2]:
                    print(f'Computing patch starting at ({start_x}, {start_y}, {start_z})')

                    patch_region = (slice(None), slice(None),
                                    slice(start_x, start_x+patch_size[0]),
                                    slice(start_y, start_y+patch_size[1]),
                                    slice(start_z, start_z+patch_size[2]))

                    img_patch = img[patch_region]

                    before_padding = img_patch.shape[2:]
                    img_patch = maybe_pad_img_patch(img_patch, patch_shape=patch_size)
                    out_patch = self.activation(self(img_patch))

                    if use_gaussian:
                        # gaussian importance weighting
                        gaussian_map = self._get_gaussian(patch_size, img.device, sigma_scale=1/4.)
                        out_patch *= gaussian_map
                        output_normalization_map[patch_region] += maybe_crop_after_padding(gaussian_map, before_padding)
                    else:
                        output_normalization_map[patch_region] += 1

                    out_patch = maybe_crop_after_padding(out_patch, before_padding)
                    output[patch_region] += out_patch

        output /= output_normalization_map
        return self.activation(output)

    def _get_gaussian(self, patch_size, device, sigma_scale=1/8.):
        if self.gaussian_weight_map is None or \
                any(shape != patch for shape, patch in zip(self.gaussian_weight_map.shape[2:], patch_size)):

            weight_map = np.zeros(patch_size)
            center_coord = tuple(p//2 for p in patch_size)
            weight_map[center_coord] = 1
            # gaussian smoothing of the dirac impulse
            # weight_map = smooth(weight_map, sigmas=[p * sigma_scale for p in patch_size])
            weight_map = gaussian_filter(weight_map, sigma=[p * sigma_scale for p in patch_size], order=0,
                                         mode='constant', cval=0)

            # prevent NaNs by converting zero values
            weight_map[weight_map == 0] = weight_map[weight_map != 0].min()

            self.gaussian_weight_map = torch.from_numpy(weight_map).unsqueeze(0).unsqueeze(0).to(device)

        elif self.gaussian_weight_map.device != torch.device(device):
            self.gaussian_weight_map = self.gaussian_weight_map.to(device)

        return self.gaussian_weight_map


def get_patch_starts(img_size, min_overlap, patch_size):
    patch_size = list(patch_size)
    patch_starts = []
    for i, (dim, patch) in enumerate(zip(img_size, patch_size)):
        if patch >= dim:
            # patch size exceeding image bounds (img will be padded later)
            patch_starts.append([0])
        else:
            steps = math.ceil((dim - patch * min_overlap) / (patch - patch * min_overlap))
            actual_overlap = (steps * patch - dim) / (steps - 1)
            patch_starts.append([math.floor(s * (patch - actual_overlap) + 0.5) for s in range(steps)])

    return patch_starts


def get_necessary_padding(img_dimensions, out_shape):
    pad = []
    for dim in range(len(out_shape)-1, -1, -1):
        residual = out_shape[dim] - img_dimensions[dim]
        if residual > 0:
            pad.append(residual // 2 + (1 if residual / 2 - residual // 2 == 0.5 else 0))
            pad.append(residual // 2)
        else:
            pad.extend([0, 0])

    if all(p == 0 for p in pad):
        return None
    else:
        return pad


def maybe_pad_img_patch(img, patch_shape):
    pad = get_necessary_padding(img.shape[2:], patch_shape)
    if pad is not None:
        img = F.pad(img, pad, mode='replicate')
    return img


def maybe_crop_after_padding(img, orig_dimensions):
    pad = get_necessary_padding(img_dimensions=orig_dimensions, out_shape=img.shape[2:])
    if pad is not None:
        pad = pad[::-1]
        crop = []
        for dim, padded in enumerate(img.shape[2:]):
            p2, p1 = pad[dim*2:dim*2+2]
            crop.append(slice(p1, padded-p2))
        img = img[2*[slice(None)] + crop]
    return img
