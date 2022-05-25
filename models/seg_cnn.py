import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D
from models.modelio import LoadableModel, store_config_args


class MobileNetASPP(LoadableModel):
    @store_config_args
    def __init__(self, num_classes):
        super(MobileNetASPP, self).__init__()

        self.num_classes = num_classes

        self.backbone = MobileNet3D()
        self.aspp = ASPP(64, (2, 4, 8, 16), 128)
        self.head = nn.Sequential(nn.Conv3d(128 + 16, 64, 1, padding=0, groups=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                                  nn.Conv3d(64, 64, 3, groups=1, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                                  nn.Conv3d(64, self.num_classes, 1))

        self.gaussian_weight_map = None

    def forward(self, x):
        if self.training:
            # necessary for running backwards through checkpointing
            x.requires_grad = True

        x1, x2 = checkpoint(self.backbone, x, preserve_rng_state=False)
        y = checkpoint(self.aspp, x2, preserve_rng_state=False)
        y = torch.cat([x1, F.interpolate(y, scale_factor=2)], dim=1)
        y1 = checkpoint(self.head, y, preserve_rng_state=False)
        output = F.interpolate(y1, scale_factor=2, mode='trilinear', align_corners=False)
        return output

    def predict_all_patches(self, img, patch_size, min_overlap=0.5, use_gaussian=False):
        assert len(img.shape)-2 == len(patch_size)
        assert 0 <= min_overlap < 1

        patch_size = list(patch_size)
        patch_starts = []
        for i, (dim, patch) in enumerate(zip(img.shape[2:], patch_size)):
            if patch >= dim:
                # patch size exceeding image bounds (img will be padded later)
                patch_starts.append([0])
            else:
                steps = math.ceil((dim - patch*min_overlap) / (patch - patch * min_overlap))
                real_overlap = math.ceil(-(dim-steps*patch) / (steps-1))
                assert real_overlap >= patch * min_overlap

                residual = dim - (patch * steps - real_overlap * (steps - 1))
                patch_starts.append([s*(patch-real_overlap) for s in range(steps)])
                patch_starts[-1][-1] += residual
                assert patch_starts[-1][-1] + patch == dim

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
                    out_patch = F.softmax(self(img_patch), dim=1)

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
        return F.softmax(output, dim=1)

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


if __name__ == '__main__':
    import torchinfo
    m = MobileNetASPP(3)
    torchinfo.summary(m, (2, 1, 128, 128, 128))
