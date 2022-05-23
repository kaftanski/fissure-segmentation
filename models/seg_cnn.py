import torch
from torch import nn
import torch.nn.functional as F
from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D
from models.modelio import LoadableModel, store_config_args
from torch.utils.checkpoint import checkpoint
import math


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

    def predict_all_patches(self, img, patch_size, min_overlap=0.25):
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
        for start_x in patch_starts[0]:
            for start_y in patch_starts[1]:
                for start_z in patch_starts[2]:
                    print(f'Computing patch starting at ({start_x}, {start_y}, {start_z})')
                    img_patch = img[..., start_x:start_x+patch_size[0],
                                    start_y:start_y+patch_size[1],
                                    start_z:start_z+patch_size[2]]

                    before_padding = img_patch.shape[2:]
                    img_patch = maybe_pad_img_patch(img_patch, patch_shape=patch_size,
                                                    pad_value=-1)  # hard-coded for CT background
                    out_patch = F.softmax(self(img_patch), dim=1)
                    out_patch = maybe_crop_after_padding(out_patch, before_padding)

                    output[..., start_x:start_x+patch_size[0],
                           start_y:start_y+patch_size[1],
                           start_z:start_z+patch_size[2]] += out_patch

        # TODO: gaussian importance weighting (?)
        return F.softmax(output, dim=1)


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


def maybe_pad_img_patch(img, patch_shape, pad_value=-1):
    pad = get_necessary_padding(img.shape[2:], patch_shape)
    if pad is not None:
        img = F.pad(img, pad, mode='constant', value=pad_value)
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
