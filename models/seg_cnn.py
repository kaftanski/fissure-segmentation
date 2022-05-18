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
                # prevent patch size exceeding image bounds
                patch_size[i] = dim
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

                    output[..., start_x:start_x+patch_size[0],
                           start_y:start_y+patch_size[1],
                           start_z:start_z+patch_size[2]] += F.softmax(self(img_patch), dim=1)

        return F.softmax(output, dim=1)


# TODO: inference from patches (stitched together overlapping patches, with gaussian weighting a la nnUnet)


if __name__ == '__main__':
    import torchinfo
    m = MobileNetASPP(3)
    torchinfo.summary(m, (2, 1, 128, 128, 128))
