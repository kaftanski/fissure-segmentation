import torch
from torch import nn
import torch.nn.functional as F
from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D
from models.modelio import LoadableModel, store_config_args
from torch.utils.checkpoint import checkpoint


class MobileNetASPP(LoadableModel):
    @store_config_args
    def __init__(self, num_classes):
        super(MobileNetASPP, self).__init__()

        self.backbone = MobileNet3D()
        self.aspp = ASPP(64, (2, 4, 8, 16), 128)
        self.head = nn.Sequential(nn.Conv3d(128 + 16, 64, 1, padding=0, groups=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                                  nn.Conv3d(64, 64, 3, groups=1, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                                  nn.Conv3d(64, num_classes, 1))

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

# TODO: inference from patches (stitched together overlapping patches, with gaussian weighting a la nnUnet)


if __name__ == '__main__':
    import torchinfo
    m = MobileNetASPP(3)
    torchinfo.summary(m, (2, 1, 128, 128, 128))
