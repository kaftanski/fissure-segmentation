from torch import nn

from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D
from models.modelio import LoadableModel, store_config_args


class MobileNetASPP(LoadableModel):
    @store_config_args
    def __init__(self, num_classes):
        super(MobileNetASPP, self).__init__()

        backbone = MobileNet3D()
        aspp = ASPP(64, (2, 4, 8, 16), 128)
        head = nn.Sequential(nn.Conv3d(128 + 16, 64, 1, padding=0, groups=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                             nn.Conv3d(64, 64, 3, groups=1, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                             nn.Conv3d(64, num_classes, 1))

        self.model = nn.Sequential(backbone, aspp, head)

    def forward(self, x):
        return self.model(x)  # TODO: use checkpoints to reduce VRAM if necessary

# TODO: inference from patches (stitched together overlapping patches, with gaussian weighting a la nnUnet)
