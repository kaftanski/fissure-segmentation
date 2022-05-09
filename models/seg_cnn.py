from torch import nn

from models.aspp_3d import ASPP
from models.mobilenet import MobileNet3D


def get_mobilenet_aspp_segmodel(num_classes):
    backbone = MobileNet3D()
    aspp = ASPP(64, (2, 4, 8, 16), 128)
    head = nn.Sequential(nn.Conv3d(128 + 16, 64, 1, padding=0, groups=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                         nn.Conv3d(64, 64, 3, groups=1, padding=1, bias=False), nn.BatchNorm3d(64), nn.ReLU(),
                         nn.Conv3d(64, num_classes, 1))

    return nn.Sequential(backbone, aspp, head)
