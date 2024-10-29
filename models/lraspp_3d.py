from typing import Dict

import torchvision.models
from torch import nn, Tensor
from torch.nn import functional as F
from torchinfo import torchinfo

from models.modelio import LoadableModel, store_config_args
from models.seg_cnn import PatchBasedModule
from thesis.utils import param_and_op_count
from utils.model_utils import init_weights


class LRASPPHead(nn.Module):
    """ implementation from XEdgeConv paper"""
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv3d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv3d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-3:], mode='trilinear', align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


class LRASPP_MobileNetv3_large_3d(LoadableModel):
    @store_config_args
    def __init__(self, num_classes, patch_size=(128, 128, 128)):
        super(LRASPP_MobileNetv3_large_3d, self).__init__()
        self.num_classes = num_classes

        self.backbone = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            pretrained=False, num_classes=self.num_classes).backbone
        self.head = LRASPPHead(low_channels=40, high_channels=960, num_classes=num_classes, inter_channels=128)

        # replace 2D layers with their 3D counterparts
        count = 0
        count2 = 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                before = get_layer(self.backbone, name)
                kernel_size = tuple((list(before.kernel_size) * 2)[:3])
                stride = tuple((list(before.stride) * 2)[:3])
                padding = tuple((list(before.padding) * 2)[:3])
                dilation = tuple((list(before.dilation) * 2)[:3])
                in_channels = before.in_channels
                if in_channels == 3:
                    before.in_channels = 1
                after = nn.Conv3d(before.in_channels, before.out_channels, kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=before.groups)
                set_layer(self.backbone, name, after)
                count += 1

            elif isinstance(module, nn.BatchNorm2d):
                before = get_layer(self.backbone, name)
                after = nn.BatchNorm3d(before.num_features)
                set_layer(self.backbone, name, after)
                count2 += 1

            elif isinstance(module, nn.AdaptiveAvgPool2d):
                before = get_layer(self.backbone, name)
                after = nn.AdaptiveAvgPool3d(before.output_size)
                set_layer(self.backbone, name, after)
                count2 += 1

            elif isinstance(module, nn.Hardswish):
                before = get_layer(self.backbone, name)
                after = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
                set_layer(self.backbone, name, after)
                count2 += 1

        print(count, '# Conv2d > Conv3d and', count2, '#batchnorm etc')

        self.apply(init_weights)

        # patch-based functionality
        self.patching = PatchBasedModule(num_classes, activation=nn.Softmax(dim=1))
        self.patching.forward = self.forward
        self.patch_size = patch_size

    def forward(self, x):
        input = F.interpolate(x, scale_factor=1.5, mode='trilinear', align_corners=False)
        features = self.backbone(x)
        out = self.head(features)
        out = F.interpolate(out, size=(int(input.shape[2] / 1.5), int(input.shape[3] / 1.5), int(input.shape[4] / 1.5)),
                            mode='trilinear', align_corners=False)
        return out

    def predict_all_patches(self, img, min_overlap=0.5, use_gaussian=True):
        return self.patching.predict_all_patches(
            img, patch_size=self.patch_size, min_overlap=min_overlap, use_gaussian=use_gaussian)


if __name__ == '__main__':
    model = LRASPP_MobileNetv3_large_3d(num_classes=3)
    print(model)
    torchinfo.summary(model, (1, 1, 128, 128, 128))

    class PatchWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.predict_all_patches(x)


    param_and_op_count(PatchWrapper(model),)
