# Mobile-Net with depth-separable convolutions and residual connections
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResBlock(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class MobileNet3D(nn.Module):
    def __init__(self):
        super(MobileNet3D, self).__init__()

        in_channels = torch.Tensor([1, 16, 24, 24, 32, 32, 32, 64]).long()
        mid_channels = torch.Tensor([32, 96, 144, 144, 192, 192, 192, 384]).long()
        out_channels = torch.Tensor([16, 24, 24, 32, 32, 32, 64, 64]).long()
        mid_stride = torch.Tensor([1, 1, 1, 1, 1, 2, 1, 1])

        layers = [nn.Identity()]
        for i in range(len(in_channels)):
            inc = int(in_channels[i])
            midc = int(mid_channels[i])
            outc = int(out_channels[i])
            strd = int(mid_stride[i])
            layer = nn.Sequential(nn.Conv3d(inc, midc, 1, bias=False), nn.BatchNorm3d(midc), nn.ReLU6(True),
                                  nn.Conv3d(midc, midc, 3, stride=strd, padding=1, bias=False, groups=midc),
                                  nn.BatchNorm3d(midc), nn.ReLU6(True),
                                  nn.Conv3d(midc, outc, 1, bias=False), nn.BatchNorm3d(outc))
            if (i == 0):
                layer[0] = nn.Conv3d(inc, midc, 3, padding=1, stride=2, bias=False)
            if ((inc == outc) & (strd == 1)):
                layers.append(ResBlock(layer))
            else:
                layers.append(layer)

        self.layers = nn.Sequential(*layers)

        # init weights
        self._init_weights()

    def forward(self, x):
        x = checkpoint(self.layers[:2], x)
        x2 = x

        x2 = self.layers[2:](x2)
        return x, x2

    def _init_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


if __name__ == '__main__':
    import torchinfo
    m = MobileNet3D()
    torchinfo.summary(m, (1, 1, 128, 128, 128))
