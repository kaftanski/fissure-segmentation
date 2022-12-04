import torch
from torch import nn
from thop import profile, clever_format


def count_parameters(model):
    params = torch.tensor([p.numel() for p in model.parameters() if p.requires_grad]).sum()
    return params


def init_weights(m):
    if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear, nn.modules.conv._ConvTransposeNd)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def params_flops(model, inputs):
    flops, params = profile(model, (inputs, ))
    clever_format([flops, params], "%.3f")
