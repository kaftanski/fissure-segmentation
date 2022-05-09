import torch
from torch import nn


def count_parameters(model):
    params = torch.tensor([p.numel() for p in model.parameters() if p.requires_grad]).sum()
    return params


def init_weights(m):
    if isinstance(m, (nn.modules.conv._ConvNd, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
