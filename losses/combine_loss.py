import torch
from torch import nn

from losses.dice_loss import GDL


def assemble_segmentation_loss_function(class_weights: torch.Tensor = None):
    ce_loss = nn.CrossEntropyLoss(class_weights)
    dice_loss = GDL(apply_nonlin=nn.Softmax(dim=1), batch_dice=True)

    def combined_loss(prediction, target):
        ce = ce_loss(prediction, target)
        dice = dice_loss(prediction, target)
        return ce + dice, {'CE': ce, 'GDL': dice}

    return combined_loss
