import torch
from torch import nn

from losses.dice_loss import GDL
from enum import Enum

from losses.recall_loss import BatchRecallLoss


class Losses(Enum):
    NNUNET = "nnunet"
    """default loss function used in the nnU-Net: CE + Dice loss"""

    CE = "ce"
    """normal cross entropy loss"""

    RECALL = "recall"
    """cross entropy loss weighted with batch-specific false-positive rate, promotes recall"""

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def assemble_nnunet_loss_function(class_weights: torch.Tensor = None):
    ce_loss = nn.CrossEntropyLoss(class_weights)
    dice_loss = GDL(apply_nonlin=nn.Softmax(dim=1), batch_dice=True)

    def combined_loss(prediction, target):
        ce = ce_loss(prediction, target)
        dice = dice_loss(prediction, target)
        return ce + dice, {'CE': ce, 'GDL': dice}

    return combined_loss


def get_loss_fn(loss: Losses, class_weights: torch.Tensor = None):
    if isinstance(loss, Losses):
        loss = loss.value

    if loss == Losses.NNUNET.value:
        return assemble_nnunet_loss_function(class_weights)

    if loss == Losses.CE.value:
        return nn.CrossEntropyLoss(class_weights)

    if loss == Losses.RECALL.value:
        return BatchRecallLoss()

    raise ValueError(f'No loss function named "{loss}". Please choose one from {Losses.list()} instead.')
