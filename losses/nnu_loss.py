from torch import nn

from losses.dice_loss import GDL


class NNULoss(nn.Module):
    """ combination of Dice and cross-entropy loss as used in the nnu-net """
    def __init__(self, class_weights, w_dice=1, w_ce=1):
        super().__init__()
        self.w_dice = w_dice
        self.w_ce = w_ce

        self.ce_loss = nn.CrossEntropyLoss(class_weights)
        self.dice_loss = GDL(apply_nonlin=nn.Softmax(dim=1), batch_dice=True)

    def forward(self, prediction, target):
        ce = self.ce_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return ce + dice, {'CE': ce, 'GDL': dice}
