import torch
from torch import nn
from torch.nn import functional as F

from metrics import binary_recall


class BatchRecallLoss(nn.Module):
    """ Weighted cross-entropy loss that promotes recall.
    Weights per class are the false-negative rates in the current batch.
    Idea from https://openreview.net/pdf?id=SlprFTIQP3. """
    def __init__(self):
        super(BatchRecallLoss, self).__init__()

    def forward(self, prediction, target):
        assert len(prediction.shape) == len(target.shape) + 1, \
            'BatchRecallLoss takes the same input shapes as nn.CrossEntropyLoss: (N, C, ...) and (N, ...)'

        num_classes = prediction.shape[1]
        weight = torch.zeros(num_classes, device=prediction.device)
        with torch.no_grad():
            prediction_map = torch.argmax(prediction, dim=1)
            for lbl in range(num_classes):
                recall = binary_recall(prediction_map == lbl, target == lbl).mean(0)
                weight[lbl] = 1 - recall  # = false negative rate (FN/(TP+FN))

        return F.cross_entropy(prediction, target, weight=weight)
