from torch import nn


class CorrespondingPointDistance(nn.Module):
    def __init__(self):
        super(CorrespondingPointDistance, self).__init__()

    def forward(self, prediction, target):
        return corresponding_point_distance(prediction, target).mean()


def corresponding_point_distance(prediction, target):
    return (prediction - target).pow(2).sum(-1).sqrt()
