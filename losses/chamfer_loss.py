from pytorch3d.loss import chamfer_distance
from torch import nn


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, prediction, target):
        if prediction.shape[1] == 3:
            # format: B x 3 x N_points
            prediction = prediction.transpose(1, 2)

        if target.shape[1] == 3:
            # format: B x 3 x N_points
            target = target.transpose(1, 2)
        assert prediction.shape[0] == target.shape[0] and prediction.shape[2] == target.shape[2]

        loss, _ = chamfer_distance(prediction, target)
        return loss
