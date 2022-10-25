from torch import nn

from losses.chamfer_loss import ChamferLoss


class DGSSMLoss(nn.Module):
    def __init__(self, w_point=1., w_coefficients=0.5):
        super(DGSSMLoss, self).__init__()
        # point_loss = CorrespondingPointDistance()
        self.point_loss = ChamferLoss()
        self.coefficient_loss = nn.MSELoss()
        self.w_point = w_point
        self.w_coefficients = w_coefficients

    def forward(self, prediction, target):
        pred_shape, pred_weights = prediction
        targ_shape, targ_weights = target

        # convert predicted shape to pt3d point cloud
        # pl = point_mesh_face_distance(targ_shape, pcl_pred)
        pl = self.point_loss(pred_shape, targ_shape)
        wl = self.coefficient_loss(pred_weights,
                                   targ_weights)  # TODO: is it better to use regularization w^T * Sigma^-1 * w
        return self.w_point * pl + self.w_coefficients * wl, {'Point-Loss': pl, 'Coefficients': wl}


class CorrespondingPointDistance(nn.Module):
    def __init__(self):
        super(CorrespondingPointDistance, self).__init__()

    def forward(self, prediction, target):
        return corresponding_point_distance(prediction, target).pow(2).mean()


def corresponding_point_distance(prediction, target):
    return (prediction - target).pow(2).sum(-1).sqrt()
