from torch import nn

from augmentations import compose_transform
from losses.chamfer_loss import ChamferLoss


class DGSSMLoss(nn.Module):
    DEFAULT_W_POINT = 1.
    DEFAULT_W_COEFFICIENTS = 0.5
    DEFAULT_W_AFFINE = 0.5

    def __init__(self, w_point=DEFAULT_W_POINT, w_coefficients=DEFAULT_W_COEFFICIENTS, w_affine=DEFAULT_W_AFFINE):
        super(DGSSMLoss, self).__init__()
        # point_loss = CorrespondingPointDistance()
        self.point_loss = ChamferLoss()
        self.coefficient_loss = nn.MSELoss()
        self.w_point = w_point
        self.w_coefficients = w_coefficients
        self.w_affine = w_affine

    def forward(self, prediction, target):
        pred_shape, pred_weights, pred_affine = prediction
        targ_shape, targ_weights, targ_affine = target

        # transform target shape into moving space (includes the data augmentation transform)
        targ_rot, targ_trans, targ_scale = targ_affine.split([3, 3, 3], dim=1)
        targ_transform = compose_transform(targ_rot, targ_trans, targ_scale)
        targ_shape_moving_space = targ_transform.transform_points(targ_shape)

        point_loss = self.point_loss(pred_shape, targ_shape_moving_space)
        ssm_param_loss = self.coefficient_loss(pred_weights, targ_weights)
        total_loss = self.w_point * point_loss + self.w_coefficients * ssm_param_loss
        components = {'Point-Loss': point_loss, 'Coefficients': ssm_param_loss}
        if self.w_affine:
            affine_loss = self.coefficient_loss(pred_affine, targ_affine)
            components['Affine-Params'] = affine_loss
            total_loss += self.w_affine * affine_loss

        return total_loss, components


class CorrespondingPointDistance(nn.Module):
    def __init__(self):
        super(CorrespondingPointDistance, self).__init__()

    def forward(self, prediction, target):
        return corresponding_point_distance(prediction, target).pow(2).mean()


def corresponding_point_distance(prediction, target):
    return (prediction - target).pow(2).sum(-1).sqrt()
