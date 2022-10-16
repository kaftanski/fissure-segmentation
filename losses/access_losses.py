from enum import Enum
from typing import List

import torch
from torch import nn

from losses.chamfer_loss import ChamferLoss
from losses.dice_loss import GDL
from losses.mesh_loss import RegularizedMeshLoss
from losses.recall_loss import BatchRecallLoss
from losses.ssm_loss import CorrespondingPointDistance


class Losses(Enum):
    NNUNET = "nnunet"
    """default loss function used in the nnU-Net: CE + Dice loss"""

    CE = "ce"
    """normal cross entropy loss"""

    RECALL = "recall"
    """cross entropy loss weighted with batch-specific false-positive rate, promotes recall"""

    SSM = "ssm"
    """ ssm loss (chamfer distance for now) """  # TODO: implement point to mesh distance

    CHAMFER = "chamfer"
    """ chamfer distance between predicted and target point cloud"""

    MESH = "mesh"
    """ regularized mesh loss """

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


def asseble_dg_ssm_loss():
    point_loss = CorrespondingPointDistance()
    coefficient_loss = nn.MSELoss()

    def combined_loss(prediction, target):
        pred_shape, pred_weights = prediction
        targ_shape, targ_weights = target
        # pl = point_loss(pred_shape, targ_shape)

        # convert predicted shape to pt3d point cloud
        # pl = point_mesh_face_distance(targ_shape, pcl_pred)
        pl = point_loss(pred_shape, targ_shape)
        wl = coefficient_loss(pred_weights, targ_weights)  # TODO: is it better to use regularization w^T * Sigma^-1 * w
        return pl + 0.5 * wl, {'Point-Loss': pl, 'Coefficients': wl}

    return combined_loss


def get_loss_fn(loss: Losses, class_weights: torch.Tensor = None, term_weights: List[float] = None):
    if isinstance(loss, Losses):
        loss = loss.value

    if loss == Losses.NNUNET.value:
        return assemble_nnunet_loss_function(class_weights)

    if loss == Losses.CE.value:
        return nn.CrossEntropyLoss(class_weights)

    if loss == Losses.RECALL.value:
        return BatchRecallLoss()

    if loss == Losses.SSM.value:
        return asseble_dg_ssm_loss()

    if loss == Losses.CHAMFER.value:
        return ChamferLoss()

    if loss == Losses.MESH.value:
        if term_weights is not None:
            assert len(term_weights) == 4
            return RegularizedMeshLoss(
                w_chamfer=term_weights[0],
                w_edge_length=term_weights[1],
                w_normal_consistency=term_weights[2],
                w_laplacian=term_weights[3])
        else:
            # default weights
            return RegularizedMeshLoss()

    raise ValueError(f'No loss function named "{loss}". Please choose one from {Losses.list()} instead.')
