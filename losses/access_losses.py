from enum import Enum
from typing import List

import torch
from torch import nn

from losses.chamfer_loss import ChamferLoss
from losses.dgssm_loss import DGSSMLoss
from losses.dice_loss import GDL
from losses.dpsr_loss import DPSRLoss
from losses.mesh_loss import RegularizedMeshLoss
from losses.nnu_loss import NNULoss
from losses.recall_loss import BatchRecallLoss


class Losses(Enum):
    NNUNET = "nnunet"
    """default loss function used in the nnU-Net: CE + Dice loss"""

    CE = "ce"
    """normal cross entropy loss"""

    RECALL = "recall"
    """cross entropy loss weighted with batch-specific false-positive rate, promotes recall"""

    SSM = "ssm"
    """ ssm loss (chamfer distance for now) """

    CHAMFER = "chamfer"
    """ chamfer distance between predicted and target point cloud"""

    MESH = "mesh"
    """ regularized mesh loss """

    DPSR = "dpsr"
    """ loss for DPSR models (combining nnU-Net (CE+Dice) and Chamfer loss"""

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def get_loss_fn(loss: Losses, class_weights: torch.Tensor = None, term_weights: List[float] = None):
    if isinstance(loss, Losses):
        loss = loss.value

    if loss == Losses.NNUNET.value:
        return NNULoss(class_weights)

    if loss == Losses.CE.value:
        return nn.CrossEntropyLoss(class_weights)

    if loss == Losses.RECALL.value:
        return BatchRecallLoss()

    if loss == Losses.SSM.value:
        if term_weights is not None:
            assert len(term_weights) == 3
            return DGSSMLoss(
                w_point=term_weights[0],
                w_coefficients=term_weights[1],
                w_affine=term_weights[2])
        else:
            # default weights
            return DGSSMLoss()

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

    if loss == Losses.DPSR.value:
        if term_weights is not None:
            assert len(term_weights) == 3
            return DPSRLoss(class_weights,
                            w_seg=term_weights[0],
                            w_chamfer=term_weights[1],
                            epoch_start_chamfer=term_weights[2])

        # else default weights:
        return DPSRLoss(class_weights)

    raise ValueError(f'No loss function named "{loss}". Please choose one from {Losses.list()} instead.')
