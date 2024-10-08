import torch
from torch import nn

from losses.mesh_loss import RegularizedMeshLoss
from losses.nnu_loss import NNULoss


class DPSRLoss(nn.Module):
    """ loss function combines point segmentation loss (nnU-Net) with
    Chamfer distance (like in SAP optimization-based) """
    DEFAULT_W_SEG = 0.5
    DEFAULT_W_CHAMFER = 0.5
    DEFAULT_EPOCH_START_CHAMFER = 0.1

    def __init__(self, class_weights, w_seg=DEFAULT_W_SEG, w_mesh=DEFAULT_W_CHAMFER, epoch_start_mesh_loss=DEFAULT_EPOCH_START_CHAMFER):
        super().__init__()
        self.w_seg = w_seg
        self.w_mesh = w_mesh
        self.epoch_start_mesh = epoch_start_mesh_loss

        self.seg_loss = NNULoss(class_weights)
        self.chamfer_loss = RegularizedMeshLoss(w_chamfer=1,
                                                w_laplacian=0,
                                                w_edge_length=0,
                                                w_normal_consistency=0)  # todo: regularize if necessary

    def forward(self, prediction, target, current_epoch_fraction=None):
        pred_seg, pred_meshes = prediction
        targ_seg, targ_meshes = target

        seg_loss, _ = self.seg_loss(pred_seg, targ_seg)

        if (current_epoch_fraction >= self.epoch_start_mesh
                and len(pred_meshes) == len(targ_meshes)  # Todo: find way to incentivise this more
                and self.w_mesh > 0):
            cham_loss, _ = self.chamfer_loss(pred_meshes, targ_meshes)
            loss = self.w_seg * seg_loss + self.w_mesh * cham_loss
        else:
            # only use seg_loss first
            cham_loss = torch.tensor(0)
            loss = seg_loss

        return loss, {'Segmentation': seg_loss, 'Chamfer': cham_loss}
