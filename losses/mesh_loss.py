from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing
from pytorch3d.ops import sample_points_from_meshes
from torch import nn


class RegularizedMeshLoss(nn.Module):
    def __init__(self, w_chamfer=1., w_edge_length=1., w_normal_consistency=0.1, w_laplacian=0.1, n_samples=2048):
        """ default weights are from voxel2mesh (or the pytorch3d mesh fitting tutorial at
        https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh)

        :param w_chamfer:
        :param w_edge_length:
        :param w_normal_consistency:
        :param w_laplacian:
        :param n_samples:
        """
        super(RegularizedMeshLoss, self).__init__()
        self.w_chamfer = w_chamfer
        self.w_edge_length = w_edge_length
        self.w_normal_consistency = w_normal_consistency
        self.w_laplacian = w_laplacian
        self.n_samples = n_samples

    def forward(self, mesh_prediction, mesh_target):
        # sample point clouds from meshes and compute their chamfer distance (see Osada et al. "Shape Distributions")
        sample_pred = sample_points_from_meshes(mesh_prediction, num_samples=self.n_samples)
        sample_targ = sample_points_from_meshes(mesh_target, num_samples=self.n_samples)
        loss_chamfer, _ = chamfer_distance(sample_pred, sample_targ)

        # regularize the edge length of the predicted mesh
        #loss_edge = mesh_edge_loss(mesh_prediction, target_length=0.3)  # VERY rough heuristic for this specific setting!
        loss_edge = mesh_edge_loss(mesh_prediction)

        # regularize mesh normal consistency
        loss_normal = mesh_normal_consistency(mesh_prediction)

        # measure laplacian smoothness of mesh
        loss_laplacian = mesh_laplacian_smoothing(mesh_prediction, method="uniform")

        # weighted sum of the losses
        loss = self.w_chamfer * loss_chamfer + \
               self.w_edge_length * loss_edge + \
               self.w_normal_consistency * loss_normal + \
               self.w_laplacian * loss_laplacian

        return loss, {'Chamfer': loss_chamfer, 'Edge Length': loss_edge,
                      'Normal Consistency': loss_normal, 'Laplacian': loss_laplacian}
