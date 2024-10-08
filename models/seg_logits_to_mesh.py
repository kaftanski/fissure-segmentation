import torch
from pytorch3d.structures import Meshes
from torch import nn

from models.access_models import get_point_seg_model_class
from models.divroc import DiVRoC
from models.dpsr_net import DPSR
from models.dpsr_utils import DifferentiableMarchingCubes
from models.modelio import LoadableModel, store_config_args
from utils.pytorch_image_filters import gaussian_differentiation


class DPSRNet2(LoadableModel):
    @store_config_args
    def __init__(self, seg_net_class, k, in_features, num_classes, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 normals_smoothing_sigma=10,
                 dpsr_res=(128, 128, 128), dpsr_sigma=10, dpsr_scale=True, dpsr_shift=True):
        """
        :param seg_net_class: point segmentation network, e.g. DGCNNSeg
        :param dpsr_res: tuple of output field resolution. eg., (128,128,128)
        :param dpsr_sigma: degree of gaussian smoothing
        """
        super(DPSRNet2, self).__init__()
        seg_net_class = get_point_seg_model_class(seg_net_class)
        self.res = dpsr_res
        self.seg_net = seg_net_class(k=k, in_features=in_features, num_classes=num_classes,
                                     spatial_transformer=spatial_transformer, dynamic=dynamic,
                                     image_feat_module=image_feat_module)
        self.seg2mesh = SoftMesh(normals_smoothing_sigma, dpsr_res, dpsr_sigma, dpsr_scale, dpsr_shift,
                                 exclude_background=True)

    def forward(self, x):
        """

        :param x: input of shape (point cloud batch x features x points)
        :return: point segmentation of shape (point cloud batch x num_classes) and
            reconstructed meshes (batch x num_classes)
        """
        seg_logits = self.seg_net(x)
        coords = x[:, :3]  # only take the coords (always the first 3 feature channels)

        # limit points to grid (points might be outside due to augmentation)
        torch.clamp_(coords, min=-1, max=1)

        # compute meshes
        meshes = self.seg2mesh(seg_logits, coords)
        return seg_logits, meshes

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs_min=50):
        seg_logits = self.seg_net.predict_full_pointcloud(pc, sample_points, n_runs_min)
        coords = pc[:, :3]  # only take the coords (always the first 3 feature channels)
        meshes = self.generate_meshes(coords, seg_logits)
        return seg_logits, meshes


class SoftMesh(nn.Module):
    def __init__(self, smoothing_sigma=10, dpsr_res=(128, 128, 128), dpsr_sigma=10, dpsr_scale=True, dpsr_shift=True,
                 exclude_background=True):
        super().__init__()

        self.smoothing_sigma = smoothing_sigma
        self.res = dpsr_res
        self.dpsr = DPSR(dpsr_res, dpsr_sigma, dpsr_scale, dpsr_shift)
        self.psr_grid_to_mesh = DifferentiableMarchingCubes.apply
        self.exclude_background = exclude_background
        self.divroc = DiVRoC.apply  # efficient differentiable voxel rasterization of point clouds

    def forward(self, seg_logits, coords):
        """

        :param seg_logits: (batch, num_classes, num_points)
        :param coords: (batch, 3, num_points)
        :return: (n_classes * batch) meshes
        """
        batch_size, num_classes, num_points = seg_logits.shape

        # softmax logits (since argmax would not be differentiable)
        seg_logits = seg_logits.softmax(1)

        if self.exclude_background:
            # discard the background class
            seg_logits = seg_logits[:, 1:]
            num_classes -= 1

        # # discretize the features to the grid
        # coords = (coords + 1) / 2  # point_rasterize expects range [0, 1]
        # coords = coords.transpose(1, 2)  # (batch, num_points, 3)
        # seg_logits = seg_logits.transpose(1, 2)  # (batch, num_points, num_classes)
        # seg_grid = point_rasterize(coords, seg_logits, self.res)

        # discretize the features to the grid with differentiable splatting
        coords = coords.transpose(1, 2).unsqueeze(-2).unsqueeze(-2)  # (batch, num_points, 1, 1, 3)
        seg_logits = seg_logits.view(*seg_logits.shape, 1, 1)  # (batch, num_classes, num_points, 1, 1)
        seg_grid = self.divroc(seg_logits, coords, (batch_size, num_classes, *self.res)).transpose(-1, -3)

        # extrapolate segmentation values to the entire grid by smoothing (combined with gaussian differentiation)
        # approximate normal field by first order gradients (Jacobian matrix)
        grad_x = gaussian_differentiation(seg_grid, self.smoothing_sigma, order=1, dim=2, padding_mode='constant', truncate=1.5)
        grad_y = gaussian_differentiation(seg_grid, self.smoothing_sigma, order=1, dim=1, padding_mode='constant', truncate=1.5)
        grad_z = gaussian_differentiation(seg_grid, self.smoothing_sigma, order=1, dim=0, padding_mode='constant', truncate=1.5)

        # combine gradients to normal field
        normals = torch.stack([grad_x, grad_y, grad_z], dim=2)  # (batch, n_classes, 3, res, res, res)

        # compute PSR from approximate normals for all objects
        normals = normals.view(-1, *normals.shape[2:])  # move all classes into batch dimension
        coords_repeated = coords.view(batch_size, num_points, 3).repeat_interleave(repeats=num_classes, dim=0)  # repeat points accordingly
        psr_grid = self.dpsr.spectral_PSR(coords_repeated, normals)

        # TODO: use tanh activation?

        # convert PSR to mesh
        verts, faces, normals = self.psr_grid_to_mesh(psr_grid)  # returns (n_classes * batch) meshes
        mesh = Meshes(verts=verts, faces=faces, verts_normals=normals)
        return mesh
