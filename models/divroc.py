"""
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)

Copyright: Mattias P. Heinrich
https://github.com/mattiaspaul/ChasingClouds/divroc.py
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
# import torchinfo
# import matplotlib.pyplot as plt

from torch.autograd import Function
from torch.autograd.functional import jacobian

from constants import ALIGN_CORNERS


class DiVRoC(Function):
    """
    modified forward splatting function to spread feature values at the specified coordinates
    """
    @staticmethod
    def forward(ctx, feature_values, coords, shape):
        """

        :param ctx: context for backward pass
        :param feature_values: values to be rasterized (bs, feat, n_pts, 1, 1)
        :param coords: coordinates to rasterize to (bs, n_pts, 1, 1, 3)
        :param shape: grid shape (bs, feat, grid_sz, grid_sz, grid_sz)
        :return:
        """
        device = feature_values.device
        dtype = feature_values.dtype

        output = -jacobian(func=lambda x: (F.grid_sample(x, coords, align_corners=ALIGN_CORNERS) - feature_values).pow(2).mul(0.5).sum(),
                           # compute derivative wrt inputs (x), it does not matter what values are used here:
                           inputs=torch.zeros(shape, dtype=dtype, device=device))

        ctx.save_for_backward(feature_values, coords, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, output = ctx.saved_tensors

        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]

        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B * C, 1, *output_dims), x,
                                             align_corners=ALIGN_CORNERS).mean(),
                     grid.unsqueeze(1).repeat(1, C, *([1] * (len(input_dims) + 1))).view(
                         B * C, *input_dims, len(input_dims))).view(B, C, *input_dims, len(input_dims))

        grad_grid = (input.numel() * input.unsqueeze(-1) * y).sum(1)

        grad_input = F.grid_sample(grad_output, grid, align_corners=ALIGN_CORNERS)

        return grad_input, grad_grid, None


if __name__ == '__main__':
    from models.dpsr_utils import point_rasterize
    ALIGN_CORNERS = True  # to compare to SAP implementation, which implicitly uses align_corners=True
    bs = 1
    feat = 4
    n_pts = 2048
    grid_sz = 128
    shape = (bs, feat, grid_sz, grid_sz, grid_sz)

    coords = torch.rand(1, n_pts, 3, device='cuda')
    coords_pt = coords * 2 - 1  # pytorch grid coords in range [-1, 1]
    # if not ALIGN_CORNERS:
    #     scale2 = (torch.tensor([grid_sz, grid_sz, grid_sz], device='cuda') - 1) / torch.tensor([grid_sz, grid_sz, grid_sz], device='cuda')
    #     coords_pt *= scale2
    values = torch.randn(1, n_pts, feat, device='cuda')

    # apply divroc and SAP rasterization
    output = DiVRoC.apply(values.transpose(1, 2).unsqueeze(-1).unsqueeze(-1),  # (bs, feat, n_pts, 1, 1)
                          coords_pt.unsqueeze(-2).unsqueeze(-2),  # (bs, n_pts, 1, 1, 3)
                          shape)
    output_sap = point_rasterize(coords, values, (grid_sz, grid_sz, grid_sz)).transpose(-1, -3)
    output_sap = output_sap[:, :, :, ::-1]

    # visualize
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(output[0, 0, 50].cpu().numpy() != 0)
    ax[1].imshow(output_sap[0, 0, 50].cpu().numpy() !=0)
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(output[0, 0, :, 50].cpu().numpy() != 0)
    ax[1].imshow(output_sap[0, 0, :, 50].cpu().numpy() != 0)
    plt.show()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(output[0, 0, :, :, 50].cpu().numpy() != 0)
    ax[1].imshow(output_sap[0, 0, :, :, 50].cpu().numpy() != 0)
    plt.show()
