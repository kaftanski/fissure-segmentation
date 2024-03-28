import warnings
from abc import ABC, abstractmethod

import torch
from torch import nn

from models.modelio import LoadableModel, store_config_args


class PointSegmentationModelBase(LoadableModel, ABC):
    @store_config_args
    def __init__(self, in_features, num_classes, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        pass

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs_min=50):
        output_activation = nn.Softmax(dim=1)

        n_leftover_runs = n_runs_min // 5
        n_initial_runs = n_runs_min - n_leftover_runs
        softmax_accumulation = torch.zeros(pc.shape[0], self.num_classes, *pc.shape[2:], device=pc.device)
        for r in range(n_initial_runs):
            perm = torch.randperm(pc.shape[-1], device=pc.device)[:sample_points]
            softmax_accumulation[..., perm] += output_activation(self(pc[..., perm]))

        # look if there are points that have been left out
        left_out_pts = torch.nonzero(softmax_accumulation.sum(1) == 0)[..., 1]
        print(f'After {n_initial_runs} runs, {left_out_pts.shape[0]} points have not been seen yet.')
        if left_out_pts.shape[0] > 0:
            other_pts = torch.nonzero(softmax_accumulation.sum(1))[..., 1]
            point_mix = sample_points // 2
            fill_out_num = sample_points - point_mix
            perm = torch.randperm(n_leftover_runs * point_mix, device=pc.device) % len(left_out_pts)
            for r in range(n_leftover_runs):
                lo_pts = left_out_pts[perm[r * point_mix:(r + 1) * point_mix]]
                other = torch.randperm(len(other_pts), device=pc.device)[:fill_out_num]
                pts = torch.cat((lo_pts, other), dim=0)
                softmax_accumulation[..., pts] += output_activation(self(pc[..., pts]))

            if (softmax_accumulation.sum(1) == 0).sum() != 0:
                warnings.warn('NOT ALL POINTS HAVE BEEN SEEN')

        return output_activation(softmax_accumulation)
