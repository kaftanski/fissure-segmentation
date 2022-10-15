import itertools
import math

import numpy as np
import torch


def get_plane_mesh(n=2025, xrange=(-1, 1), yrange=(-1, 1), device='cpu'):
    steps = int(math.sqrt(n))
    x = torch.linspace(xrange[0], xrange[1], steps=steps, device=device)
    y = torch.linspace(yrange[0], yrange[1], steps=steps, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # create faces
    faces = []
    for j in range(steps - 1):
        for i in range(steps - 1):
            cur = j * steps + i
            faces.append([cur, cur + 1, cur + steps])
            faces.append([cur + 1, cur + steps, cur + 1 + steps])

    return points, torch.tensor(faces, device=device)


# from https://github.com/antao97/UnsupervisedPointCloudReconstruction/
def get_sphere():
    return np.load("shapes/sphere.npy")


def get_gaussian():
    return np.load("shapes/gaussian.npy")


def get_plane():
    meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    plane = np.array(list(itertools.product(x, y)))
    return plane
