import torch
from torch import optim, nn

from data import CorrespondingPoints
from losses.dgssm_loss import CorrespondingPointDistance, corresponding_point_distance
from shape_model.ssm import LSSM, vector2shape, shape2vector


def sanity_check():
    """
    Can Adam optimize a weight vector that is equal to the SSM encoding?
    """
    n_iter = 1000
    lr = 1
    device = 'cuda:3'

    ds = CorrespondingPoints()

    shapes = ds.get_shape_datamatrix_with_affine_reg().to(device)

    ssm = LSSM()
    ssm.fit(shapes)
    ssm.to(device)

    weights = nn.Parameter(torch.zeros(1, ssm.num_modes, device=device))
    optimizer = optim.Adam(params=[weights], lr=lr)

    criterion = CorrespondingPointDistance()
    for s in range(len(shapes)):
        with torch.no_grad():
            weights.zero_()
        target_shape = shapes[s].to(device)
        for it in range(n_iter):
            optimizer.zero_grad()
            reconstruction = ssm.decode(weights)
            loss = criterion(reconstruction, target_shape)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(loss.item())

        with torch.no_grad():
            optimal_reconstruction = ssm.decode(ssm(target_shape))
            optimal_reconstruction_error = corresponding_point_distance(target_shape, optimal_reconstruction)
            reconstruction_difference = corresponding_point_distance(reconstruction, optimal_reconstruction)
        print(f'Error: {reconstruction_difference.mean().item():.4f} | Baseline: {optimal_reconstruction_error.mean().item():.4f}')


def sanity_check2():
    """
    Can Adam reconstruct the eigenvector matrix?
    """
    def encode(shapes, mean_shape, matrix):
        shapes = shape2vector(shapes)
        projection = torch.matmul(matrix.transpose(-1, -2), (shapes - mean_shape).unsqueeze(-1))
        return projection.squeeze(-1)

    def decode(weights, mean_shape, matrix):
        weights = weights.view(*weights.shape[:2], 1)
        reconstruction = mean_shape + torch.matmul(matrix, weights).squeeze(-1)
        return vector2shape(reconstruction, 3)

    n_iter = 10000
    lr = .01
    device = 'cuda:1'

    ds = CorrespondingPoints()

    shapes = ds.get_shape_datamatrix_with_affine_reg().to(device)

    ssm = LSSM()
    ssm.fit(shapes)
    ssm.to(device)

    optimal_reconstruction = ssm.decode(ssm(shapes))
    optimal_reconstruction_error = corresponding_point_distance(shapes, optimal_reconstruction)

    weight = nn.Parameter(torch.randn_like(ssm.eigenvectors, device=device)*0.1)
    mean_shape = shape2vector(shapes.mean(0, keepdim=True))
    optimizer = optim.Adam(params=[weight], lr=lr)

    criterion = CorrespondingPointDistance()
    for it in range(n_iter):
        optimizer.zero_grad()
        reconstruction = decode(encode(shapes, mean_shape, weight), mean_shape, weight)
        loss = criterion(reconstruction, shapes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(loss.item())

        with torch.no_grad():
            reconstruction_difference = corresponding_point_distance(reconstruction, optimal_reconstruction)
        print(f'Error: {reconstruction_difference.mean().item():.4f} | Baseline: {optimal_reconstruction_error.mean().item():.4f}')

    print(weight)
    print(weight.mean(), weight.min(), weight.max())
    print(ssm.eigenvectors.mean(), ssm.eigenvectors.min(), ssm.eigenvectors.max())


def sanity_check3():
    """
    Can Adam reconstruct the coefficients for all shapes at once?
    """
    n_iter = 10000
    lr = .01
    device = 'cuda:1'

    ds = CorrespondingPoints()

    shapes = ds.get_shape_datamatrix_with_affine_reg().to(device)

    ssm = LSSM()
    ssm.fit(shapes)
    ssm.to(device)

    optimal_reconstruction = ssm.decode(ssm(shapes))
    optimal_reconstruction_error = corresponding_point_distance(shapes, optimal_reconstruction)

    weight = nn.Parameter(torch.zeros(len(shapes), ssm.num_modes, device=device))
    optimizer = optim.Adam(params=[weight], lr=lr)

    criterion = CorrespondingPointDistance()
    for it in range(n_iter):
        optimizer.zero_grad()
        reconstruction = ssm.decode(weight)
        loss = criterion(reconstruction, shapes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(loss.item())

        with torch.no_grad():
            reconstruction_difference = corresponding_point_distance(reconstruction, optimal_reconstruction)
        print(f'Error: {reconstruction_difference.mean().item():.4f} | Baseline: {optimal_reconstruction_error.mean().item():.4f}')

    print(weight)
    print(weight.mean(), weight.min(), weight.max())
    print(ssm.eigenvectors.mean(), ssm.eigenvectors.min(), ssm.eigenvectors.max())


def ransac():
    # TODO
    pass


if __name__ == '__main__':
    sanity_check3()
