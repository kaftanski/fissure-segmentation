import torch


def pairwise_dist(x):
    """ distance from each point in x to itself and others

    :param x: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    xTx = torch.bmm(x, x.transpose(2, 1))
    dist = xx - 2.0 * xTx + xx.transpose(2, 1)
    dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0  # ensure diagonal is 0
    return dist


def pairwise_dist2(x, y):
    """ distance from each point in x to its corresponding point in y

    :param x: point cloud batch of shape (B x N x 3)
    :param y: point cloud batch of shape (B x N x 3)
    :return: distance matrix of shape (B x N x N)
    """
    xx = (x ** 2).sum(2, keepdim=True)
    yy = (y ** 2).sum(2, keepdim=True)
    xTy = torch.bmm(x, y.transpose(2, 1))
    dist = xx - 2.0 * xTy + yy.transpose(2, 1)
    return dist
