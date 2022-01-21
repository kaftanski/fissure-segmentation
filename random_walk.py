import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from matplotlib import pyplot as plt

from utils_rw_scribble import sparse_cols, sparse_rows, sparseMultiGrid, overlay_segment, display_colours


def compute_laplace_matrix(im: torch.Tensor, edge_weights: str, graph_mask: torch.Tensor = None) -> torch.sparse.Tensor:
    """ Computes Laplacian matrix for an n-dimensional image with intensity weights.

    :param im: input image
    :param edge_weights: either 'binary' for binary probabilities, 'intensity' for intensity difference weights
    :param graph_mask: tensor mask where each zero pixel should be excluded from graph (e.g.: lung-mask)
    :return: Laplacian matrix
    """
    # parameters and image dimensions
    sigma = 8
    lambda_ = 1
    n_nodes = im.numel()

    # create 1D index vector
    ind = torch.arange(n_nodes).view(*im.size())

    # define the graph, one dimension at a time
    A_per_dim = []
    for dim in range(len(im.shape)):
        slices = [slice(None)] * dim

        # define one step forward (neighbors) in the current dimension via the continuous index
        i_from = ind[slices + [slice(None, -1)]].reshape(-1, 1)
        i_to = ind[slices + [slice(1, None)]].reshape(-1, 1)
        ii = torch.cat((i_from, i_to), dim=1)

        if graph_mask is not None:
            # remove edges containing pixels not in the graph-mask
            graph_indices = ind[graph_mask != 0].view(-1)
            # ii = torch.take_along_dim(ii, indices=graph_indices, dim=0)
            ii = torch.stack([edge for edge in ii if edge[0] in graph_indices and edge[0] in graph_indices], dim=0)  # TODO: make this more performant

        if edge_weights == 'intensity':
            # compute exponential edge weights from image intensities
            val = torch.exp(-(torch.take(im, ii[:, 0]) - torch.take(im, ii[:, 1])).pow(2) / (2 * sigma ** 2))
        elif edge_weights == 'binary':
            # 1 if values are the same, 0 if not
            val = (torch.take(im, ii[:, 0]) == torch.take(im, ii[:, 1])).float()
        else:
            raise ValueError(f'No edge weights named "{edge_weights}" known.')

        # create first part of neighbourhood matrix (similar to setFromTriplets in Eigen)
        A = torch.sparse.FloatTensor(ii.t(), val, torch.Size([n_nodes, n_nodes]))

        # make graph symmetric (add backward edges)
        A = A + A.t()
        A_per_dim.append(A)

    # combine all dimensions into one graph
    A = A_per_dim[0]
    for a in A_per_dim[1:]:
        A += a

    # compute degree matrix (diagonal sum)
    D = torch.sparse.sum(A, 0).to_dense()

    # put D and A together
    L = torch.sparse.FloatTensor(torch.cat((ind.view(1, -1), ind.view(1, -1)), 0), .00001 + lambda_ * D,
                                 torch.Size([n_nodes, n_nodes]))
    L += (A * (-lambda_))

    return L


def random_walk(L: torch.sparse.Tensor, labels: torch.Tensor, graph_mask: torch.Tensor = None) -> torch.Tensor:
    # linear index tensor
    ind = torch.arange(labels.numel())

    # extract seeded (x_s) and unseeded indices (x_u) and limit to within the mask, if provided
    seeded = labels.view(-1) != 0
    if graph_mask is None:
        graph_mask = torch.tensor(True)
    x_s = ind[torch.logical_and(seeded, graph_mask.view(-1))]
    x_u = ind[torch.logical_and(torch.logical_not(seeded), graph_mask.view(-1))]

    # get blocks from L: L_u (edges between unseeded nodes) and B^T (edges between an unseeded and an seeded node)
    L_u = sparse_cols(sparse_rows(L, x_u), x_u)
    B_T = sparse_rows(sparse_cols(L, x_s), x_u)

    # create seeded probabilities u_s
    u_s = F.one_hot(labels.view(-1)[seeded] - 1).float()

    # solve sparse LSE
    u_u = sparseMultiGrid(L_u, -1 * torch.sparse.mm(B_T, u_s), 1)

    probabilities = torch.zeros(labels.numel(), u_u.shape[-1])
    probabilities[x_s] = u_s
    probabilities[x_u] = u_u
    return probabilities.view(*labels.shape, -1)


def regularize_fissure_segmentation(image: nib.Nifti1Image, fissure_seg: nib.Nifti1Image, lung_mask: nib.Nifti1Image, lobe_scribbles: nib.Nifti1Image) -> nib.Nifti1Image:
    # convert SimpleITK images to tensors
    img_tensor = torch.from_numpy(image.get_fdata()).float()
    graph_mask = torch.from_numpy(lung_mask.get_fdata()).float().contiguous()
    seeds = torch.from_numpy(lobe_scribbles.get_fdata()).float().contiguous()
    fissure_tensor = torch.from_numpy(fissure_seg.get_fdata()).float().contiguous()

    # downscale images for faster computation
    target_size = 128
    downsample_factor = target_size / max(img_tensor.shape)
    img_downsampled = F.interpolate(img_tensor.unsqueeze(0).unsqueeze(0), scale_factor=downsample_factor, mode='trilinear').squeeze()
    graph_mask_downsampled = F.interpolate(graph_mask.unsqueeze(0).unsqueeze(0), scale_factor=downsample_factor, mode='nearest').squeeze().bool()
    seeds_downsampled = F.interpolate(seeds.unsqueeze(0).unsqueeze(0), scale_factor=downsample_factor, mode='nearest').squeeze().long()
    fissures_downsampled = F.interpolate(fissure_tensor.unsqueeze(0).unsqueeze(0), scale_factor=downsample_factor, mode='nearest').squeeze().long()

    # compute graph laplacian
    print('Computing graph Laplacian matrix')
    L = compute_laplace_matrix(fissures_downsampled, 'binary')

    # random walk lobe segmentation
    print('Performing random walk lobe segmentation')
    probabilities = random_walk(L, seeds_downsampled, graph_mask_downsampled)

    # undo downscaling
    probabilities = F.interpolate(probabilities.movedim(-1, 0).unsqueeze(0), size=img_tensor.shape, mode='trilinear').squeeze()

    # final lobe segmentation
    lobes = torch.where(condition=graph_mask.bool(), input=probabilities.argmax(0) + 1, other=0)  # background is set to zero
    # assert torch.all(seeds[seeds != 0] == lobes[seeds != 0]), 'Seed scribbles have changed label'

    lobe_segmentation = nib.Nifti1Image(lobes.numpy(), image.affine, header=image.header)
    return lobe_segmentation


def regularize(case):
    data_path = '/home/kaftan/FissureSegmentation/data'
    sequence = 'fixed'
    lobes = regularize_fissure_segmentation(
        nib.load(os.path.join(data_path, f'{case}_img_{sequence}.nii.gz')),
        nib.load(os.path.join(data_path, f'{case}_fissures_{sequence}.nii.gz')),
        nib.load(os.path.join(data_path, f'{case}_mask_{sequence}.nii.gz')),
        nib.load(os.path.join(data_path, f'{case}_lobescribbles_{sequence}.nii.gz'))
    )
    nib.save(lobes, os.path.join(data_path, f'{case}_lobes_{sequence}.nii.gz'))


def toy_example():
    data_path = '/home/kaftan/FissureSegmentation'
    img = torch.from_numpy(imageio.imread(os.path.join(data_path, 'toy_example.png'))).float()[..., 0]
    img[img != 0] = 255
    lungmask = torch.from_numpy(imageio.imread(os.path.join(data_path, 'toy_example_lungmask.png'))).float()[..., 0]
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(inverted_mask.float())
    # plt.show()

    L = compute_laplace_matrix(img, 'binary', graph_mask=lungmask)

    seeds = torch.zeros_like(img).long()
    seeds[50, 100] = 1
    seeds[110, 75] = 2
    seeds[75, 200] = 3
    seeds[205, 200] = 4

    prob = random_walk(L, seeds, lungmask)

    rgb = torch.tensor([[235, 175,  86],
                        [111, 163, 91],
                        [225, 140, 154],
                        [78, 129, 170],
                        [45, 170, 170]])
    display_colours(rgb)
    segmentation = prob.argmax(2) + 1
    segmentation[lungmask == 0] = 0
    overlay = overlay_segment(img, segmentation, rgb)
    plt.imshow(overlay)
    plt.show()


if __name__ == '__main__':
    # toy_example()
    regularize('EMPIRE02')
