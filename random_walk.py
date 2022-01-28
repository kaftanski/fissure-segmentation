import os

import SimpleITK as sitk
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils_rw_scribble import sparse_cols, sparse_rows, sparseMultiGrid, overlay_segment, display_colours
from visualization import visualize_with_overlay


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
            val = torch.where((torch.take(im, ii[:, 0]) == torch.take(im, ii[:, 1])), 10., 0.)
            # val = (torch.take(im, ii[:, 0]) == torch.take(im, ii[:, 1])).float()
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


def regularize_fissure_segmentation(image: sitk.Image, fissure_seg: sitk.Image, lung_mask: sitk.Image, lobe_scribbles: sitk.Image) -> sitk.Image:
    # make fissure segmentation binary (disregard the 3 different fissures)
    fissure_seg = sitk.BinaryThreshold(fissure_seg, upperThreshold=0.5, insideValue=0, outsideValue=1)

    # post-process fissure segmentation (make it continuous)
    fissure_seg = sitk.BinaryMorphologicalClosing(fissure_seg, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    # fissure_seg = sitk.Cast(fissure_seg, sitk.sitkFloat32)
    # fissure_seg = sitk.DiscreteGaussian(fissure_seg, variance=10, maximumKernelWidth=50, useImageSpacing=True)
    sitk.WriteImage(fissure_seg, '../fissure_postprocess.nii.gz')

    # convert SimpleITK images to tensors
    img_tensor = torch.from_numpy(sitk.GetArrayFromImage(image)).float()
    graph_mask = torch.from_numpy(sitk.GetArrayFromImage(lung_mask)).float()
    seeds = torch.from_numpy(sitk.GetArrayFromImage(lobe_scribbles).astype(int)).float()
    fissure_tensor = torch.from_numpy(sitk.GetArrayFromImage(fissure_seg).astype(int)).float()

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

    lobe_segmentation = sitk.GetImageFromArray(lobes.numpy())
    lobe_segmentation.CopyInformation(fissure_seg)
    return lobe_segmentation


def simple_regularization(image: sitk.Image, fissure_seg: sitk.Image, lung_mask: sitk.Image, lobe_scribbles: sitk.Image) -> sitk.Image:
    # post-process fissures
    # make fissure segmentation binary (disregard the 3 different fissures)
    fissure_seg_binary = sitk.BinaryThreshold(fissure_seg, upperThreshold=0.5, insideValue=0, outsideValue=1)

    # create inverted lobe mask by combining fissures and not-lung
    not_lobes = sitk.Or(sitk.Not(lung_mask), fissure_seg_binary)

    # close some gaps
    # not_lobes = sitk.BinaryMorphologicalClosing(not_lobes, kernelRadius=(2, 2, 2), kernelType=sitk.sitkBall)
    not_lobes = sitk.BinaryDilate(not_lobes, kernelRadius=(4, 4, 4), kernelType=sitk.sitkBall)

    # find connected components in lobes mask
    lobes_mask = sitk.Not(not_lobes)
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    lobes_components = connected_component_filter.Execute(lobes_mask)
    print(connected_component_filter.GetObjectCount())

    # find the biggest components (= the 5 lobes)
    # shape_stats = sitk.LabelShapeStatisticsImageFilter()
    # shape_stats.Execute(lobes_components)
    # labels = torch.tensor(shape_stats.GetLabels())
    # object_sizes = torch.tensor([shape_stats.GetPhysicalSize(l.item()) for l in labels])
    # values, indices = torch.topk(object_sizes, k=5)

    # sort objects by size
    relabel_filter = sitk.RelabelComponentImageFilter()
    relabel_filter.SetSortByObjectSize(True)
    lobes_components_sorted = relabel_filter.Execute(lobes_components)
    print(f'The 5 largest objects have sizes {relabel_filter.GetSizeOfObjectsInPhysicalUnits()[:5]}')

    # extract the 5 biggest objects (the 5 lobes)
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap({l: 0 for l in range(6, relabel_filter.GetOriginalNumberOfObjects() + 1)})
    lobes_components_top5 = change_label_filter.Execute(lobes_components_sorted)

    return lobes_components_top5


def regularize(case):
    data_path = '/home/kaftan/FissureSegmentation/data'
    sequence = 'fixed'

    print(f'REGULARIZATION of case {case}, {sequence}')

    lobes = simple_regularization(
        sitk.ReadImage(os.path.join(data_path, f'{case}_img_{sequence}.nii.gz')),
        sitk.ReadImage(os.path.join(data_path, f'{case}_fissures_{sequence}.nii.gz')),
        sitk.ReadImage(os.path.join(data_path, f'{case}_mask_{sequence}.nii.gz'), outputPixelType=sitk.sitkUInt8),
        sitk.ReadImage(os.path.join(data_path, f'{case}_lobescribbles_{sequence}.nii.gz'))
    )
    sitk.WriteImage(lobes, os.path.join(data_path, f'{case}_lobes_{sequence}.nii.gz'))


def toy_example():
    data_path = '/home/kaftan/FissureSegmentation'
    img = torch.from_numpy(imageio.imread(os.path.join(data_path, 'toy_example.png'))).float()[..., 0]
    img[img != 0] = 255
    for i in range(len(img)):
        if not i % 10:
            img[i, i] = 255

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


def toy_example_2d():
    # create image with one diagonal line
    img = torch.zeros(128, 128)
    for i in range(128):
        img[i, i] = 1
        img[min(127, i+1), i] = 1
    img[60:68, 60:68] = 0
    # img[63, 63] = 0

    # seed points above and below plane
    sp = np.array([[0, 63], [80, 63]])
    seeds = torch.zeros_like(img, dtype=torch.long)
    seeds[sp[0, 0], sp[0, 1]] = 1
    seeds[sp[1, 0], sp[1, 1]] = 2

    # # mask of foreground pixels
    # mask = torch.zeros_like(img, dtype=torch.bool)
    # mask[10:110, 10:110, 10:110] = 1

    # random walk
    L = compute_laplace_matrix(img.contiguous(), 'binary')
    prob = random_walk(L, seeds.contiguous())
    seg = prob.argmax(-1) + 1
    # seg = torch.where(mask, seg, 0)

    # visualize
    fig = plt.figure(dpi=300)
    ax = fig.gca()
    ax.scatter(sp[:, 1], sp[:, 0])
    visualize_with_overlay(img.numpy(), seg.numpy(), title=f'Seeds: {sp[0].tolist()}, {sp[1].tolist()}', ax=ax)
    plt.show()


def toy_example_3d():
    # create image with one diagonal plane (incomplete)
    img = torch.zeros(128, 128, 128)
    for i in range(128):
        if not i % 16:
            img[i, :, i] = 0
        else:
            img[i, :, i] = 1

    # seed points above and below plane
    seeds = torch.zeros_like(img, dtype=torch.long)
    seeds[50:60, 70:80, 30:40] = 1
    seeds[50:60, 70:80, 100:110] = 2

    # mask of foreground pixels
    mask = torch.zeros_like(img, dtype=torch.bool)
    mask[10:110, 10:110, 10:110] = 1

    # random walk
    L = compute_laplace_matrix(img.contiguous(), 'binary')
    prob = random_walk(L, seeds.contiguous(), mask)
    seg = prob.argmax(-1) + 1
    seg = torch.where(mask, seg, 0)

    # output
    sitk.WriteImage(sitk.GetImageFromArray(img.numpy()), '../3D_RandomWalk_toy_example_img_incomplete.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(seg.numpy()), '../3D_RandomWalk_toy_example_seg_incomplete.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(seeds.numpy()), '../3D_RandomWalk_toy_example_seed.nii.gz')

    ### DEBUG NOTES ###
    # plane along 2 dimensions works!
    # hole works too
    # very incomplete boundaries are insufficient with binary weights


if __name__ == '__main__':
    # toy_example()
    regularize('EMPIRE01')
    # toy_example_3d()
    # toy_example_2d()
