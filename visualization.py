import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def visualize_with_overlay(image: np.ndarray, segmentation: np.ndarray, title: str = None, onehot_encoding: bool = False, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    if segmentation.ndim == 2:
        if onehot_encoding:
            segmentation = segmentation.reshape([*segmentation.shape, 1])
        else:
            # encoding is a label map with numbers representing the objects
            labels = np.unique(segmentation)

            # excluding background with value 0
            labels = labels[labels != 0]

            segmentation_onehot = np.zeros((*image.shape, len(labels)))
            for i, l in enumerate(labels):
                segmentation_onehot[segmentation == l, i] = 1

            segmentation = segmentation_onehot

    ax.imshow(image)
    colors = ['g', 'y', 'c', 'r']

    for i in range(segmentation.shape[-1]):
        ax.imshow(np.ma.masked_where(segmentation[:, :, i] == 0, np.full([*segmentation.shape[:2]], fill_value=255)),
                   cmap=ListedColormap([colors[i % len(colors)]]), alpha=0.5, interpolation=None)

    if title:
        ax.set_title(title)
