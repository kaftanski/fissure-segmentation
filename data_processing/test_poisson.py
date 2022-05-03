import SimpleITK as sitk
import numpy as np
import torch

from data import PointDataset, LungData
from data_processing.surface_fitting import o3d_mesh_to_labelmap, pointcloud_surface_fitting
from utils import kpts_to_world

ds = PointDataset(1024, 'foerstner', exclude_rhf=True)
i = 0
points, _, labels = ds.get_full_pointcloud(i)

img_ds = LungData('../data/')

case, sequence = ds.ids[i]
image = img_ds.get_image(img_ds.get_index(case, sequence))
spacing = torch.tensor(image.GetSpacing()[::-1])
shape = torch.tensor(image.GetSize()[::-1]) * spacing
points = kpts_to_world(points.transpose(0, 1), shape)  # points in millimeters

for depth in range(2, 10):
    meshes = []
    for f in labels.unique()[1:]:
        meshes.append(pointcloud_surface_fitting(points[labels == f], depth=depth))

    labelmap = o3d_mesh_to_labelmap(meshes, image.GetSize()[::-1], image.GetSpacing())
    label_image_predict = sitk.GetImageFromArray(labelmap.numpy().astype(np.uint8))
    label_image_predict.CopyInformation(image)
    sitk.WriteImage(label_image_predict, f"results/{case}_poisson_d{depth}_consistentnormals_{sequence}.nii.gz")
