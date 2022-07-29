import matplotlib.pyplot as plt
import torch
from math import sqrt
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import random_rotations
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.loss import point_mesh_face_distance
from torch import optim

from data import CorrespondingPointDataset
from losses.access_losses import get_loss_fn, Losses
from losses.ssm_loss import corresponding_point_distance
from models.dg_ssm import DGSSM
from visualization import visualize_point_cloud


def random_transformation(n_samples, device):
    rotations = random_rotations(n_samples, device=device)

    # random translation
    translation_amount = 0.1  # pytorch grid coordinates (positive or negative)
    translations = (torch.rand(n_samples, 3, device=device) * 2 - 1) * translation_amount

    return Transform3d(device=device).rotate(rotations)#.translate(translations)


def get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target):
    # get random transformation parameters for batch
    transform = random_transformation(batch_size, device)

    # transform the input point cloud and the pytorch3d target mesh
    inputs_transformed = transform.transform_points(input_pc.T.unsqueeze(0)).transpose(1, 2)
    mesh_transformed = Meshes(verts=transform.transform_points(mesh_target.verts_packed()),
                              faces=mesh_target.faces_packed().unsqueeze(0).repeat(batch_size, 1, 1),
                              verts_normals=transform.transform_normals(
                                  mesh_target.verts_normals_packed()))  # TODO: tranlation is in grid coordinates, while meshes are in world coords.
    shape_transformed = transform.transform_points(ssm_target)
    return inputs_transformed, mesh_transformed, shape_transformed


def mean_pairwise_shape_dist(shapes, meshes):
    corr_dists = []
    p2m_dists = []
    for i in range(shapes.shape[0]):
        for j in range(shapes.shape[0]):
            if i == j:
                continue

            # corr_dists.append(corresponding_point_distance(shapes[i], shapes[j]).mean())
            p2m_dists.append(point_mesh_face_distance(meshes[i], Pointclouds(shapes[j:j+1])).sqrt())

    corr_dists = torch.tensor(corr_dists)
    p2m_dists = torch.tensor(p2m_dists)
    return p2m_dists.mean()  # , corr_dists.mean()


device = 'cuda:3'

ds = CorrespondingPointDataset(1024, 'cnn')
shapes = ds.corr_points.get_shape_datamatrix().to(device)

# augment shapes with random transformations
n_augment = 10  # times the data set
n_shapes = shapes.shape[0]
for i in range(n_augment):
    augmentations = random_transformation(n_shapes, device)
    shapes = torch.concat([shapes, augmentations.transform_points(shapes[:n_shapes])], dim=0)

dgssm = DGSSM(20, 3, spatial_transformer=True).to(device)
dgssm.fit_ssm(shapes.to(device))

criterion = get_loss_fn(Losses.SSM)
optimizer = optim.Adam(dgssm.parameters(), lr=0.001)

epochs = 10
steps_per_epoch = 10
batch_size = 32

fixed_index = 0
_, (ssm_target, mesh_target) = ds[fixed_index]
input_pc, input_labels = super(CorrespondingPointDataset, ds).__getitem__(0)
input_pc = input_pc.to(device)
ssm_target = ssm_target.to(device)
mesh_target = mesh_target.to(device)

ssm_valid_error = torch.zeros(epochs, device=device)
dgssm_valid_error = torch.zeros_like(ssm_valid_error, device=device)
loss_progression = torch.zeros_like(ssm_valid_error, device=device)
components_progression = {'Point-Loss': torch.zeros_like(ssm_valid_error, device=device),
                          'Coefficients': torch.zeros_like(ssm_valid_error, device=device)}
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        dgssm.train()

        # get randomly transformed input and target data
        inputs_transformed, mesh_transformed, shape_transformed = \
            get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target)

        # train the model
        output = dgssm(inputs_transformed)

        target_weights = dgssm.ssm(shape_transformed)

        optimizer.zero_grad()
        loss, components = criterion(output, (mesh_transformed, target_weights))
        loss.backward()
        optimizer.step()

        loss_progression[epoch] += loss.item() / steps_per_epoch
        components_progression['Point-Loss'][epoch] += components['Point-Loss'].item() / steps_per_epoch
        components_progression['Coefficients'][epoch] += components['Coefficients'].item() / steps_per_epoch

        # validation
        dgssm.eval()

        inputs_transformed, mesh_transformed, shape_transformed = \
            get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target)

        with torch.no_grad():
            # validate DGSSM
            outputs = dgssm(inputs_transformed)
            target_weights = dgssm.ssm(shape_transformed)
            _, valid_components = criterion(output, (mesh_transformed, target_weights))

            dgssm_valid_error[epoch] += torch.sqrt(valid_components['Point-Loss']) / steps_per_epoch

            # SSM baseline error of validation data
            ssm_reconstruction = dgssm.ssm.decode(target_weights)
            ssm_valid_error[epoch] += corresponding_point_distance(shape_transformed, ssm_reconstruction).mean() / steps_per_epoch

        # VISUALIZATION
        # visualize_point_cloud(input_pc.T, input_labels)
        # for i, inp in enumerate(inputs_transformed):
        #     visualize_point_cloud(inp.T, input_labels, title='input points')
        #     visualize_point_cloud(ssm_reconstruction[i], torch.ones(ssm_reconstruction.shape[1]), title='ssm reconstruction')
        #     visualize_point_cloud(mesh_transformed.verts_list()[i], torch.ones(mesh_transformed.verts_list()[i].shape[0]), title='mesh target')

    print(f'TRAINING epoch {epoch}')
    print(f'\tTotal: {loss_progression[epoch].item():.4f} | Point-Loss: {sqrt(components_progression["Point-Loss"][epoch]):.4f} mm | Coefficients: {components_progression["Coefficients"][epoch]:.4f}')

    print('VALIDATION')
    print(f'\tDGSSM: {dgssm_valid_error[epoch].item():.4f} mm | SSM baseline: {ssm_valid_error[epoch].item():.4f} mm\n')

# loss plots
plt.figure()
plt.plot(loss_progression.cpu())
plt.title('total loss')
for key, val in components_progression.items():
    plt.figure()
    plt.plot(val.cpu())
    plt.title(key)
plt.show()

# compute statistics of the random data for reference
inputs_transformed, mesh_transformed, shape_transformed = \
    get_random_transformed_data(50, device, input_pc, mesh_target, ssm_target)

mean_valid_data_distance = mean_pairwise_shape_dist(shape_transformed, mesh_transformed)
print(f'MEAN DATASET DISTANCES: {mean_valid_data_distance.item():.4f} mm\n')
