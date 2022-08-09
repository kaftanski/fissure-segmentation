import csv
import json
import os

import matplotlib.pyplot as plt
import torch
from math import sqrt
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import random_rotations
from pytorch3d.transforms.transform3d import Transform3d
from pytorch3d.loss import point_mesh_face_distance
from torch import optim, nn

from data import CorrespondingPointDataset
from losses.access_losses import get_loss_fn, Losses
from losses.ssm_loss import corresponding_point_distance, CorrespondingPointDistance
from models.dg_ssm import DGSSM
from shape_model.qualitative_evaluation import mode_plot
from visualization import visualize_point_cloud


def random_transformation(n_samples, device):
    rotations = random_rotations(n_samples, device=device)

    # random translation
    translation_amount = 0.1  # pytorch grid coordinates (positive or negative)
    translations = (torch.rand(n_samples, 3, device=device) * 2 - 1) * translation_amount

    return Transform3d(device=device).rotate(rotations)#.translate(translations)


def rotate_around_center(shapes, rotations: Transform3d):
    assert shapes.ndim == 3
    translation_to_center = shapes.mean(1, keepdim=True)
    return rotations.transform_points(shapes - translation_to_center) + translation_to_center


def get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target):
    # get random transformation parameters for batch
    transform = random_transformation(batch_size, device)

    # transform the input point cloud and the pytorch3d target mesh
    inputs_transformed = rotate_around_center(input_pc.T.unsqueeze(0), transform).transpose(1, 2)
    mesh_transformed = Meshes(verts=rotate_around_center(mesh_target.verts_packed().unsqueeze(0), transform),
                              faces=mesh_target.faces_packed().unsqueeze(0).repeat(batch_size, 1, 1),
                              verts_normals=transform.transform_normals(
                                  mesh_target.verts_normals_packed()))  # TODO: translation is in grid coordinates, while meshes are in world coords.
    shape_transformed = rotate_around_center(ssm_target.unsqueeze(0), transform)
    return inputs_transformed, mesh_transformed, shape_transformed


def mean_pairwise_shape_dist(shapes, meshes):
    corr_dists = []
    # p2m_dists = []
    for i in range(shapes.shape[0]):
        for j in range(shapes.shape[0]):
            if i == j:
                continue

            corr_dists.append(corresponding_point_distance(shapes[i], shapes[j]).mean())
            # p2m_dists.append(point_mesh_face_distance(meshes[i], Pointclouds(shapes[j:j+1])).sqrt())

    corr_dists = torch.tensor(corr_dists)
    # p2m_dists = torch.tensor(p2m_dists)
    return corr_dists.mean() # p2m_dists.mean()


show = True

out_dir = 'results/dgssm_toy_example_only_corr_point_loss'
os.makedirs(out_dir, exist_ok=True)
plot_dir = os.path.join(out_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

device = 'cuda:3'

ds = CorrespondingPointDataset(1024, 'cnn')
shapes = ds.corr_points.get_shape_datamatrix().to(device)

# # augment shapes with random transformations
# n_augment = 10  # times the data set
# n_shapes = shapes.shape[0]
# for i in range(n_augment):
#     augmentations = random_transformation(n_shapes, device)
#     shapes = torch.concat([shapes, rotate_around_center(shapes[:n_shapes], augmentations)], dim=0)

# only use one rotated shape for training
shapes = shapes[0:1]
augmentations = random_transformation(200, device)
shapes = torch.concat([shapes, rotate_around_center(shapes, augmentations)], dim=0)

dgssm = DGSSM(20, 3, spatial_transformer=True).to(device)
dgssm.fit_ssm(shapes.to(device))
mode_plot(dgssm.ssm, savepath=os.path.join(plot_dir, 'ssm_modes.png'), show=show)

# for i in range(5):
#     visualize_point_cloud(shapes[i], torch.ones(shapes.shape[1]), title='train shape')
# visualize_point_cloud(dgssm.ssm.mean_shape.data.squeeze().view(2048, 3), torch.ones(shapes.shape[1]), title='mean shape')

criterion = CorrespondingPointDistance()# get_loss_fn(Losses.SSM)
optimizer = optim.Adam(dgssm.parameters(), lr=0.001)

coefficient_distance = nn.MSELoss()

epochs = 1000
steps_per_epoch = 10
batch_size = 32

fixed_index = 0
_, (ssm_target, mesh_target) = ds[fixed_index]
input_pc, input_labels = super(CorrespondingPointDataset, ds).__getitem__(0)
input_pc = input_pc.to(device)
ssm_target = ssm_target.to(device)
mesh_target = mesh_target.to(device)

ssm_valid_error = torch.zeros(epochs, device=device)
valid_corr_error = torch.zeros_like(ssm_valid_error, device=device)
train_corr_error = torch.zeros_like(ssm_valid_error, device=device)
loss_progression = torch.zeros_like(ssm_valid_error, device=device)
train_coefficient_error = torch.zeros_like(ssm_valid_error, device=device)
valid_coefficient_error = torch.zeros_like(ssm_valid_error, device=device)
# components_progression = {'Point-Loss': torch.zeros_like(ssm_valid_error, device=device),
#                           'Coefficients': torch.zeros_like(ssm_valid_error, device=device)}
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        dgssm.train()

        # get randomly transformed input and target data
        inputs_transformed, _, shape_transformed = \
            get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target)

        # train the model
        output = dgssm(inputs_transformed)

        target_weights = dgssm.ssm(shape_transformed)

        optimizer.zero_grad()
        loss = criterion(output[0], shape_transformed)# , components = criterion(output, (mesh_transformed, target_weights))
        loss.backward()
        optimizer.step()

        loss_progression[epoch] += loss.item() / steps_per_epoch
        # components_progression['Point-Loss'][epoch] += components['Point-Loss'].item() / steps_per_epoch
        # components_progression['Coefficients'][epoch] += components['Coefficients'].item() / steps_per_epoch
        with torch.no_grad():
            train_corr_error[epoch] += corresponding_point_distance(output[0], shape_transformed).mean() / steps_per_epoch
            train_coefficient_error[epoch] += coefficient_distance(output[1], target_weights) / steps_per_epoch

        # validation
        dgssm.eval()
        with torch.no_grad():
            inputs_transformed, _, shape_transformed = \
                get_random_transformed_data(batch_size, device, input_pc, mesh_target, ssm_target)

            # validate DGSSM
            output = dgssm(inputs_transformed)
            target_weights = dgssm.ssm(shape_transformed)
            # _, valid_components = criterion(output, (mesh_transformed, target_weights))

            valid_corr_error[epoch] += corresponding_point_distance(output[0], shape_transformed).mean() / steps_per_epoch#torch.sqrt(valid_components['Point-Loss']) / steps_per_epoch
            valid_coefficient_error[epoch] += coefficient_distance(output[1], target_weights) / steps_per_epoch

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
    print(f'\tTotal: {loss_progression[epoch].item():.4f} | Coefficients: {train_coefficient_error[epoch]:.4f} | Corr. Point Error: {train_corr_error[epoch].item():.4f} mm')

    print('VALIDATION (Corr. Point Error)')
    print(f'\tDGSSM: {valid_corr_error[epoch].item():.4f} mm | SSM baseline: {ssm_valid_error[epoch].item():.4f} mm | Coefficients: {valid_coefficient_error[epoch]:.4f}\n')

# save model
dgssm.save(os.path.join(out_dir, 'model.pth'))

# loss plots
plt.figure()
plt.plot(loss_progression.cpu())
plt.title('train loss')
plt.savefig(os.path.join(out_dir, 'training_progression.png'))
# for key, val in components_progression.items():
#     plt.figure()
#     plt.plot(val.cpu())
#     plt.title(key)
if show:
    plt.show()

with open(os.path.join(out_dir, 'training_progression.csv'), 'w') as progression_csv:
    writer = csv.writer(progression_csv)
    writer.writerow(['Train Loss'] + loss_progression.tolist())
    writer.writerow(['Train Corr. Point Error [mm]'] + train_corr_error.tolist())
    writer.writerow(['Train Coefficient MSE'] + train_coefficient_error.tolist())
    writer.writerow(['Validation Corr. Point Error [mm]'] + valid_corr_error.tolist())
    writer.writerow(['Valid Coefficient MSE'] + valid_coefficient_error.tolist())
    writer.writerow(['SSM Error [mm]'] + ssm_valid_error.tolist())

# compute statistics of the random data for reference
inputs_transformed, mesh_transformed, shape_transformed = \
    get_random_transformed_data(50, device, input_pc, mesh_target, ssm_target)

mean_valid_data_distance = mean_pairwise_shape_dist(shape_transformed, mesh_transformed)
print(f'MEAN DATASET DISTANCES: {mean_valid_data_distance.item():.4f} mm\n')

# plot some results
with torch.no_grad():
    # validate DGSSM
    output = dgssm(inputs_transformed)
    target_weights = dgssm.ssm(shape_transformed)

    # SSM baseline error of validation data
    ssm_reconstruction = dgssm.ssm.decode(target_weights)

visualize_point_cloud(input_pc.T, input_labels, title='original input points',
                      savepath=os.path.join(plot_dir, 'original_pc.png'), show=show)
for i, inp in enumerate(inputs_transformed[:5]):
    visualize_point_cloud(inp.T, input_labels, title='transformed input points',
                          savepath=os.path.join(plot_dir, f'{i}_transformed_pc.png'), show=show)
    visualize_point_cloud(ssm_reconstruction[i], torch.ones(ssm_reconstruction.shape[1]), title='ssm reconstruction',
                          savepath=os.path.join(plot_dir, f'{i}_ssm_reconstruction.png'), show=show)
    visualize_point_cloud(mesh_transformed.verts_list()[i], torch.ones(mesh_transformed.verts_list()[i].shape[0]),
                          title='mesh target', savepath=os.path.join(plot_dir, f'{i}_mesh_target.png'), show=show)
    visualize_point_cloud(output[0][i], torch.ones(output[0].shape[1]), title='DGSSM prediction',
                          savepath=os.path.join(plot_dir, f'{i}_dgssm_prediction.png'), show=show)

