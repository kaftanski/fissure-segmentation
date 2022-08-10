import csv
import os
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from pytorch3d.transforms import random_rotations
from pytorch3d.transforms.transform3d import Transform3d
from torch import optim, nn

from data import CorrespondingPointDataset
from losses.ssm_loss import corresponding_point_distance, CorrespondingPointDistance
from models.dgcnn import DGCNNReg
from models.dgcnn_opensrc import DGCNN
from models.modelio import LoadableModel, store_config_args
from visualization import point_cloud_on_axis


class AffineDGCNN(DGCNNReg):
    def __init__(self, k, in_features=3, num_outputs=3):
        super(AffineDGCNN, self).__init__(k, in_features, num_outputs, spatial_transformer=False)
        # last layer bias is 0-init (like all biases per default)

# class AffineDGCNN(LoadableModel):
#     @store_config_args
#     def __init__(self, k, num_outputs=3):
#         super(AffineDGCNN, self).__init__()
#         dgcnn_args = SimpleNamespace(
#             k=k,
#             emb_dims=1024,  # length of global feature vector
#             dropout=0.5
#         )
#         self.dgcnn = DGCNN(dgcnn_args, output_channels=num_outputs)
#
#     def forward(self, x):
#         return self.dgcnn(x)


def random_transformation(n_samples, device, rotation=False, translation=True):
    rotations = random_rotations(n_samples, device=device)

    # random translation
    translation_amount = 30
    translations = (torch.rand(n_samples, 3, device=device) * 2 - 1) * translation_amount

    # assemble transform
    transforms = Transform3d(device=device)
    if rotation:
        transforms = transforms.rotate(rotations)
    if translation:
        transforms = transforms.translate(translations)
    return transforms


def rotate_around_center(shapes, transforms: Transform3d):
    assert shapes.ndim == 3
    translation_to_center = shapes.mean(1, keepdim=True)
    return transforms.transform_points(shapes - translation_to_center) + translation_to_center


def mean_pairwise_shape_dist(shapes):
    corr_dists = []
    for i in range(shapes.shape[0]):
        for j in range(shapes.shape[0]):
            if i == j:
                continue

            corr_dists.append(corresponding_point_distance(shapes[i], shapes[j]).mean())

    corr_dists = torch.tensor(corr_dists)
    return corr_dists.mean()


def get_batch(target_shape, batch_size, device):
    augmentations = random_transformation(batch_size, device)
    shapes_batch = rotate_around_center(target_shape, augmentations)
    target_translation = augmentations.get_matrix()[:, 3, :3]

    # z-normalize
    mean = shapes_batch.mean(dim=(0, 1), keepdim=True)
    std = shapes_batch.std(dim=(0, 1), keepdim=True)
    normalized_shapes = shapes_batch - mean
    normalized_shapes = normalized_shapes / std
    return shapes_batch, normalized_shapes, mean, std, target_translation


show = True

out_dir = 'results/dgcnn_translation_sanitycheck'
os.makedirs(out_dir, exist_ok=True)
plot_dir = os.path.join(out_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

device = 'cuda:3'

ds = CorrespondingPointDataset(1024, 'cnn')

# # augment shapes with random transformations
# n_augment = 10  # times the data set
# n_shapes = shapes.shape[0]
# for i in range(n_augment):
#     augmentations = random_transformation(n_shapes, device)
#     shapes = torch.concat([shapes, rotate_around_center(shapes[:n_shapes], augmentations)], dim=0)

model = AffineDGCNN(k=40).to(device)

# for i in range(5):
#     visualize_point_cloud(shapes[i], torch.ones(shapes.shape[1]), title='train shape')
# visualize_point_cloud(dgssm.ssm.mean_shape.data.squeeze().view(2048, 3), torch.ones(shapes.shape[1]), title='mean shape')

criterion = CorrespondingPointDistance()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
steps_per_epoch = 10
batch_size = 8

fixed_index = 0
_, target_shape = ds[fixed_index]
target_shape = target_shape.unsqueeze(0).to(device)

loss_progression = torch.zeros(epochs, device=device)
train_corr_error = torch.zeros_like(loss_progression, device=device)
valid_corr_error = torch.zeros_like(loss_progression, device=device)
train_pred_std = torch.zeros_like(loss_progression, device=device)
valid_pred_std = torch.zeros_like(loss_progression, device=device)
target_translation_std = torch.zeros_like(loss_progression, device=device)
for epoch in range(epochs):
    ep_start = time.time()
    for step in range(steps_per_epoch):
        model.train()

        # only use one transformed shape for training
        shapes_batch, _, _, _, target_translation = get_batch(target_shape, batch_size, device)

        # compute prediction
        translation_pred = model(shapes_batch.transpose(1, 2))
        pred_shapes = target_shape + translation_pred.squeeze().unsqueeze(1) #* std  # denormalize translation

        # training step
        optimizer.zero_grad()
        loss = criterion(pred_shapes, shapes_batch)
        # loss = 0.5 * criterion(translation_pred, target_translation) + 0.5 * criterion(pred_shapes, shapes_batch)
        loss.backward()
        optimizer.step()

        loss_progression[epoch] += loss.item() / steps_per_epoch
        # additional metrics
        with torch.no_grad():
            train_corr_error[epoch] += corresponding_point_distance(pred_shapes, shapes_batch).mean() / steps_per_epoch
            train_pred_std[epoch] += translation_pred.std() / steps_per_epoch
            target_translation_std[epoch] += target_translation.std() / steps_per_epoch

        # validation
        model.eval()
        with torch.no_grad():
            shapes_batch, _, _, _, target_translation = get_batch(target_shape, batch_size, device)

            # validate DGSSM
            translation_pred = model(shapes_batch.transpose(1, 2))
            pred_shapes = target_shape + translation_pred.squeeze().unsqueeze(1) #* std

            valid_corr_error[epoch] += corresponding_point_distance(pred_shapes, shapes_batch).mean() / steps_per_epoch
            valid_pred_std[epoch] += translation_pred.std() / steps_per_epoch

    # plot some results
    if not epoch % 20 and show:
        for i in range(len(pred_shapes)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            point_cloud_on_axis(ax, pred_shapes[i], c='r', label='prediction', title=f'Epoch {epoch}. Translation: {[int(t.item()+0.5) for t in target_translation[i]]}')
            point_cloud_on_axis(ax, shapes_batch[i], c='b', label='target')

            if show:
                plt.show()

    print(f'EPOCH {epoch} (took {time.time() - ep_start:.4f} s)')
    print(f'\tLoss: {loss_progression[epoch].item():.4f} | Corr. Point Error: {train_corr_error[epoch].item():.4f} mm | Prediction StdDev: {train_pred_std[epoch].item():.4f}')
    print(f'\tValidation Corr. Point Error: {valid_corr_error[epoch].item():.4f} mm | Prediction StdDev: {valid_pred_std[epoch].item():.4f}')
    print(f'\tTarget StdDev: {target_translation_std[epoch].item():.4f}\n')

# save model
model.save(os.path.join(out_dir, 'model.pth'))

# loss plots
plt.figure()
plt.plot(loss_progression.cpu())
plt.title('train loss')
plt.savefig(os.path.join(out_dir, 'training_progression.png'))
if show:
    plt.show()

with open(os.path.join(out_dir, 'training_progression.csv'), 'w') as progression_csv:
    writer = csv.writer(progression_csv)
    writer.writerow(['Train Loss'] + loss_progression.tolist())
    writer.writerow(['Train Corr. Point Error [mm]'] + train_corr_error.tolist())
    writer.writerow(['Train Pred. StdDev'] + train_pred_std.tolist())
    writer.writerow(['Validation Corr. Point Error [mm]'] + valid_corr_error.tolist())
    writer.writerow(['Valid Pred. StdDev'] + valid_pred_std.tolist())
    writer.writerow(['Target StdDev'] + target_translation_std.tolist())

# plot some results
for i in range(len(pred_shapes)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_on_axis(ax, pred_shapes[i], c='r', label='prediction', title=f'Translation: {[int(t.item()+0.5) for t in target_translation[i]]}')
    point_cloud_on_axis(ax, shapes_batch[i], c='b', label='target')

    fig.savefig(os.path.join(plot_dir, f'pred{i}.png'), dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# compute statistics of the random data for reference
augmentations = random_transformation(batch_size, device)
shapes_batch = rotate_around_center(target_shape, augmentations)

mean_valid_data_distance = mean_pairwise_shape_dist(shapes_batch)
print(f'MEAN DATASET DISTANCES: {mean_valid_data_distance.item():.4f} mm\n')
