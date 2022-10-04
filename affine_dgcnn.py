import csv
import os
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from pytorch3d.transforms import so3_exp_map
from pytorch3d.transforms.transform3d import Transform3d
from torch import optim, nn

from data import CorrespondingPointDataset
from losses.ssm_loss import corresponding_point_distance, CorrespondingPointDistance
from models.dgcnn import DGCNNReg
from models.dgcnn_opensrc import DGCNN, PointNet
from models.modelio import LoadableModel, store_config_args
from utils.detached_run import run_detached_from_pycharm
from visualization import point_cloud_on_axis


class AffineDGCNN(DGCNNReg):
    def __init__(self, k, in_features=3, do_rotation=True, do_translation=True):
        super(AffineDGCNN, self).__init__(k, in_features, do_rotation*3 + do_translation*3, spatial_transformer=False)
        # last layer bias is 0-init (like all biases per default)
        self.rot = do_rotation
        self.trans = do_translation

    def forward(self, x):
        y = super(AffineDGCNN, self).forward(x).squeeze()
        index = 0
        if self.rot:
            rot = y[:, :3]
            index = 3
        else:
            rot = torch.zeros(x.shape[0], 3, device=x.device)

        if self.trans:
            trans = y[:, index:]
        else:
            trans = torch.zeros(x.shape[0], 3, device=x.device)

        return rot, trans


class AffineOpenDGCNN(LoadableModel):
    @store_config_args
    def __init__(self, k, do_rotation=True, do_translation=True):
        super(AffineOpenDGCNN, self).__init__()
        dgcnn_args = SimpleNamespace(
            k=k,
            emb_dims=1024,  # length of global feature vector
            dropout=0.
        )
        self.dgcnn = DGCNN(dgcnn_args, output_channels=do_rotation*3 + do_translation*3)
        self.rot = do_rotation
        self.trans = do_translation

    def forward(self, x):
        y = self.dgcnn(x)
        index = 0
        if self.rot:
            rot = y[:, :3]
            index = 3
        else:
            rot = torch.zeros(x.shape[0], 3, device=x.device)

        if self.trans:
            trans = y[:, index:]
        else:
            trans = torch.zeros(x.shape[0], 3, device=x.device)

        return rot, trans


class AffinePointNet(LoadableModel):
    @store_config_args
    def __init__(self, k, do_rotation=True, do_translation=True):
        super(AffinePointNet, self).__init__()
        dgcnn_args = SimpleNamespace(
            k=k,
            emb_dims=1024,  # length of global feature vector
            dropout=0.
        )
        self.pointnet = PointNet(dgcnn_args, output_channels=do_rotation * 3 + do_translation * 3)
        self.rot = do_rotation
        self.trans = do_translation

    def forward(self, x):
        y = self.pointnet(x)
        index = 0
        if self.rot:
            rot = y[:, :3]
            index = 3
        else:
            rot = torch.zeros(x.shape[0], 3, device=x.device)

        if self.trans:
            trans = y[:, index:]
        else:
            trans = torch.zeros(x.shape[0], 3, device=x.device)

        return rot, trans


MODELS = {
    'DGCNN': AffineDGCNN,
    'OpenDGCNN': AffineOpenDGCNN,
    'PointNet': AffinePointNet
}


def random_transformation(n_samples, device, rotation=True, translation=True):
    # random rotation angles
    if rotation:
        random_angles = (torch.rand(n_samples, 3, device=device) * 2 - 1) * 2
    else:
        random_angles = torch.zeros(n_samples, 3, device=device)

    # random translation vectors
    if translation:
        translation_amount = 0.2  # for [-1, 1] grid coordinates
        translations = (torch.rand(n_samples, 3, device=device) * 2 - 1) * translation_amount
    else:
        translations = torch.zeros(n_samples, 3, device=device)

    # assemble transform
    transforms = compose_transform(random_angles, translations)
    return transforms, random_angles, translations


def compose_transform(log_rotation_matrix: torch.Tensor, translation: torch.Tensor):
    t = Transform3d(device=log_rotation_matrix.device) \
        .rotate(so3_exp_map(log_rotation_matrix)) \
        .translate(translation)
    # log_rotation_matrix is an R^3 vector of which the direction is the rotation axis and the magnitude is the
    # angle magnitude of rotation around the axis
    return t


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


def get_batch(target_shape, batch_size, device, do_rotation, do_translation):
    augmentations, target_angles, target_translation = random_transformation(batch_size, device, do_rotation, do_translation)
    shapes_batch = rotate_around_center(target_shape, augmentations)
    return shapes_batch, target_angles, target_translation


def run_example(model, epochs, steps_per_epoch, batch_size, do_rotation, do_translation, use_point_loss, use_param_loss, show, device):
    assert use_point_loss or use_param_loss
    assert do_rotation or do_translation

    # output directories
    out_dir = f'results/affine_dgcnn_experiments/{model}_sanity_check/{model}{"_rot" if do_rotation else ""}{"_translation" if do_translation else ""}{"_pointloss" if use_point_loss else ""}{"_paramloss" if use_param_loss else ""}'
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # dataset
    ds = CorrespondingPointDataset(1024, 'cnn', corr_folder='results/corresponding_points/simple/fissures')

    # setup model
    model = MODELS[model](k=40, do_rotation=do_rotation, do_translation=do_translation).to(device)

    # setup losses
    point_loss = CorrespondingPointDistance()
    param_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # pick one exemplary shape
    fixed_index = 0
    target_shape = ds.corr_points[fixed_index]
    target_shape = target_shape.unsqueeze(0).to(device)

    # rescale shape into [-1, 1] coordinate grid
    mean = target_shape.mean(dim=(0, 1), keepdim=True)
    target_shape = target_shape - mean
    scale, _ = torch.max(torch.sqrt(torch.sum(torch.square(target_shape), dim=2, keepdim=True)), dim=1, keepdim=True)
    target_shape = target_shape / scale

    # setup training progression tensors
    loss_progression = torch.zeros(epochs, device=device)
    train_corr_error = torch.zeros_like(loss_progression, device=device)
    valid_corr_error = torch.zeros_like(loss_progression, device=device)
    train_angle_error = torch.zeros_like(loss_progression, device=device)
    valid_angle_error = torch.zeros_like(loss_progression, device=device)
    train_trans_error = torch.zeros_like(loss_progression, device=device)
    valid_trans_error = torch.zeros_like(loss_progression, device=device)
    train_pred_std = torch.zeros_like(loss_progression, device=device)
    valid_pred_std = torch.zeros_like(loss_progression, device=device)
    target_translation_std = torch.zeros_like(loss_progression, device=device)
    for epoch in range(epochs):
        ep_start = time.time()
        for step in range(steps_per_epoch):
            model.train()

            # only use one transformed shape for training
            shapes_batch, target_angles, target_translation = \
                get_batch(target_shape, batch_size, device, do_rotation, do_translation)

            # compute prediction
            angles_pred, translation_pred = model(shapes_batch.transpose(1, 2))
            pred_transform = compose_transform(angles_pred, translation_pred)
            pred_shapes = rotate_around_center(target_shape, pred_transform)

            # training step
            optimizer.zero_grad()
            pts_ls = point_loss(pred_shapes, shapes_batch)
            par_ls = param_loss(torch.cat([angles_pred, translation_pred], dim=1),
                                torch.cat([target_angles, target_translation], dim=1))
            loss = 0
            if use_point_loss:
                loss += pts_ls
            if use_param_loss:
                loss += par_ls
            loss /= use_point_loss + use_param_loss

            loss.backward()
            optimizer.step()

            # remember training metrics
            with torch.no_grad():
                loss_progression[epoch] += loss.item() / steps_per_epoch
                train_angle_error[epoch] += (angles_pred - target_angles).square().mean().sqrt() / steps_per_epoch
                train_trans_error[epoch] += ((translation_pred - target_translation) * scale.squeeze()).square().sum(1).sqrt().mean() / steps_per_epoch
                train_corr_error[epoch] += corresponding_point_distance(pred_shapes * scale,
                                                                        shapes_batch * scale).mean() / steps_per_epoch
                train_pred_std[epoch] += translation_pred.std() / steps_per_epoch
                target_translation_std[epoch] += target_translation.std() / steps_per_epoch

            # validation
            model.eval()  # TODO: validation loss much lower than train loss
            with torch.no_grad():
                shapes_batch, target_angles, target_translation = \
                    get_batch(target_shape, batch_size, device, do_rotation, do_translation)

                angles_pred, translation_pred = model(shapes_batch.transpose(1, 2))
                pred_transform = compose_transform(angles_pred, translation_pred)
                pred_shapes = rotate_around_center(target_shape, pred_transform)

                valid_angle_error[epoch] += (angles_pred - target_angles).square().mean().sqrt() / steps_per_epoch
                valid_trans_error[epoch] += ((translation_pred - target_translation) * scale.squeeze()).square().sum(1).sqrt().mean() / steps_per_epoch
                valid_corr_error[epoch] += corresponding_point_distance(pred_shapes * scale,
                                                                        shapes_batch * scale).mean() / steps_per_epoch
                valid_pred_std[epoch] += translation_pred.std() / steps_per_epoch

        # plot some results
        if not epoch % 20 and show:
            for i in range(len(pred_shapes)):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                point_cloud_on_axis(ax, pred_shapes[i] * scale, c='r', label='prediction',
                                    title=f'Epoch {epoch}. Translation: {[int(t.item() + 0.5) for t in target_translation[i] * scale.squeeze()]}')
                point_cloud_on_axis(ax, shapes_batch[i] * scale, c='b', label='target')

                if show:
                    plt.show()

        print(f'EPOCH {epoch} (took {time.time() - ep_start:.4f} s)')
        print(
            f'\tLoss: {loss_progression[epoch].item():.4f} | Corr. Point Error: {train_corr_error[epoch].item():.4f} mm | Angle Error: {train_angle_error[epoch].item():.4f} | Translation Error: {train_trans_error[epoch].item():.4f} mm | Prediction StdDev: {train_pred_std[epoch].item():.4f}')
        print(
            f'\tValidation Corr. Point Error: {valid_corr_error[epoch].item():.4f} mm | Angle Error: {valid_angle_error[epoch].item():.4f} | Translation Error: {valid_trans_error[epoch].item():.4f} mm | Prediction StdDev: {valid_pred_std[epoch].item():.4f}')
        print(f'\tTarget StdDev: {target_translation_std[epoch].item():.4f}\n')

    # save model
    model.save(os.path.join(out_dir, 'model.pth'))

    # plot some results
    errors = corresponding_point_distance(pred_shapes * scale, shapes_batch * scale).mean(-1)
    for i in range(len(pred_shapes)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        point_cloud_on_axis(ax, pred_shapes[i] * scale, c='r', label='prediction',
            title=f'Translation: {[int(t.item() + 0.5) for t in target_translation[i] * scale.squeeze()]}. P2P-Error: {errors[i].item():.4f}')
        point_cloud_on_axis(ax, shapes_batch[i] * scale, c='b', label='target')

        fig.savefig(os.path.join(plot_dir, f'pred{i}.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    # compute statistics of the random data for reference
    augmentations, _, _ = random_transformation(200, device, do_rotation, do_translation)
    shapes_batch = rotate_around_center(target_shape, augmentations)

    initial_error = corresponding_point_distance(target_shape * scale, shapes_batch * scale).mean()  # mean_pairwise_shape_dist(shapes_batch).item()
    print(f'INITIAL ERROR: {initial_error:.4f} mm\n')

    # assemble all statistics
    metrics = {
        'Train Loss': loss_progression,
        'Train Corr. Point Error [mm]': train_corr_error,
        'Train Angle RMSE [deg]': train_angle_error,
        'Train Translation RMSE [mm]': train_trans_error,
        'Train Pred. StdDev': train_pred_std,
        'Valid Corr. Point Error [mm]': valid_corr_error,
        'Valid Angle RMSE [deg]': valid_angle_error,
        'Valid Translation RMSE [mm]': valid_trans_error,
        'Valid Pred. StdDev': valid_pred_std,
        'Target StdDev': target_translation_std
    }

    # plot metrics
    for title, values in metrics.items():
        plt.figure()
        plt.plot(values.cpu())
        plt.title(title)
        fn = title.replace(' ', '_').replace('.', '').replace('[mm]', '').replace('[deg]', '')
        plt.savefig(os.path.join(out_dir, fn + '.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    # write raw data
    with open(os.path.join(out_dir, 'training_progression.csv'), 'w') as progression_csv:
        writer = csv.writer(progression_csv)
        for title, values in metrics.items():
            writer.writerow([title] + values.tolist())

        writer.writerow(['Initial Error [mm]', initial_error])


if __name__ == '__main__':
    run_detached_from_pycharm()
    epochs = 1000
    steps_per_epoch = 10
    batch_size = 8

    show = False
    device = 'cuda:2'

    model = 'DGCNN'
    # run_example(model, epochs, steps_per_epoch, batch_size, do_rotation=True, do_translation=False, use_point_loss=True,
    #             use_param_loss=False, show=show, device=device)

    for do_rotation in [False, True]:
        for do_translation in [False, True]:
            if not (do_rotation or do_translation):
                continue
            for use_param_loss in [False, True]:
                for use_point_loss in [False, True]:
                    if not (use_param_loss or use_point_loss):
                        continue
                    run_example(model, epochs, steps_per_epoch, batch_size, do_rotation, do_translation, use_point_loss, use_param_loss, show, device)
