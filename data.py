import collections
import csv
import glob
import os
import os.path
import pickle
import warnings
from abc import ABC
from copy import deepcopy
from glob import glob
from typing import OrderedDict

import SimpleITK as sitk
import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.transforms import so3_log_map, Transform3d
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import Dataset

from augmentations import image_augmentation, point_augmentation, compose_transform, transform_meshes
from constants import POINT_DIR, IMG_DIR
from shape_model.ssm import load_shape
from utils.general_utils import load_points, ALIGN_CORNERS, kpts_to_grid, kpts_to_world, inverse_affine_transform, \
    decompose_similarity_transform, load_meshes, o3d_to_pt3d_meshes
from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor, multiple_objects_morphology, \
    get_resample_factors, load_image_metadata

IMG_MIN = -1000
IMG_MAX = 1500


def _load_files_from_file_list(item, the_list):
    if isinstance(item, int):
        item = [item]

    item = torch.arange(len(the_list))[item]

    result = []
    for i in item:
        filename = the_list[i]
        # load each item
        if filename is not None:
            if filename.endswith('.nii.gz'):
                instance = sitk.ReadImage(filename)
            elif filename.endswith('.csv'):
                instance = load_landmarks(filename)
            else:
                instance = None
        else:
            instance = None
        result.append(instance)

    if len(item) == 1:
        result = result[0]

    return result


class LungData(Dataset):
    def __init__(self, folder):
        self.images = sorted(glob(os.path.join(folder, '*_img_*.nii.gz')))
        self.lung_masks = sorted(glob(os.path.join(folder, '*_mask_*.nii.gz')))
        self.landmarks = []
        self.fissures = []
        self.fissure_meshes = []
        self.lobes = []
        self.lobe_meshes = []
        self.ids = []
        self.fissures_enhanced = []
        self.left_right_masks = []

        # fill missing landmarks and fissure segmentations with None
        for img in self.images:
            lm_file = img.replace('_img_', '_lms_').replace('.nii.gz', '.csv')
            if os.path.exists(lm_file):
                self.landmarks.append(lm_file)
            else:
                self.landmarks.append(None)

            fissure_file = img.replace('_img_', '_fissures_')
            if os.path.exists(fissure_file):
                self.fissures.append(fissure_file)
            else:
                self.fissures.append(None)

            lobes_file = img.replace('_img_', '_lobes_')
            if os.path.exists(lobes_file):
                self.lobes.append(lobes_file)
            else:
                self.lobes.append(None)

            enhanced_file = img.replace('_img_', '_fissures_enhanced_')
            if os.path.exists(enhanced_file):
                self.fissures_enhanced.append(enhanced_file)
            else:
                self.fissures_enhanced.append(None)

            mask_lr_file = img.replace('_img_', '_masklr_')
            if os.path.exists(mask_lr_file):
                self.left_right_masks.append(mask_lr_file)
            else:
                self.left_right_masks.append(None)

            case, _, sequence = img.split(os.sep)[-1].split('_')
            sequence = sequence.split('.')[0]
            self.ids.append((case, sequence))

            meshlist = sorted(glob(os.path.join(folder, f'{case}_mesh_{sequence}', f'{case}_fissure*_{sequence}.obj')))
            self.fissure_meshes.append(meshlist if meshlist else None)

            meshlist = sorted(glob(os.path.join(folder, f'{case}_mesh_{sequence}', f'{case}_lobe*_{sequence}.obj')))
            self.lobe_meshes.append(meshlist if meshlist else None)

    def get_image(self, item):
        return _load_files_from_file_list(item, self.images)

    def get_fissures(self, item):
        return _load_files_from_file_list(item, self.fissures)

    def get_regularized_fissures(self, item):
        # this specifies what data should be considered "regularized"
        return _load_files_from_file_list(item, [f.replace('_fissures_', '_fissures_poisson_') if f is not None
                                                 else None for f in self.fissures])

    def get_enhanced_fissures(self, item):
        return _load_files_from_file_list(item, self.fissures_enhanced)

    def get_landmarks(self, item):
        return _load_files_from_file_list(item, self.landmarks)

    def get_lung_mask(self, item):
        return _load_files_from_file_list(item, self.lung_masks)

    def get_left_right_lung_mask(self, item):
        return _load_files_from_file_list(item, self.left_right_masks)

    def get_filename(self, item):
        if isinstance(item, int):
            item = [item]

        item = torch.arange(len(self))[item]

        filenames = []
        for i in item:
            # remember the filename
            filenames.append(self.images[i])

        if len(item) == 1:
            filenames = filenames[0]

        return filenames

    def get_fissure_meshes(self, item):
        if self.fissure_meshes[item] is None:
            return None
        else:
            return tuple(o3d.io.read_triangle_mesh(m) for m in self.fissure_meshes[item])

    def get_lobe_meshes(self, item):
        if self.fissure_meshes[item] is None:
            return None
        else:
            return tuple(o3d.io.read_triangle_mesh(m) for m in self.lobe_meshes[item])

    def get_lobes(self, item):
        return _load_files_from_file_list(item, self.lobes)

    def get_id(self, item):
        return self.ids[item]

    def get_index(self, case, sequence):
        occurrence = list(j for j, fn in enumerate(self.images) if f'{case}_img_{sequence}' in fn)
        if occurrence:
            return occurrence[0]
        else:
            raise ValueError(f'No data with ID {case}_{sequence}.')

    def __getitem__(self, item):
        return self.get_image(item), self.get_fissures(item)

    def __len__(self):
        return len(self.images)


class CustomDataset(Dataset, ABC):
    def __init__(self, exclude_rhf, do_augmentation, binary, copd=False):
        self.exclude_rhf = exclude_rhf
        self._do_augmentation = do_augmentation
        self.binary = binary
        self.copd = copd
        try:
            getattr(self, 'ids')
        except AttributeError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    @property
    def num_classes(self):
        if self.binary:
            return 2
        else:
            return 3 if self.exclude_rhf or self.copd else 4

    @property
    def do_augmentation(self):
        return self._do_augmentation

    @do_augmentation.setter
    def do_augmentation(self, value):
        self._do_augmentation = value

    @do_augmentation.deleter
    def do_augmentation(self):
        del self._do_augmentation

    def get_class_weights(self):
        return None

    def get_batch_collate_fn(self):
        return None

    def split_data_set(self, split: OrderedDict[str, np.ndarray], fold_nr=None):
        train_ds = deepcopy(self)
        val_ds = deepcopy(self)

        # check if nnUnet format is used or not
        nnu = ('_img_' not in split['train'][0])

        for i in range(len(self.ids) - 1, -1, -1):
            case, sequence = self.ids[i]
            if not nnu:
                # my own split file creation method
                id = case + '_img_' + sequence
            else:
                # split file from nnunet
                id = case + '_' + sequence.replace('fixed', 'fix').replace('moving', 'mov')

            if id in split['val']:
                train_ds._pop_item(i)
            elif id in split['train']:
                val_ds._pop_item(i)
            else:
                warnings.warn(f'Train/Validation split incomplete: instance {id} is contained in neither.')
                val_ds._pop_item(i)
                train_ds._pop_item(i)

        val_ds.do_augmentation = False
        return train_ds, val_ds

    def _pop_item(self, i):
        # pop index from all lists in this dataset
        for name in dir(self):
            if '__' in name:
                continue
            attr = getattr(self, name)
            if isinstance(attr, list):
                attr.pop(i)


class ImageDataset(LungData, CustomDataset):
    def __init__(self, folder, resample_spacing=1.5, patch_size=(128, 128, 128), exclude_rhf=False,
                 do_augmentation=True, binary=False, copd=False):
        LungData.__init__(self, folder)
        CustomDataset.__init__(self, exclude_rhf, do_augmentation, binary)

        self.resample_spacing = resample_spacing
        self.patch_size = patch_size

        # remove images without fissure label
        def remove_indices(ls: list, indices):
            for i in sorted(indices)[::-1]:
                ls.pop(i)

        self.copd = copd

        if self.copd:
            to_remove = [i for i in range(len(self.fissures)) if self.fissures[i] is None or 'COPD' not in self.ids[i][0]]
        else:
            to_remove = [i for i in range(len(self.fissures)) if self.fissures[i] is None]

        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, list):
                remove_indices(attr, to_remove)

    def __getitem__(self, item):
        img = self.get_image(item)
        label = self.get_regularized_fissures(item)

        # change all labels to 1, if binary segmentation is desired
        change_map = {}
        if self.binary:
            for lbl in range(1, 4):
                change_map[lbl] = 1
        # change the right horizontal fissure to background, if it is to be excluded
        if self.exclude_rhf:
            change_map[3] = 0
        if change_map:
            change_label_filter = sitk.ChangeLabelImageFilter()
            change_label_filter.SetChangeMap(change_map)
            label = change_label_filter.Execute(label)

        # dilate fissures (so they don't vanish when downsampling)
        factors = get_resample_factors(label.GetSpacing(), target_spacing=self.resample_spacing)
        radius = [max(0, round(1/f - 1)) for f in factors]
        # print(radius)  # TODO: find a better way to preserve labels! (maybe through gt meshes)
        label = multiple_objects_morphology(label, radius=radius, mode='dilate')

        # resampling to unit spacing
        img = resample_equal_spacing(img, target_spacing=self.resample_spacing)
        label = resample_equal_spacing(label, target_spacing=self.resample_spacing, use_nearest_neighbor=True)

        # to tensor
        img_array = sitk_image_to_tensor(img).float().unsqueeze(0).unsqueeze(0).numpy()
        label_array = sitk_image_to_tensor(label).long().unsqueeze(0).unsqueeze(0).numpy()

        if self.do_augmentation:
            img_array, label_array = image_augmentation(img_array, label_array, patch_size=self.patch_size)

        # get inputs into range [-1, 1]
        img_array = normalize_img(img_array)

        return img_array.squeeze(), label_array.squeeze()

    def get_batch_collate_fn(self):
        def collate_fn(list_of_samples):
            # shapes = torch.zeros(len(list_of_samples), 3, dtype=torch.long)
            # for i, (img, label) in enumerate(list_of_samples):
            #     shapes[i] = torch.tensor(img.shape[-3:])
            #
            # median_shape, _ = torch.median(shapes, dim=0)

            img_batch = []
            label_batch = []
            for img, label in list_of_samples:
                # img = center_crop_3D_image(img, tuple(median_shape))
                # label = center_crop_3D_image(label, tuple(median_shape))

                img_batch.append(img)
                label_batch.append(label)

            # convert lists of arrays to one array, makes conversion to tensor faster
            img_batch = np.array(img_batch)
            label_batch = np.array(label_batch)

            return torch.from_numpy(img_batch).unsqueeze(1), torch.from_numpy(label_batch).long()

        return collate_fn

    def get_class_weights(self):
        frequency = torch.zeros(self.num_classes)
        for i in range(len(self)):
            f = self.get_regularized_fissures(i)
            for c in range(self.num_classes):
                frequency[c] += (sitk.GetArrayFromImage(f) == c).sum().item()

        class_weights = compute_class_weights(frequency)
        return class_weights


def normalize_img(img, min_val=IMG_MIN, max_val=IMG_MAX):
    return (img - min_val) / (max_val - min_val) * 2 - 1  # get inputs into range [-1, 1]


class PointDataset(CustomDataset):
    def __init__(self, sample_points, kp_mode,
                 folder=POINT_DIR, image_folder=IMG_DIR,
                 use_coords=True, patch_feat=None, exclude_rhf=False, lobes=False, binary=False, do_augmentation=True,
                 copd=False, all_to_device='cpu'):

        super(PointDataset, self).__init__(exclude_rhf=exclude_rhf, do_augmentation=do_augmentation, binary=binary, copd=copd)

        if lobes and binary:
            raise NotImplementedError(
                'Binary prediction for lobe labels is not implemented. Use fissure data or remove the binary option.')

        if not use_coords:
            raise ValueError('Coords have to be present for this to work...')
            # assert patch_feat is not None, 'Neither Coords nor Features specified for PointDataset'

        if kp_mode not in folder:
            self.folder = os.path.join(folder, kp_mode)
        else:
            self.folder = folder
        files = sorted(glob(os.path.join(self.folder, '*_coords_*')))
        self.image_folder = image_folder
        self.kp_mode = kp_mode
        self.use_coords = use_coords
        self.patch_feat = patch_feat
        self.lobes = lobes
        self.sample_points = sample_points
        self.points = []
        self.features = []
        self.labels = []
        for file in files:
            case, _, sequence = file.split('/')[-1].split('_')
            if self.copd:
                # ignore non-copd cases
                if 'COPD' not in case:
                    continue
            sequence = sequence.split('.')[0]
            pts, lbls, lobe_lbl, feat = load_points(self.folder, case, sequence, self.patch_feat, device=all_to_device)
            if lobes:
                lbls = lobe_lbl
            else:
                if exclude_rhf:
                    lbls[lbls == 3] = 0
            self.points.append(pts)
            self.labels.append(lbls)
            if feat is not None:
                self.features.append(feat)
            else:
                self.features.append(torch.empty(0, pts.shape[1]))
            self.ids.append((case, sequence))

        # load img sizes for (un-)normalization
        self.spacings = []
        self.img_sizes_index = []
        self.img_sizes_world = []
        for case, sequence in self.ids:
            size, spacing = load_image_metadata(os.path.join(image_folder, f"{case}_img_{sequence}.nii.gz"))
            self.spacings.append(spacing)
            self.img_sizes_index.append(size)
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes_world.append(size_world)

    # @property
    # def num_classes(self):
    #     return 2 if self.binary else max(len(torch.unique(lbl)) for lbl in self.labels)

    def __getitem__(self, item, return_aug_transform=False):
        # randomly sample points
        feat = self.features[item]
        transform = None
        if self.use_coords:
            pts = self.points[item]
            if self._do_augmentation:
                # random transform of the coordinates
                pts, transform = point_augmentation(pts.unsqueeze(0))
                pts = pts.squeeze(0)

            x = torch.cat([pts, feat], dim=0)
        else:
            x = feat

        lbls = self.labels[item]
        sample = torch.randperm(x.shape[1])[:self.sample_points]

        lbls_sampled = lbls[sample]
        if self.binary:
            lbls_sampled = (lbls[sample] != 0).long()

        if return_aug_transform:
            return x[:, sample], lbls_sampled, transform

        return x[:, sample], lbls_sampled

    def get_class_weights(self):
        frequency = torch.zeros(self.num_classes)
        for lbl in self.labels:
            for c in range(self.num_classes):
                frequency[c] += torch.sum(lbl == c)

        class_weights = compute_class_weights(frequency)
        return class_weights

    def get_full_pointcloud(self, i):
        if self.use_coords:
            x = torch.cat([self.points[i], self.features[i]], dim=0)
        else:
            x = self.features[i]

        lbls = self.labels[i]
        if self.binary:
            lbls[lbls != 0] = 1

        return x, lbls

    def get_coords(self, i):
        return self.points[i]

    def split_data_set(self, split: OrderedDict[str, np.ndarray], fold_nr=None):
        if self.copd:
            # this is a pure validation data set so no splitting required, no training-ds either!
            if self.kp_mode == 'cnn':
                assert fold_nr is not None, 'Please specify the number of the fold to use'
                # different folds yielded different pre-seg CNNs
                return None, type(self)(self.sample_points, self.kp_mode, os.path.join(self.folder, f"fold{fold_nr}"),
                    self.image_folder, self.use_coords, self.patch_feat, self.exclude_rhf, self.lobes, self.binary,
                    self.do_augmentation, self.copd)
            else:
                return None, self
        else:
            return super().split_data_set(split)


def compute_class_weights(class_frequency):
    class_frequency = class_frequency / class_frequency.sum()
    print(f'Label frequency in point data set: {class_frequency.tolist()}')

    class_weights = 1 - class_frequency
    class_weights *= len(class_frequency)
    print(f'Class weights: {class_weights.tolist()}')

    return class_weights


class CorrespondingPointDataset(PointDataset):
    def __init__(self, sample_points, kp_mode, point_folder=POINT_DIR,
                 use_coords=True, patch_feat=None, corr_folder="./results/corresponding_points",
                 image_folder=IMG_DIR, do_augmentation=True, undo_affine_reg=False):
        super(CorrespondingPointDataset, self).__init__(sample_points, kp_mode, point_folder, image_folder, use_coords,
                                                        patch_feat, exclude_rhf=True, do_augmentation=False)
        self.corr_points = CorrespondingPoints(corr_folder)
        self._do_augmentation_correspondingly = do_augmentation

        # # load meshes for supervision
        # self.meshes = []
        # for case, sequence in self.ids:
        #     meshes = load_meshes(image_folder, case, sequence, obj_name='fissure')
        #     self.meshes.append(meshes)

        # remove non-matched data points
        self._remove_non_matched_from_corr_points()

        # # affine transform the ground truth meshes to fit the corresponding points
        # for meshes, transforms in zip(self.meshes, self.corr_points.transforms):
        #     for m, t in zip(meshes, transforms):
        #         # rotate around [0,0,0] (o3d uses the PC's center of mass by default)
        #         m.rotate(t[..., :3], center=np.zeros([3, 1]))
        #         m.translate(t[..., -1])

    @property
    def do_augmentation(self):
        return self._do_augmentation_correspondingly

    @do_augmentation.setter
    def do_augmentation(self, value):
        self._do_augmentation_correspondingly = value

    def __getitem__(self, item, return_aug_transform=False):
        pts, lbl = super(CorrespondingPointDataset, self).__getitem__(item)
        target_corr_pts, norm_transform = self.normalize_pc(self.corr_points[item], item, return_transform=True)

        # we want the network to predict the inverse rigid transformation (back into moving space)
        inverse_prereg_transform = compose_transform(
            so3_log_map(self.corr_points.transforms[item]['rotation'].T.unsqueeze(0)),
            self.corr_points.transforms[item]['translation'].unsqueeze(0),
            torch.tensor([[self.corr_points.transforms[item]['scale']]])).inverse()

        # prereg transform is in unnormalized form, therefore apply inverse normalization first
        # then back into normalized space
        target_transform = norm_transform.inverse().compose(inverse_prereg_transform).compose(norm_transform)

        # augmentations
        if self._do_augmentation_correspondingly:
            # augment the input points
            pts_augment, aug_transform = point_augmentation(pts[:3].unsqueeze(0))
            pts[:3] = pts_augment.squeeze(0)

            # augmentations need to be predicted as well (happen in moving space, therefore after inverse prereg)
            # the transformation therefore bridges the spaces:
            # F -> prereg^-1 -> M -> aug -> M_aug
            target_transform = target_transform.compose(aug_transform)

        # decompose the transformation matrix (shear should stay 0 except for some quantization error)
        target_translation, target_rotation, target_scale = decompose_similarity_transform(
            target_transform.get_matrix().squeeze().T)  # other scheme of notating homogeneous coordinates -> transpose
        # target_translation, target_rotation, target_scale = compose_rigid_transforms(
        #     norm_transform.inverse(), inverse_prereg_transform, norm_transform, aug_transform)
        target_rotation = target_rotation.T.unsqueeze(0)
        target_scale = target_scale.unsqueeze(0)
        target_translation = target_translation.unsqueeze(0)
        recomposed = Transform3d()\
            .rotate(target_rotation) \
            .scale(target_scale) \
            .translate(target_translation)
        assert torch.allclose(recomposed.get_matrix(), target_transform.get_matrix(), atol=1e-7)

        target_transform_params = torch.cat([so3_log_map(target_rotation), target_translation, target_scale], dim=1).squeeze()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # point_cloud_on_axis(ax, pts[:3].T[lbl > 0], c='b', label='input')
        # point_cloud_on_axis(ax, target_transform.transform_points(target_corr_pts), c='r', label='target', alpha=0.5)
        # plt.show()

        return pts, (target_corr_pts, target_transform_params)  # (self.corr_points[item], o3d_to_pt3d_meshes([concat_meshes]))

    def split_data_set(self, split: OrderedDict[str, np.ndarray], fold_nr=None):
        train_ds, val_ds = super(CorrespondingPointDataset, self).split_data_set(split)
        train_ds._remove_non_matched_from_corr_points()
        val_ds._remove_non_matched_from_corr_points()
        return train_ds, val_ds

    @property
    def num_classes(self):
        return self.corr_points.num_objects

    def _remove_non_matched_from_corr_points(self):
        for i in range(len(self.ids) - 1, -1, -1):
            if self.ids[i] not in self.corr_points.ids:
                self._pop_item(i)

        for i in range(len(self.corr_points) - 1, -1, -1):
            if self.corr_points.ids[i] not in self.ids:
                self.corr_points.points.pop(i)
                self.corr_points.transforms.pop(i)
                self.corr_points.ids.pop(i)

    def normalize_pc(self, pc, index, return_transform=False):
        return kpts_to_grid(pc, self.img_sizes_world[index][::-1], align_corners=ALIGN_CORNERS, return_transform=return_transform)

    def unnormalize_pc(self, pc, index):
        return kpts_to_world(pc, self.img_sizes_world[index][::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_mean_pc(self, pc):
        mean_size = torch.stack([torch.tensor(self.img_sizes_world[i][::-1]) for i in range(len(self))], dim=0).mean(0)
        return kpts_to_world(pc, mean_size, align_corners=ALIGN_CORNERS)

    def get_normalized_corr_datamatrix_with_affine_reg(self):
        return torch.stack([self.normalize_pc(pts, i) for i, pts in enumerate(self.corr_points.points)], dim=0)

    # def get_batch_collate_fn(self):
    #     def collate_fn(list_of_samples):
    #         pcs = []
    #         corr_pts = []
    #         meshes = []
    #         for pc, (corr, mesh) in list_of_samples:
    #             pcs.append(pc)
    #             corr_pts.append(corr)
    #             meshes.append(mesh)
    #
    #         return torch.stack(pcs, dim=0), (torch.stack(corr_pts), join_meshes_as_batch(meshes))
    #
    #     return collate_fn


class CorrespondingPoints:
    def __init__(self, folder="./results/corresponding_points"):
        self.folder = folder
        self.points = []
        self.transforms = []
        self.ids = []
        files = sorted(glob(os.path.join(self.folder, '*_corr_pts.npz')))
        for f in files:
            s, t = load_shape(f)
            self.points.append(s)
            self.transforms.append(t)
            tail = f.split(os.sep)[-1]
            case, sequence = tail.split('_')[:2]
            self.ids.append((case, sequence))

        # points are corresponding, one label is applicable to all
        self.label = load_shape(files[0], return_labels=True)[-1]
        self.num_objects = len(np.unique(self.label))

    def __getitem__(self, item):
        return self.points[item]

    def __len__(self):
        return len(self.points)

    def get_shape_datamatrix_with_affine_reg(self):
        return torch.stack(self.points, dim=0)

    def get_points_without_affine_reg(self, index):
        points_registered = self.points[index]
        return self.inverse_affine_transform(points_registered, index)

    def inverse_affine_transform(self, pts_affine, index):
        unregistered = []
        trf = self.transforms[index]
        rot = trf['rotation'].to(pts_affine.device)
        trans = trf['translation'].to(pts_affine.device)
        for i, lbl in enumerate(self.label.unique()):
            unregistered.append(inverse_affine_transform(pts_affine[self.label == lbl], scaling=trf['scale'],
                                                         rotation_mat=rot, affine_translation=trans))

        return torch.concat(unregistered, dim=0)


class SampleFromMeshDS(CustomDataset):
    def __init__(self, folder, sample_points, fixed_object: int = None, exclude_rhf=False, lobes=False, mesh_as_target=True, device='cpu', all_to_device='cpu'):
        super(SampleFromMeshDS, self).__init__(exclude_rhf=exclude_rhf, binary=False, do_augmentation=True)

        self.sample_points = sample_points
        self.fixed_object = fixed_object
        self.mesh_as_target = mesh_as_target
        self.lobes = lobes
        self.device = device

        mesh_dirs = sorted(glob(os.path.join(folder, "*_mesh_*")))

        self.ids = []
        self.meshes = []
        self.pt3d_meshes = []
        self.img_sizes = []
        for mesh_dir in mesh_dirs:
            case, sequence = os.path.basename(mesh_dir).split("_mesh_")
            meshes = load_meshes(folder, case, sequence, obj_name='fissure' if not lobes else 'lobe')
            if not lobes and exclude_rhf:
                meshes = meshes[:2]
            self.meshes.append(meshes)
            self.pt3d_meshes.append(o3d_to_pt3d_meshes(meshes).to(all_to_device))
            self.ids.append((case, sequence))

            size, spacing = load_image_metadata(os.path.join(folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

        assert all(len(self.meshes[0]) == len(m) for m in self.meshes)
        self.num_objects = len(self.meshes[0])

    def __len__(self):
        return len(self.ids) * self.num_objects if self.fixed_object is None else len(self.ids)

    def __getitem__(self, item):
        # TODO: normalization and augmentation are performed on CPU, creating a bottleneck
        #  (could be done on GPU per batch by moving it into the collate function)
        meshes = self.meshes[self.continuous_to_pat_index(item)]
        obj_index = self.continuous_to_obj_index(item)
        current_mesh = meshes[obj_index]
        current_mesh_pt3d = self.pt3d_meshes[self.continuous_to_pat_index(item)][obj_index]

        # # sample point cloud from meshes
        # samples = current_mesh.sample_points_uniformly(number_of_points=self.sample_points)
        # samples = torch.from_numpy(np.asarray(samples.points)).float()

        # # normalize to pytorch grid coordinates
        # samples = self.normalize_sampled_pc(samples, item).transpose(0, 1)
        #
        # samples, transform = self.perform_augmentation(samples)
        #
        # # get the target: either the PC itself or the GT mesh
        # if self.mesh_as_target:
        #     target = self.normalize_mesh(o3d_to_pt3d_meshes([current_mesh]), item)
        #     if transform is not None:
        #         # augment the mesh
        #         mesh_verts, mesh_faces = target.get_mesh_verts_faces(0)
        #         target = Meshes([transform.transform_points(mesh_verts)], [mesh_faces])
        # else:
        #     target = samples

        samples = sample_points_from_meshes(current_mesh_pt3d, self.sample_points).squeeze(0).transpose(0, 1)
        target = current_mesh_pt3d if self.mesh_as_target else samples
        return samples, target, item

    def perform_augmentation(self, samples):
        # augmentation
        if self.do_augmentation:
            samples, transform = point_augmentation(samples)
            samples = samples.squeeze(0)

            # add point jitter
            samples += torch.randn_like(samples) * 0.005
        else:
            transform = None
        return samples, transform

    def normalize_sampled_pc(self, samples, index):
        return kpts_to_grid(samples, self.get_img_size(index)[::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_sampled_pc(self, samples, index):
        return kpts_to_world(samples, self.get_img_size(index)[::-1], align_corners=ALIGN_CORNERS)

    def normalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.normalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def unnormalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.unnormalize_sampled_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def get_batch_collate_fn(self):
        def mesh_collate_fn(list_of_samples):
            items = [item for _, _, item in list_of_samples]
            pcs = torch.stack([pc for pc, _, _ in list_of_samples], dim=0).to(self.device)
            meshes = join_meshes_as_batch([mesh for _, mesh, _ in list_of_samples]).to(self.device)

            # normalize PCs to pytorch grid coordinates
            for i, pc in enumerate(pcs):
                pcs[i] = self.normalize_sampled_pc(pc.transpose(0, 1), items[i]).transpose(0, 1)

            # data augmentation
            samples, transform = self.perform_augmentation(pcs)

            # normalize meshes
            meshes = join_meshes_as_batch([self.normalize_mesh(mesh.to(self.device), item) for _, mesh, item in list_of_samples])

            # transform mesh according to the augmentation
            if transform is not None:
                # augment the meshes (can only be performed per mesh, since meshes have different amount of vertices)
                meshes = Meshes([transform[i].transform_points(meshes.verts_list()[i]) for i in range(len(list_of_samples))], meshes.faces_list())

            return pcs, meshes

        def sample_collate_fn(list_of_samples):
            items = [item for _, _, item in list_of_samples]
            pcs = torch.stack([pc for pc, _, _ in list_of_samples], dim=0).to(self.device)

            # normalize PCs to pytorch grid coordinates
            for i, pc in enumerate(pcs):
                pcs[i] = self.normalize_sampled_pc(pc.transpose(0, 1), items[i]).transpose(0, 1)

            # data augmentation
            pcs_aug, transform = self.perform_augmentation(pcs)

            return pcs_aug, pcs_aug

        return mesh_collate_fn if self.mesh_as_target else sample_collate_fn

    def continuous_to_pat_index(self, item):
        return item // self.num_objects if self.fixed_object is None else item

    def continuous_to_obj_index(self, item):
        return item % self.num_objects if self.fixed_object is None else self.fixed_object

    def get_id(self, item):
        return self.ids[self.continuous_to_pat_index(item)]

    def get_img_size(self, item):
        return self.img_sizes[self.continuous_to_pat_index(item)]

    def get_obj_mesh(self, item):
        return self.meshes[self.continuous_to_pat_index(item)][self.continuous_to_obj_index(item)]


class PointToMeshDS(PointDataset):
    def __init__(self, sample_points, kp_mode, folder=POINT_DIR, image_folder=IMG_DIR, use_coords=True,
                 patch_feat=None, exclude_rhf=False, lobes=False, binary=False, do_augmentation=False, copd=False,
                 all_to_device='cpu'):
        super(PointToMeshDS, self).__init__(sample_points=sample_points, kp_mode=kp_mode, folder=folder,
                                            image_folder=image_folder,
                                            use_coords=use_coords, patch_feat=patch_feat, exclude_rhf=exclude_rhf,
                                            lobes=lobes, binary=binary, do_augmentation=do_augmentation, copd=copd,
                                            all_to_device=all_to_device)
        self.meshes = []
        self.img_sizes = []
        for case, sequence in self.ids:
            meshes = load_meshes(image_folder, case, sequence, obj_name='fissure' if not lobes else 'lobe')
            if not lobes and exclude_rhf:
                meshes = meshes[:2]
            self.meshes.append(meshes)

            size, spacing = load_image_metadata(os.path.join(image_folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

    def normalize_pc(self, samples, index):
        return kpts_to_grid(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_pc(self, samples, index):
        return kpts_to_world(samples, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def normalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.normalize_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())

    def unnormalize_mesh(self, mesh: Meshes, index):
        return Meshes([self.unnormalize_pc(m, index) for m in mesh.verts_list()], mesh.faces_list())


class PointToMeshAndLabelDataset(PointToMeshDS):
    def __init__(self, sample_points, kp_mode, folder=POINT_DIR, image_folder=IMG_DIR, use_coords=True,
                 patch_feat=None, exclude_rhf=False, lobes=False, binary=False, do_augmentation=True, copd=False):
        super().__init__(sample_points=sample_points, kp_mode=kp_mode, folder=folder,
                         image_folder=image_folder,
                         use_coords=use_coords, patch_feat=patch_feat, exclude_rhf=exclude_rhf,
                         lobes=lobes, binary=binary, do_augmentation=do_augmentation, copd=copd)

        # convert open3d meshes to pytorch3d and normalize vertices to grid coordinates
        self.meshes_pt3d = [self.normalize_mesh(o3d_to_pt3d_meshes(m), index=i) for i, m in enumerate(self.meshes)]

    def __getitem__(self, item, return_aug_transform=False):
        pts, lbls, transform = super().__getitem__(item, return_aug_transform=True)

        # get the meshes
        meshes = self.meshes_pt3d[item]
        if transform is not None:
            # transform the ground truth meshes to match augmented point cloud
            meshes = transform_meshes(meshes, transform)

        return pts, (lbls, meshes)

    def get_batch_collate_fn(self):
        def collate_fn(list_of_samples):
            pcs = []
            lbls = []
            meshes = []
            for pc, (lbl, mesh) in list_of_samples:
                pcs.append(pc)
                lbls.append(lbl)
                meshes.append(mesh)

            pcs = torch.stack(pcs, dim=0)
            lbls = torch.stack(lbls, dim=0)
            meshes = join_meshes_as_batch(meshes)
            return pcs, (lbls, meshes)

        return collate_fn


def load_landmarks(filepath):
    points = []
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            points.append([eval(coord) for coord in line])

    return torch.tensor(points)


def image2tensor(img: sitk.Image, dtype=None) -> torch.Tensor:
    array = sitk.GetArrayFromImage(img)
    if dtype == torch.bool:
        array = array.astype(bool)
    if array.dtype == np.uint16:
        array = array.astype(int)
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype)

    return tensor


def create_split(k: int, dataset: LungData, filepath: str, seed=42):
    # get the names of images that have fissures segmented
    names = np.asarray([img.split(os.sep)[-1].split('.')[0] for i, img in enumerate(dataset.images) if dataset.fissures[i] is not None])

    instances_per_test_set = len(names) // k
    val_set_sizes = (k) * [instances_per_test_set]

    # increment the test set sizes by one for the remainders
    remainder = len(names) - k * instances_per_test_set
    for i in range(remainder):
        val_set_sizes[i % k] += 1

    # random permutation
    np.random.seed(seed)
    permutation = np.random.permutation(len(names))

    # assemble test and val sets
    split = []
    test_start = 0
    for fold in range(k):
        # test dataset bounds
        test_end = test_start + val_set_sizes[fold]

        # index based on permutation
        train_names = np.concatenate([names[permutation[0:test_start]], names[permutation[test_end:]]])
        val_names = names[permutation[test_start:test_end]]

        split.append(collections.OrderedDict(
            {'train': np.sort(train_names),
             'val': np.sort(val_names)}
        ))

        test_start += val_set_sizes[fold]

    save_split_file(split, filepath)

    return split


def load_split_file(filepath):
    return np.load(filepath, allow_pickle=True)


def save_split_file(split, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(split, file)


if __name__ == '__main__':
    ds = PointToMeshAndLabelDataset(1024, 'cnn', patch_feat='image', exclude_rhf=False)
    exit()

    ds = CorrespondingPointDataset(1024, 'cnn', patch_feat='mind')
    splitfile = load_split_file('../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl')
    tr, vl = ds.split_data_set(splitfile[0])
    pass
    # ds = ImageDataset('../data')
    # splitfile = load_split_file('../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl')
    # for fold in splitfile:
    #     train, test = ds.split_data_set(fold)
