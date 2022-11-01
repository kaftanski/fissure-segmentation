import collections
import csv
import glob
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
from pytorch3d.transforms import so3_log_map, Transform3d
from torch.utils.data import Dataset

from augmentations import image_augmentation, point_augmentation, compose_transform
from shape_model.ssm import load_shape
from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor, multiple_objects_morphology, \
    get_resample_factors, load_image_metadata
from utils.utils import load_points, ALIGN_CORNERS, kpts_to_grid, kpts_to_world, inverse_affine_transform, \
    decompose_similarity_transform

IMG_MIN = -1000
IMG_MAX = 1500


DEFAULT_SPLIT = "../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl"
DEFAULT_SPLIT_TS = "../nnUNet_baseline/nnu_preprocessed/Task503_FissuresTotalSeg/splits_final.pkl"


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
    def __init__(self, exclude_rhf, do_augmentation, binary):
        self.exclude_rhf = exclude_rhf
        self.do_augmentation = do_augmentation
        self.binary = binary
        try:
            getattr(self, 'ids')
        except AttributeError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    def get_class_weights(self):
        return None

    def get_batch_collate_fn(self):
        return None

    def split_data_set(self, split: OrderedDict[str, np.ndarray]):
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
                 do_augmentation=True, binary=False):
        LungData.__init__(self, folder)
        CustomDataset.__init__(self, exclude_rhf, do_augmentation, binary)

        self.resample_spacing = resample_spacing
        self.patch_size = patch_size

        # remove images without fissure label
        def remove_indices(ls: list, indices):
            for i in sorted(indices)[::-1]:
                ls.pop(i)

        to_remove = [i for i in range(len(self.fissures)) if self.fissures[i] is None]
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, list):
                remove_indices(attr, to_remove)

    @property
    def num_classes(self):
        if self.binary:
            return 2
        else:
            return 3 if self.exclude_rhf else 4

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

        return img_array.squeeze(), label_array.squeeze()  # TODO: return pat ids

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


def normalize_img(img, min_val=IMG_MIN, max_val=IMG_MAX):
    return (img - min_val) / (max_val - min_val) * 2 - 1  # get inputs into range [-1, 1]


class PointDataset(CustomDataset):
    def __init__(self, sample_points, kp_mode, folder='/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/point_data/', use_coords=True,
                 patch_feat=None, exclude_rhf=False, lobes=False, binary=False, do_augmentation=True):

        super(PointDataset, self).__init__(exclude_rhf=exclude_rhf, do_augmentation=do_augmentation, binary=binary)

        if lobes and binary:
            raise NotImplementedError(
                'Binary prediction for lobe labels is not implemented. Use fissure data or remove the binary option.')

        if not use_coords:
            raise ValueError('Coords have to be present for this to work...')
            # assert patch_feat is not None, 'Neither Coords nor Features specified for PointDataset'

        self.folder = os.path.join(folder, kp_mode)
        files = sorted(glob(os.path.join(self.folder, '*_coords_*')))
        self.kp_mode = kp_mode
        self.use_coords = use_coords
        self.lobes = lobes
        self.sample_points = sample_points
        self.points = []
        self.features = []
        self.labels = []
        for file in files:
            case, _, sequence = file.split('/')[-1].split('_')
            sequence = sequence.split('.')[0]
            pts, lbls, lobe_lbl, feat = load_points(self.folder, case, sequence, patch_feat)
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

    @property
    def num_classes(self):
        return 2 if self.binary else max(len(torch.unique(lbl)) for lbl in self.labels)

    def __getitem__(self, item):
        # randomly sample points
        feat = self.features[item]
        if self.use_coords:
            pts = self.points[item]
            if self.do_augmentation:
                # random transform of the coordinates
                pts, transform = point_augmentation(pts.unsqueeze(0))
                pts = pts.squeeze(0)

            x = torch.cat([pts, feat], dim=0)
        else:
            x = feat

        lbls = self.labels[item]
        sample = torch.randperm(x.shape[1])[:self.sample_points]

        if self.binary:
            return x[:, sample], (lbls[sample] != 0).long()
        else:
            return x[:, sample], lbls[sample]

    def get_class_weights(self):
        frequency = torch.zeros(self.num_classes)
        for lbl in self.labels:
            for c in range(self.num_classes):
                frequency[c] += torch.sum(lbl == c)
        frequency /= frequency.sum()
        print(f'Label frequency in point data set: {frequency.tolist()}')

        class_weights = 1 - frequency
        class_weights *= self.num_classes
        print(f'Class weights: {class_weights.tolist()}')

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


class CorrespondingPointDataset(PointDataset):
    def __init__(self, sample_points, kp_mode, point_folder='/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/point_data/',
                 use_coords=True, patch_feat=None, corr_folder="./results/corresponding_points",
                 image_folder='/share/data_rechenknecht03_2/students/kaftan/FissureSegmentation/data/',
                 do_augmentation=True, undo_affine_reg=False):
        super(CorrespondingPointDataset, self).__init__(sample_points, kp_mode, point_folder, use_coords, patch_feat,
                                                        exclude_rhf=True, do_augmentation=False)
        self.corr_points = CorrespondingPoints(corr_folder)
        self.do_augmentation_correspondingly = do_augmentation

        # # load meshes for supervision
        # self.meshes = []
        # for case, sequence in self.ids:
        #     meshes = load_meshes(image_folder, case, sequence, obj_name='fissure')
        #     self.meshes.append(meshes)

        # remove non-matched data points
        self._remove_non_matched_from_corr_points()

        # load img sizes for pytorch grid normalization
        self.img_sizes = []
        for case, sequence in self.ids:
            size, spacing = load_image_metadata(os.path.join(image_folder, f"{case}_img_{sequence}.nii.gz"))
            size_world = tuple(sz * sp for sz, sp in zip(size, spacing))
            self.img_sizes.append(size_world)

        # # affine transform the ground truth meshes to fit the corresponding points
        # for meshes, transforms in zip(self.meshes, self.corr_points.transforms):
        #     for m, t in zip(meshes, transforms):
        #         # rotate around [0,0,0] (o3d uses the PC's center of mass by default)
        #         m.rotate(t[..., :3], center=np.zeros([3, 1]))
        #         m.translate(t[..., -1])

    def __getitem__(self, item):
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
        if self.do_augmentation_correspondingly:
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

    def split_data_set(self, split: OrderedDict[str, np.ndarray]):
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
        return kpts_to_grid(pc, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS, return_transform=return_transform)

    def unnormalize_pc(self, pc, index):
        return kpts_to_world(pc, self.img_sizes[index][::-1], align_corners=ALIGN_CORNERS)

    def unnormalize_mean_pc(self, pc):
        mean_size = torch.stack([torch.tensor(self.img_sizes[i][::-1]) for i in range(len(self))], dim=0).mean(0)
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
    ds = CorrespondingPointDataset(1024, 'cnn', patch_feat='mind')
    splitfile = load_split_file('../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl')
    tr, vl = ds.split_data_set(splitfile[0])
    pass
    # ds = ImageDataset('../data')
    # splitfile = load_split_file('../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl')
    # for fold in splitfile:
    #     train, test = ds.split_data_set(fold)
