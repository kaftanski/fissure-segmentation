import collections
import csv
import glob
import os.path
import pickle
import warnings
from abc import ABC, abstractmethod

import open3d as o3d
from copy import deepcopy
from glob import glob
from typing import OrderedDict

from augmentations import image_augmentation
from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor, multiple_objects_morphology, get_resample_factors
from utils.utils import load_points
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset


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

            case, _, sequence = img.split(os.sep)[-1].split('_')
            sequence = sequence.split('.')[0]
            self.ids.append((case, sequence))

            meshlist = sorted(glob(os.path.join(folder, f'{case}_mesh_{sequence}', f'{case}_fissure*_{sequence}.obj')))
            self.fissure_meshes.append(meshlist if meshlist else None)

            meshlist = sorted(glob(os.path.join(folder, f'{case}_mesh_{sequence}', f'{case}_lobe*_{sequence}.obj')))
            self.lobe_meshes.append(meshlist if meshlist else None)

        self.num_classes = 4

    def get_image(self, item):
        return _load_files_from_file_list(item, self.images)

    def get_fissures(self, item):
        return _load_files_from_file_list(item, self.fissures)

    def get_regularized_fissures(self, item):
        # this specifies what data should be considered "regularized"
        return _load_files_from_file_list(item, [f.replace('_fissures_', '_fissures_poisson_') if f is not None
                                                 else None for f in self.fissures])

    def get_landmarks(self, item):
        return _load_files_from_file_list(item, self.landmarks)

    def get_lung_mask(self, item):
        return _load_files_from_file_list(item, self.lung_masks)

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

    @abstractmethod
    def get_class_weights(self):
        pass

    @abstractmethod
    def get_batch_collate_fn(self):
        pass

    def split_data_set(self, split: OrderedDict[str, np.ndarray]):
        train_ds = deepcopy(self)
        val_ds = deepcopy(self)

        # check if nnUnet format is used or not
        nnu = ('_img_' not in split['train'][0])

        for i in range(len(self) - 1, -1, -1):
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
        img_array = (img_array - IMG_MIN) / (IMG_MAX - IMG_MIN) * 2 - 1

        return img_array.squeeze(), label_array.squeeze()  # TODO: return pat ids

    def get_class_weights(self):
        return None

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


def preprocessing_generator(dataloader, preproc_fn, **preproc_kwargs):
    for x_batch, y_batch in dataloader:
        yield preproc_fn(x_batch, y_batch, **preproc_kwargs)


def preprocess(img, label, device):
    img = (img.float().to(device) + IMG_MIN) / (IMG_MAX - IMG_MIN) * 2 - 1  # get inputs into range [-1, 1]
    label = label.long().to(device)
    return img, label


class PointDataset(CustomDataset):
    def __init__(self, sample_points, kp_mode, folder='/home/kaftan/FissureSegmentation/point_data/', use_coords=True,
                 patch_feat=None, exclude_rhf=False, lobes=False, binary=False):

        super(PointDataset, self).__init__(exclude_rhf=exclude_rhf, do_augmentation=False, binary=binary)

        if lobes and binary:
            raise NotImplementedError(
                'Binary prediction for lobe labels is not implemented. Use fissure data or remove the binary option.')

        assert patch_feat in [None, 'mind', 'mind_ssc']
        if not use_coords:
            assert patch_feat is not None, 'Neither Coords nor Features specified for PointDataset'

        files = sorted(glob(os.path.join(folder, kp_mode, '*_coords_*')))
        self.folder = os.path.join(folder, kp_mode)
        self.use_coords = use_coords
        self.lobes = lobes
        self.sample_points = sample_points
        self.ids = []
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
            x = torch.cat([pts, feat], dim=0)
        else:
            x = feat

        lbls = self.labels[item]
        sample = torch.randperm(x.shape[1])[:self.sample_points]

        if self.binary:
            return x[:, sample], (lbls[sample] != 0).long()
        else:
            return x[:, sample], lbls[sample]

    def __len__(self):
        return len(self.points)

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

    def get_batch_collate_fn(self):
        # point data needs no special collate function, the point sampling is already handled in __get_item__
        return None

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


class FaustDataset(Dataset):
    def __init__(self, sample_points, points_folder='/share/data_sam1/bigalke/datasets/faust/training/registrations/',
                 label_file='/share/data_sam1/bigalke/datasets/faust/training/body_part_labels.npy'):
        self.sample_points = sample_points

        # load point clouds
        filenames = glob(os.path.join(points_folder, '*.npy'))
        self.point_clouds = []
        for file in filenames:
            pts = torch.from_numpy(np.load(file)).T
            self.point_clouds.append(pts.float())

        # load label file (same for all point clouds, 1-to-1 correspondence)
        self.labels = torch.from_numpy(np.load(label_file)).long() - 1  # labels in range 0..14

        self.num_classes = len(torch.unique(self.labels))

    def __getitem__(self, item):
        # randomly sample points
        pts = self.point_clouds[item]
        sample = torch.randperm(pts.shape[1])[:self.sample_points]
        pts = pts[:, sample]
        lbl = self.labels[sample]
        return pts, torch.empty(0, pts.shape[1]), lbl

    def __len__(self):
        return len(self.point_clouds)


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
    ds = ImageDataset('../data')
    splitfile = load_split_file('../nnUNet_baseline/nnu_preprocessed/Task501_FissureCOPDEMPIRE/splits_final.pkl')
    for fold in splitfile:
        train, test = ds.split_data_set(fold)
