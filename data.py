import collections
import csv
import glob
import os.path
import pickle
from copy import deepcopy
from glob import glob
from typing import OrderedDict

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_points


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
        return pts[:, sample], self.labels[sample]

    def __len__(self):
        return len(self.point_clouds)


class LungData(Dataset):
    def __init__(self, folder):
        super(LungData, self).__init__()
        self.images = sorted(glob(os.path.join(folder, '*_img_*.nii.gz')))
        self.lung_masks = sorted(glob(os.path.join(folder, '*_mask_*.nii.gz')))
        self.landmarks = []
        self.fissures = []
        self.lobescribbles = []

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

            lobes_file = img.replace('_img_', '_lobescribbles_')
            if os.path.exists(lobes_file):
                self.lobescribbles.append(lobes_file)
            else:
                self.lobescribbles.append(None)

        self.num_classes = 4

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]

        item = torch.arange(len(self))[item]

        imgs = []
        fissures = []
        for i in item:
            # load the i-th image
            img = sitk.ReadImage(self.images[i])
            imgs.append(img)

            # load the ground truth fissure segmentation, if available
            if self.fissures[i] is not None:
                fissure = sitk.ReadImage(self.fissures[i])
            else:
                fissure = None
            fissures.append(fissure)

        if len(item) == 1:
            imgs = imgs[0]
            fissures = fissures[0]

        return imgs, fissures

    def __len__(self):
        return len(self.images)

    def get_landmarks(self, item):
        if isinstance(item, int):
            item = [item]

        item = torch.arange(len(self))[item]

        lms = []
        for i in item:
            # load the reference landmarks, if available
            if self.landmarks[i] is not None:
                lm = load_landmarks(self.landmarks[i])
            else:
                lm = None
            lms.append(lm)

        if len(item) == 1:
            lms = lms[0]

        return lms

    def get_lung_mask(self, item):
        if isinstance(item, int):
            item = [item]

        item = torch.arange(len(self))[item]

        masks = []
        for i in item:
            # load the lung mask
            mask = sitk.ReadImage(self.lung_masks[i])
            masks.append(mask)

        if len(item) == 1:
            masks = masks[0]

        return masks

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

    def get_lobescribbles(self, item):
        if isinstance(item, int):
            item = [item]

        item = torch.arange(len(self))[item]

        lobes = []
        for i in item:
            # load the lobe scribbles
            if self.lobescribbles[i] is not None:
                lobe = sitk.ReadImage(self.lobescribbles[i])
            else:
                lobe = None
            lobes.append(lobe)

        if len(item) == 1:
            lobes = lobes[0]

        return lobes


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


class PointDataset(Dataset):
    def __init__(self, sample_points, folder='/home/kaftan/FissureSegmentation/point_data/', exclude_rhf=False):
        files = sorted(glob(os.path.join(folder, '*_points_*')))
        self.folder = folder
        self.exclude_rhf = exclude_rhf
        self.sample_points = sample_points
        self.ids = []
        self.points = []
        self.labels = []
        for file in files:
            case, _, sequence = file.split('/')[-1].split('_')
            sequence = sequence.split('.')[0]
            pts, lbls = load_points(folder, case, sequence)
            if exclude_rhf:
                lbls[lbls == 3] = 0
            self.points.append(pts)
            self.labels.append(lbls)
            self.ids.append((case, sequence))

        self.num_classes = max(len(torch.unique(lbl)) for lbl in self.labels)

    def __getitem__(self, item):
        # randomly sample points
        pts = self.points[item]
        lbls = self.labels[item]
        sample = torch.randperm(pts.shape[1])[:self.sample_points]
        return pts[:, sample], lbls[sample]

    def __len__(self):
        return len(self.points)

    def get_label_frequency(self):
        frequency = torch.zeros(self.num_classes)
        for lbl in self.labels:
            for c in range(self.num_classes):
                frequency[c] += torch.sum(lbl == c)
        frequency /= frequency.sum()
        print(f'Label frequency in point data set: {frequency.tolist()}')
        return frequency

    def split_data_set(self, split: OrderedDict[str, np.ndarray]):
        train_ds = deepcopy(self)
        val_ds = deepcopy(self)

        for i in range(len(self) - 1, -1, -1):
            case, sequence = self.ids[i]
            id = case + '_img_' + sequence
            if id in split['val']:
                train_ds.points.pop(i)
                train_ds.labels.pop(i)
                train_ds.ids.pop(i)
            else:
                assert id in split['train'], f'Train/Validation split incomplete: instance {id} is contained in neither.'
                val_ds.points.pop(i)
                val_ds.labels.pop(i)
                val_ds.ids.pop(i)

        return train_ds, val_ds


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
    ds = LungData('/home/kaftan/FissureSegmentation/data/')
    # res = ds[0]
    split = create_split(5, ds, '../data/split.np.pkl')
    split_ld = load_split_file('../data/split.np.pkl')
