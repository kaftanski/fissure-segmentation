import csv
import os.path
from glob import glob

import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class LungData(Dataset):
    def __init__(self, folder):
        super(LungData, self).__init__()
        self.images = sorted(glob(os.path.join(folder, '*_img_*.nii.gz')))
        self.lung_masks = sorted(glob(os.path.join(folder, '*_mask_*.nii.gz')))
        self.landmarks = sorted(glob(os.path.join(folder, '*_lms_*.csv')))
        self.fissures = sorted(glob(os.path.join(folder, '*_fissures_*.nii.gz')))

        # fill missing landmarks and fissure segmentations with None
        for i, img in enumerate(self.images):
            lm_file = img.replace('_img_', '_lms_').replace('.nii.gz', '.csv')
            try:
                if lm_file != self.landmarks[i]:
                    self.landmarks.insert(i, None)
            except IndexError:
                self.landmarks.insert(i, None)

            fissure_file = img.replace('_img_', '_fissures_')
            try:
                if fissure_file != self.fissures[i]:
                    self.fissures.insert(i, None)
            except IndexError:
                self.fissures.insert(i, None)

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


def load_landmarks(filepath):
    points = []
    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            points.append([eval(coord) for coord in line])

    return torch.tensor(points)


if __name__ == '__main__':
    ds = LungData('/home/kaftan/FissureSegmentation/data/')
    res = ds[0]
    pass