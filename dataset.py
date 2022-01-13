import csv
import os.path
from glob import glob

import nibabel as nib
import torch
from torch.utils.data import Dataset


class LungData(Dataset):
    def __init__(self, folder):
        super(LungData, self).__init__()
        self.images = sorted(glob(os.path.join(folder, '*_img_*.nii.gz')))
        self.lung_masks = sorted(glob(os.path.join(folder, '*_mask_*.nii.gz')))
        self.landmarks = sorted(glob(os.path.join(folder, '*_lms_*.csv')))

        for i, img in enumerate(self.images):
            lm_file = img.replace('_img_', '_lms_').replace('.nii.gz', '.csv')
            if lm_file != self.landmarks[i]:
                self.landmarks.insert(i, None)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = [item]
        # elif isinstance(idx, slice):
        item = torch.arange(len(self))[item]

        imgs = []
        masks = []
        lms = []
        for i in item:
            img = nib.load(self.images[i])
            imgs.append(img)

            mask = nib.load(self.lung_masks[i])
            masks.append(mask)

            if self.landmarks[i] is not None:
                lm = load_landmarks(self.landmarks[i])
            else:
                lm = None

            lms.append(lm)

        result = {'img': imgs,
                  'mask': masks,
                  'lms': lms}

        return result

    def __len__(self):
        return len(self.images)


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