import os

import SimpleITK as sitk
import torch

from data_processing.find_lobes import lobes_to_fissures
from data import LungData
from metrics import label_label_assd
from train import write_results, batch_dice

ds = LungData('../data')

folder = 'results/test_lobes_to_fissures'
os.makedirs(folder, exist_ok=True)

all_dice = []

all_assd = []
all_sdsd = []
all_hd = []
all_hd95 = []
for i in range(len(ds)):
    case, sequence = ds.get_id(i)
    print(f'\n{", ".join(ds.get_id(i))}')
    lobes = ds.get_lobes(i)
    if lobes is None:
        print('No lobes segmented, skipping ...')
        continue

    fissures_reconstructed, lobes_filled_img = lobes_to_fissures(lobes, ds.get_lung_mask(i))

    sitk.WriteImage(lobes_filled_img, os.path.join(folder, f'{case}_lobes_filled_{sequence}.nii.gz'))
    sitk.WriteImage(fissures_reconstructed, os.path.join(folder, f'{case}_fissures_from_lobes_{sequence}.nii.gz'))

    f_rec = torch.from_numpy(sitk.GetArrayFromImage(fissures_reconstructed))
    f_gt = torch.from_numpy(sitk.GetArrayFromImage(ds.get_regularized_fissures(i)))

    dice = batch_dice(f_rec.unsqueeze(0), f_gt.unsqueeze(0), n_labels=3)
    all_dice.append(dice)

    all_assd.append([])
    all_sdsd.append([])
    all_hd.append([])
    all_hd95.append([])
    for f in f_rec.unique()[1:]:
        mean, std, hd, hd95 = label_label_assd(f_rec == f, f_gt == f,
                                               fissures_reconstructed.GetSpacing())
        all_assd[-1].append(mean)
        all_sdsd[-1].append(std)
        all_hd[-1].append(hd)
        all_hd95[-1].append(hd95)

write_results(os.path.join(folder, 'lobes_to_fissures_error.csv'),
              torch.stack(all_dice, dim=0).mean(0), torch.stack(all_dice, dim=0).std(0),
              torch.tensor(all_assd).mean(0), torch.tensor(all_assd).std(0),
              torch.tensor(all_sdsd).mean(0), torch.tensor(all_sdsd).std(0),
              torch.tensor(all_hd).mean(0), torch.tensor(all_hd).std(0),
              torch.tensor(all_hd95).mean(0), torch.tensor(all_hd95).std(0))
