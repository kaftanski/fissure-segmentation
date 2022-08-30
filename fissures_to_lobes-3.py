import os
from glob import glob

import cc3d
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


def assign_nearest(labels_out, labels_in, labels_in_new):
    xyz_surf = torch.empty(0, 3).cuda()
    label_surf = torch.empty(0).cuda()

    # newly added candidates (region growing)
    idx_new = (labels_in == 0) & (labels_in_new == 1)
    xyz_new = torch.nonzero(idx_new)

    # extract surfaces
    for i in range(1, 5):
        surface = F.conv3d((labels_out == i).float().unsqueeze(0).unsqueeze(0).cuda(),
                           torch.tensor([-.5, 0, .5]).cuda().view(1, 1, 3, 1, 1), padding=(1, 0, 0)).abs()
        surface += F.conv3d((labels_out == i).float().unsqueeze(0).unsqueeze(0).cuda(),
                            torch.tensor([-.5, 0, .5]).cuda().view(1, 1, 1, 3, 1), padding=(0, 1, 0)).abs()
        surface += F.conv3d((labels_out == i).float().unsqueeze(0).unsqueeze(0).cuda(),
                            torch.tensor([-.5, 0, .5]).cuda().view(1, 1, 1, 1, 3), padding=(0, 0, 1)).abs()
        xyz_surfi = torch.nonzero(surface.squeeze())
        xyz_surfi = xyz_surfi[torch.randperm(xyz_surfi.shape[0])[:xyz_surfi.shape[0] // 3], :]
        xyz_surf = torch.cat((xyz_surf, xyz_surfi), 0)

        label_surf = torch.cat((label_surf, torch.ones_like(xyz_surfi[:, 0]) * i), 0)

    # pdist2 in chunks to assign the nearest label
    # print('pdist2',xyz_surf.shape,xyz_new.shape)
    chk = torch.chunk(xyz_surf.cuda(), 256, 0)
    chk_label = torch.chunk(label_surf.cuda(), 256)
    with torch.no_grad():
        nearest = torch.ones_like(xyz_new) * 1e5
        for i in range(len(chk)):
            with torch.cuda.amp.autocast():
                dist_min = (chk[i].unsqueeze(1) - xyz_new.unsqueeze(0)).pow(2).sum(-1).sqrt().min(0)
                better = dist_min[0] < nearest[:, 0]
                nearest[better, 0] = dist_min[0][better]
                best_label = (chk_label[i][dist_min[1]]).squeeze()
                nearest[better, 1] = best_label[better]

    labels_out_new = labels_out.clone().detach().cuda().long()
    labels_out_new.reshape(-1)[idx_new.reshape(-1)] = nearest[:, 1].long()
    return labels_out_new


def fissures_to_lobes(fissure_file: str, out_folder: str, spacing=1.25, exclude_rhf=True):
    """ Converts fissure segmentation to lobe segmentation (by Mattias Heinrich)

    :param fissure_file: fissure segmentation
    :param out_folder: output will land in this folder
    """
    gpu_id = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print(torch.cuda.get_device_name())

    print("processing case", fissure_file)

    nii_img = nib.load(fissure_file)
    fissures = torch.from_numpy(nii_img.get_fdata()).long()
    if exclude_rhf:
        fissures[fissures == 3] = 0
    print(fissures.unique(), fissures.shape)
    H, W, D = fissures.shape
    fissures_header = nii_img.header

    mask = torch.from_numpy(nib.load(fissure_file.replace('_fissures_poisson_', '_mask_')).get_fdata()).long()#.flip(-1)
    print(mask.unique(), mask.shape)

    # make isotropic
    pixdim = fissures_header['pixdim']
    iso_dim = tuple(int(s * p / spacing + 0.5) for s, p in zip(mask.shape, pixdim[1:4]))
    print(iso_dim)
    fissure1 = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(
        F.interpolate(F.one_hot(fissures.cuda(), 3).float().unsqueeze(0).permute(0, 4, 1, 2, 3),
                      size=iso_dim, mode='trilinear'),
        5, stride=1, padding=2), 5, stride=1, padding=2), 5, stride=1, padding=2), 5, stride=1, padding=2),
        5, stride=1, padding=2)

    mask1 = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(
        F.interpolate(F.one_hot(mask.cuda(), 2).float().unsqueeze(0).permute(0, 4, 1, 2, 3),
                      size=iso_dim, mode='trilinear'),
        5, stride=1, padding=2), 5, stride=1, padding=2), 5, stride=1, padding=2), 5, stride=1, padding=2),
        5, stride=1, padding=2)

    y = 120

    ratios_f = torch.tensor([0.000001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])
    ratios_m = torch.tensor([0.000001, 0.00001, 0.00003, 0.001, 0.005, 0.02, 0.1, .5, 1])

    i = 0
    fissure1_ = fissure1 * torch.tensor([ratios_f[i], 1, 1]).cuda().view(1, 3, 1, 1, 1)
    mask1_ = mask1 * torch.tensor([1, ratios_m[i]]).cuda().view(1, 2, 1, 1, 1)

    labels_in = mask1_.argmax(1).squeeze()
    labels_in[fissure1_.argmax(1).squeeze() > 0] = 0
    labels_out = torch.from_numpy(
        cc3d.connected_components(labels_in.cpu().numpy()).astype('int32')).cuda()  # 26-connected
    print('labels_out.unique', labels_out.unique())

    assert len(labels_out.unique()) == 5, "must be four+1 components for lobes"

    for i in trange(1, len(ratios_f)):
        fissure1_ = fissure1 * torch.tensor([ratios_f[i], 1, 1]).cuda().view(1, 3, 1, 1, 1)
        mask1_ = mask1 * torch.tensor([1, ratios_m[i]]).cuda().view(1, 2, 1, 1, 1)

        labels_in_new = mask1_.argmax(1).squeeze()
        labels_in_new[fissure1_.argmax(1).squeeze() > 0] = 0

        labels_out_new = assign_nearest(labels_out, labels_in, labels_in_new)

        labels_in = labels_in_new
        labels_out = labels_out_new

    lobes_new = F.interpolate(F.one_hot(labels_out_new.squeeze().cuda(), 5).float().unsqueeze(0).permute(0, 4, 1, 2, 3),
                              size=(H, W, D), mode='trilinear').argmax(1).squeeze().cpu()

    nib.save(nib.Nifti1Image(lobes_new.short().numpy(), np.diag([pixdim[1], pixdim[2], pixdim[3], 1]),
                             header=fissures_header),
             os.path.join(out_folder, os.path.split(fissure_file)[1].replace('_fissures_poisson_', '_lobes_')))


if __name__ == '__main__':
    data_path = '../data'
    out_path = 'results/fissures_to_lobes_mattias'
    os.makedirs(out_path, exist_ok=True)
    fissures = sorted(glob(os.path.join(data_path, 'EMPIRE*_fissures_poisson_*')))[21:22]
    for f in fissures:
        torch.cuda.empty_cache()
        fissures_to_lobes(f, out_path)
