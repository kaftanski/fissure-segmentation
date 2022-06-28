import time

import torch
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from torch import nn
from data_processing.point_features import mind
from torch.nn import functional as F
import SimpleITK as sitk

from utils.image_ops import tensor_to_sitk_image, sitk_image_to_tensor, resample_equal_spacing, write_image

GRID_SP = 2


def get_data(fixed_file, moving_file, fixed_mask_file, moving_mask_file, device):
    spacing = 1

    img_fix = sitk_image_to_tensor(
        resample_equal_spacing(sitk.ReadImage(fixed_file), spacing)).float().unsqueeze(0).unsqueeze(0)
    img_mov = sitk_image_to_tensor(
        resample_equal_spacing(sitk.ReadImage(moving_file), spacing)).float().unsqueeze(0).unsqueeze(0)

    mask_fix = sitk_image_to_tensor(
        resample_equal_spacing(
            sitk.ReadImage(fixed_mask_file), spacing, use_nearest_neighbor=True)
    ).float().unsqueeze(0).unsqueeze(0)

    mask_mov = sitk_image_to_tensor(
        resample_equal_spacing(
            sitk.ReadImage(moving_mask_file), spacing, use_nearest_neighbor=True)
    ).float().unsqueeze(0).unsqueeze(0)

    img_fix += 1000  # important that scans are in HU
    img_mov += 1000

    # compute MIND descriptors and downsample (using average pooling)
    with torch.no_grad():
        #with torch.cuda.amp.autocast():
        mind_fix_ = mask_fix.to(device) * mind(img_fix.to(device), ssc=True)
        mind_mov_ = mask_mov.to(device) * mind(img_mov.to(device), ssc=True)
        mind_fix = F.avg_pool3d(mind_fix_, GRID_SP, stride=GRID_SP)
        mind_mov = F.avg_pool3d(mind_mov_, GRID_SP, stride=GRID_SP)

    # ch = 0
    # plt.figure()
    # plt.imshow(mind_fix[0, ch, :, mind_fix.shape[3]//2].cpu())
    # plt.title(f'MIND fixed channel {ch}')
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(mind_mov[0, ch, :, mind_mov.shape[3]//2].cpu())
    # plt.title(f'MIND moving channel {ch}')
    # plt.colorbar()
    #
    # plt.show()

    return mind_fix, mind_mov, img_fix, img_mov, mask_fix, mask_mov


def adam_registration(fixed_file, moving_file, fixed_mask_file, moving_mask_file, warped_file, disp_file, case_number, device):
    grid_sp = 2

    mind_fix, mind_mov, img_fix, img_mov, mask_fix, mask_mov = get_data(
        fixed_file, moving_file, fixed_mask_file, moving_mask_file, device)

    # generate random keypoints (TODO: why?)
    keypts_rand = 2 * torch.rand(2048 * 24, 3).to(device) - 1
    val = F.grid_sample(mask_fix.to(device), keypts_rand.view(1, -1, 1, 1, 3), align_corners=False)
    idx1 = torch.nonzero(val.squeeze() == 1).reshape(-1)
    keypts_fix = keypts_rand[idx1[:1024 * 2]]

    H, W, D = img_fix.shape[2:]
    t0 = time.time()

    mind_fix = mind_fix.to(device).half()
    mind_mov = mind_mov.to(device).half()

    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), (1, 1, H // grid_sp, W // grid_sp, D // grid_sp),
                          align_corners=False)
    reg_net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False))
    reg_net[0].weight.data[:] = torch.clone(grid0.permute(0, 4, 1, 2, 3)).float().cpu().data
    reg_net.to(device)
    optimizer = torch.optim.Adam(reg_net.parameters(), lr=1)

    # run Adam optimisation with diffusion regularisation and B-spline smoothing
    lambda_weight = .65  # with tps: .5, without:0.7
    for iter in range(50):  # 80
        optimizer.zero_grad()
        disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(reg_net[0].weight, 3, stride=1, padding=1),
                                                3, stride=1, padding=1), 3, stride=1, padding=1).permute(0, 2, 3, 4, 1)
        reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                   lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                   lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        scale = torch.tensor([(H // grid_sp - 1) / 2, (W // grid_sp - 1) / 2, (D // grid_sp - 1) / 2]).to(
            device).unsqueeze(0)
        grid_disp = grid0.view(-1, 3).to(device).float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        patch_mov_sampled = F.grid_sample(mind_mov.float(),
                                          grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).to(device),
                                          align_corners=False, mode='bilinear')  # ,padding_mode='border')
        sampled_cost = (patch_mov_sampled - mind_fix).pow(2).mean(1) * 12
        loss = sampled_cost.mean()
        (loss + reg_loss).backward()
        optimizer.step()

        # TODO: segmentation loss

    fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
    disp_hr = F.interpolate(fitted_grid * grid_sp, size=(H, W, D), mode='trilinear', align_corners=False)

    disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr, 3, padding=1, stride=1),
                                            3, padding=1, stride=1), 3, padding=1, stride=1)

    disp_hr = torch.flip(disp_smooth / torch.tensor([H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1).to(device) * 2, [1])

    pred_xyz = F.grid_sample(disp_hr.float(), keypts_fix.to(device).view(1, -1, 1, 1, 3), mode='bilinear',
                             align_corners=False).squeeze().t()
    torch.cuda.synchronize()
    t1 = time.time()
    t_adam = t1 - t0

    print(f'Adam run time: {t_adam:.4f} sec')

    if disp_file is not None:
        write_image(fitted_grid.squeeze().permute(1, 2, 3, 0), disp_file)

    # compute_tre if possible
    if case_number is not None:
        i = int(case_number) - 1
        copd_lms = torch.load('copd_converted_lms.pth')

        disp = F.grid_sample(disp_hr.cpu(), copd_lms['lm_copd_exh'][i].view(1, -1, 1, 1, 3),
                             align_corners=False).squeeze().t()
        tre_before = ((copd_lms['lm_copd_exh'][i] - copd_lms['lm_copd_insp'][i]) * (
                torch.tensor([208 / 2, 192 / 2, 192 / 2]).view(1, 3) * torch.tensor([1.25, 1., 1.75]))).pow(2).sum(
            1).sqrt()

        tre_after = ((copd_lms['lm_copd_exh'][i] - copd_lms['lm_copd_insp'][i] + disp) * (
                torch.tensor([208 / 2, 192 / 2, 192 / 2]).view(1, 3) * torch.tensor([1.25, 1., 1.75]))).pow(2).sum(
            1).sqrt()
        print('DIRlab COPD #', case_number, 'TRE before (mm)', '%0.3f' % (tre_before.mean().item()),
              'TRE after (mm)', '%0.3f' % (tre_after.mean().item()))
    warped = F.grid_sample(img_mov.view(1, 1, H, W, D), disp_hr.cpu().permute(0, 2, 3, 4, 1) +
                           F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, H, W, D), align_corners=False),
                           align_corners=False).cpu()
    if warped_file is not None:
        sitk.WriteImage(tensor_to_sitk_image(warped - 1000), warped_file)  # TODO: figure out spacing
        # nib.save(nib.Nifti1Image((warped - 1000).data.squeeze().numpy(), np.diag([1.75, 1.25, 1.75, 1])), warped_file)

    if (warped_file is None) & (disp_file is None):
        img_fix *= mask_fix.view_as(img_fix)
        warped *= mask_fix.view_as(warped)

        plt.imshow(torch.clamp(img_fix.squeeze().cpu()[:, 110], 0, 700).div(700).pow(1.25).t().flip(0), 'Blues')
        plt.imshow(torch.clamp(warped.squeeze().squeeze()[:, 110], 0, 500).div(500).pow(1.25).t().flip(0), 'Oranges',
                   alpha=.5)
        plt.axis('off')
        plt.savefig('voxelmorph_plusplus_warped.png')
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-F', '--fixed_file', required=True, help="fixed scan (exhale) nii.gz")
    parser.add_argument('-M', '--moving_file', required=True, help="moving scan (inspiration) nii.gz")
    parser.add_argument('-f', '--fixed_mask_file', required=True, help="mask fixed nii.gz")
    parser.add_argument('-m', '--moving_mask_file', required=True, help="mask moving nii.gz")

    parser.add_argument('-w', '--warped_file', required=False, help="output nii.gz file")
    parser.add_argument('-d', '--disp_file', required=False, help="output displacements pth-file")
    parser.add_argument('-c', '--case_number', required=False, help="DIRlab COPD case number (for TRE)")
    parser.add_argument('-D', '--device', required=False, help="pytorch device to compute on", default="cuda:3")

    adam_registration(**vars(parser.parse_args()))
