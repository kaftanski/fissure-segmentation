import csv
import os

import torch

from cli.cl_args import get_seg_cnn_train_parser
from cli.cli_utils import load_args_for_testing
from constants import IMG_DIR_TS
from data import ImageDataset, LungData
from data_processing import foerstner
from data_processing.keypoint_extraction import MAX_KPTS
from train_segmentation_net import get_model_class
from utils.image_ops import resample_equal_spacing, sitk_image_to_tensor
from utils.utils import kpts_to_grid, sample_patches_at_kpts, ALIGN_CORNERS, no_print, new_dir
import SimpleITK as sitk


OUT_DIR = new_dir('results', 'preproc_timing')


def time_cnn_kp(cnn_dir, data_dir, device):
    default_parser = get_seg_cnn_train_parser()
    args, _ = default_parser.parse_known_args()
    args = load_args_for_testing(cnn_dir, args)

    ds = ImageDataset(folder=data_dir, do_augmentation=False, patch_size=(args.patch_size,) * 3,
                      resample_spacing=args.spacing)

    model_class = get_model_class(args)

    model = model_class.load(os.path.join(cnn_dir, f'fold0', 'model.pth'), device=device)
    model.eval()
    model.to(device)

    # set up time measurement
    all_inference_times = []
    all_feature_times = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for img_index in range(len(ds)):
        input_img = ds.get_batch_collate_fn()([ds[img_index]])[0].to(device)
        lung_mask = resample_equal_spacing(ds.get_lung_mask(img_index),
                                           ds.resample_spacing, use_nearest_neighbor=True)
        lung_mask = sitk_image_to_tensor(lung_mask).to(device)

        with no_print():
            with torch.no_grad():
                torch.cuda.synchronize()
                starter.record()
                softmax_pred = model.predict_all_patches(input_img)

            fissure_points = softmax_pred.argmax(1).squeeze() != 0

            # apply lung mask
            fissure_points = torch.logical_and(fissure_points, lung_mask)

            # nonzero voxels to points
            kp = torch.nonzero(fissure_points) * ds.resample_spacing

            # limit to subset
            if len(kp) > MAX_KPTS:
                perm = torch.randperm(len(kp), device=kp.device)[:MAX_KPTS]
                kp = kp[perm]

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

            kp_grid = kpts_to_grid(kp.flip(-1),
                                   shape=torch.tensor(fissure_points.shape) * ds.resample_spacing, align_corners=ALIGN_CORNERS)

            # compute feature time
            torch.cuda.synchronize()
            starter.record()
            features = sample_patches_at_kpts(softmax_pred[:, 1:].sum(1, keepdim=True), kp_grid, 5).squeeze().flatten(
                start_dim=1).transpose(0, 1)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            all_feature_times.append(curr_time)

        print(f'{ds.get_id(img_index)}: {all_inference_times[-1]:.4f} + {all_feature_times[-1]:.4f} s')

    write_times(os.path.join(OUT_DIR, 'cnn_kpts.csv'), all_inference_times, all_feature_times)


def time_foerstner_kp(data_dir, device):
    ds = LungData(folder=data_dir)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_inference_times = []
    for i in range(len(ds)):
        img = resample_equal_spacing(ds.get_image(i), target_spacing=1.)
        mask = resample_equal_spacing(ds.get_lung_mask(i), target_spacing=1.)

        img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0).float().to(device)
        mask_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask).astype(bool)).unsqueeze(0).unsqueeze(0).to(device)

        with no_print():
            torch.cuda.synchronize()
            starter.record()
            kp = foerstner.foerstner_kpts(img_tensor, mask=mask_tensor, sigma=0.5, thresh=1e-8, d=5)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

        print(f'{ds.get_id(i)}: {all_inference_times[-1]:.4f} s')

    write_times(os.path.join(OUT_DIR, 'foerstner_kpts.csv'), all_inference_times)


def write_times(out_filename, inference_times, feature_times=None):
    inference_times = torch.tensor(inference_times)

    if feature_times is not None:
        feature_times = torch.tensor(feature_times)
        total_times = inference_times + feature_times
    else:
        feature_times = torch.zeros_like(inference_times)
        total_times = inference_times

    with open(out_filename, 'w') as time_file:
        writer = csv.writer(time_file)
        writer.writerow(['Inference', 'Inference_std', 'Feature', 'Feature_std', 'Total', 'Total_std'])
        writer.writerow([inference_times.mean().item(), inference_times.std().item(),
                         feature_times.mean().item(), feature_times.std().item(),
                         total_times.mean().item(), total_times.std().item()])


if __name__ == '__main__':
    # time_cnn_kp('results/lraspp_recall_loss', IMG_DIR_TS, 'cuda:3')
    time_foerstner_kp(IMG_DIR_TS, 'cuda:3')
