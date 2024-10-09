import csv
import os

import SimpleITK as sitk
import torch

from cli.cli_args import get_seg_cnn_train_parser
from cli.cli_utils import load_args_for_testing
from constants import IMG_DIR_TS_PREPROC, KEYPOINT_CNN_DIR, ALIGN_CORNERS
from data import ImageDataset, LungData
from data_processing import foerstner
from data_processing.fissure_enhancement import load_fissure_stats, FISSURE_STATS_FILE, \
    HessianEnhancementFilter
from data_processing.keypoint_extraction import MAX_KPTS, limit_keypoints
from data_processing.point_features import mind
from models.lraspp_3d import LRASPP_MobileNetv3_large_3d
from utils.general_utils import kpts_to_grid, sample_patches_at_kpts, no_print, new_dir, topk_alldims
from utils.sitk_image_ops import resample_equal_spacing, sitk_image_to_tensor
from utils.pytorch_image_filters import smooth

OUT_DIR = new_dir('results', 'preproc_timing_node2')


def time_cnn_kp(cnn_dir=KEYPOINT_CNN_DIR, data_dir=IMG_DIR_TS_PREPROC, device='cuda:0'):
    default_parser = get_seg_cnn_train_parser()
    args, _ = default_parser.parse_known_args()
    args = load_args_for_testing(cnn_dir, args)

    ds = ImageDataset(folder=data_dir, do_augmentation=False, patch_size=(args.patch_size,) * 3,
                      resample_spacing=args.spacing)

    model = LRASPP_MobileNetv3_large_3d.load(os.path.join(cnn_dir, f'fold0', 'model.pth'), device=device)
    model.eval()
    model.to(device)

    # set up time measurement
    all_inference_times = []
    all_feature_times = []
    all_num_points = []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for img_index in range(len(ds)):
        input_img = ds.get_batch_collate_fn()([ds[img_index]])[0].to(device)
        lung_mask = resample_equal_spacing(ds.get_lung_mask(img_index),
                                           ds.resample_spacing, use_nearest_neighbor=True)
        lung_mask = sitk_image_to_tensor(lung_mask).to(device)

        with no_print():
            with torch.no_grad():
                torch.cuda.synchronize(device)
                starter.record(torch.cuda.current_stream(device))
                softmax_pred = model.predict_all_patches(input_img)

            fissure_points = softmax_pred.argmax(1).squeeze() != 0

            # apply lung mask
            fissure_points = torch.logical_and(fissure_points, lung_mask)

            # nonzero voxels to points
            kp = torch.nonzero(fissure_points) * ds.resample_spacing

            # limit to subset
            kp, _ = limit_keypoints(kp)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

            kp_grid = kpts_to_grid(kp.flip(-1),
                                   shape=torch.tensor(fissure_points.shape) * ds.resample_spacing, align_corners=ALIGN_CORNERS)

            # compute feature time
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))
            features = sample_patches_at_kpts(softmax_pred[:, 1:].sum(1, keepdim=True), kp_grid, 5).squeeze().flatten(
                start_dim=1).transpose(0, 1)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_feature_times.append(curr_time)

            all_num_points.append(len(kp))

        print(f'{ds.get_id(img_index)}: {all_inference_times[-1]:.4f} + {all_feature_times[-1]:.4f} s ({all_num_points[-1]} pts)')

    write_times(os.path.join(OUT_DIR, 'cnn_kpts.csv'), all_inference_times)
    write_times(os.path.join(OUT_DIR, 'patch_feat.csv'), all_feature_times, num_points=all_num_points)


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
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))
            kp = foerstner.foerstner_kpts(img_tensor, mask=mask_tensor, sigma=0.5, thresh=1e-8, d=5)

            # limit to MAX_KPTS
            kp, _ = limit_keypoints(kp)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times.append(curr_time)

        print(f'{ds.get_id(i)}: {all_inference_times[-1]:.4f} s')

    write_times(os.path.join(OUT_DIR, 'foerstner_kpts.csv'), all_inference_times)


def time_enhancement_kp(data_dir, device):
    # setup
    ds = LungData(folder=data_dir)
    fissure_mu, fissure_sigma = load_fissure_stats(FISSURE_STATS_FILE)

    hessian_filter = HessianEnhancementFilter(fissure_mu, fissure_sigma, gaussian_smoothing_sigma=1.,
                                              gaussian_derivation_sigma=1., show=False)
    hessian_filter.to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    all_inference_times = []

    for i in range(len(ds)):
        # load data
        img = ds.get_image(i)
        lung_mask = ds.get_lung_mask(i)

        # unit spacing
        img = resample_equal_spacing(img, target_spacing=1)
        lung_mask = resample_equal_spacing(lung_mask, target_spacing=1, use_nearest_neighbor=True)

        # images to tensor
        img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img))
        img_tensor = img_tensor.view(1, 1, *img_tensor.shape)
        img_tensor = img_tensor.float().to(device)

        inv_mask_tensor = sitk_image_to_tensor(lung_mask).view(img_tensor.shape).to(device).bool().logical_not()

        # start time
        with no_print():
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))
            # fissure enhancement filter
            fissures_enhanced = hessian_filter.predict_all_patches(img_tensor, min_overlap=0.25, patch_size=(64, 64, 64))

            # apply mask
            fissures_enhanced[inv_mask_tensor] = 0

            # extract keypoints
            fissures_enhanced = smooth(fissures_enhanced, 1.).squeeze()
            top_vals, top_idx = topk_alldims(fissures_enhanced, MAX_KPTS)
            top_idx = torch.stack(top_idx, dim=1)
            kp = top_idx[top_vals > 0.2]

        # stop time
        ender.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        curr_time = starter.elapsed_time(ender) / 1000
        all_inference_times.append(curr_time)

        print(f'{ds.get_id(i)}: {all_inference_times[-1]:.4f} s')

    write_times(os.path.join(OUT_DIR, 'enhancement_kpts.csv'), all_inference_times)


def time_mind_feat(data_dir, device):
    # setup
    ds = LungData(folder=data_dir)
    mind_sigma = 0.8
    delta = 1
    spacing = 1

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    all_inference_times_mind = []
    all_inference_times_ssc = []
    for i in range(len(ds)):
        img = ds.get_image(i)
        img = resample_equal_spacing(img, target_spacing=spacing)
        img_tensor = sitk_image_to_tensor(img).float().to(device)

        # measure mind feature computation
        with no_print():
            # MIND
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))

            mind(img_tensor.view(1, 1, *img_tensor.shape), sigma=mind_sigma, dilation=delta, ssc=False)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times_mind.append(curr_time)

            # MIND-SSC
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            starter.record(torch.cuda.current_stream(device))

            mind(img_tensor.view(1, 1, *img_tensor.shape), sigma=mind_sigma, dilation=delta, ssc=True)

            ender.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            curr_time = starter.elapsed_time(ender) / 1000
            all_inference_times_ssc.append(curr_time)

        print(f'{ds.get_id(i)}: [MIND] {all_inference_times_mind[-1]:.4f} s, [SSC] {all_inference_times_ssc[-1]:.4f} s')

    write_times(os.path.join(OUT_DIR, f'mind_feat_{str(spacing).replace(".",",")}mm.csv'), all_inference_times_mind)
    write_times(os.path.join(OUT_DIR, f'ssc_feat_{str(spacing).replace(".",",")}mm.csv'), all_inference_times_ssc)


def write_times(out_filename, inference_times, num_points=None):
    inference_times = torch.tensor(inference_times)
    if num_points is not None:
        num_points = torch.tensor(num_points).float()

    with open(out_filename, 'w') as time_file:
        writer = csv.writer(time_file)
        writer.writerow(['Inference', 'Inference_std']
                        + (['Num_Points', 'Num_Points_std'] if num_points is not None else []))
        writer.writerow([inference_times.mean().item(), inference_times.std().item()]
                        + ([num_points.mean().item(), num_points.std().item()] if num_points is not None else []))


if __name__ == '__main__':
    device = 'cuda:0'
    time_cnn_kp(KEYPOINT_CNN_DIR, IMG_DIR_TS_PREPROC, device)
    time_foerstner_kp(IMG_DIR_TS_PREPROC, device)
    time_enhancement_kp(IMG_DIR_TS_PREPROC, device)
    time_mind_feat(IMG_DIR_TS_PREPROC, device)
