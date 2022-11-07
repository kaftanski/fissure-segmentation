import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import open3d as o3d
import torch

from cli.cl_args import get_seg_cnn_train_parser
from cli.cli_utils import load_args_for_testing, store_args, load_args_dict
from data import ImageDataset
from data_processing.surface_fitting import poisson_reconstruction
from metrics import batch_dice, binary_recall, binary_precision
from models.lraspp_3d import LRASPP_MobileNetv3_large_3d
from models.seg_cnn import MobileNetASPP
from preprocess_totalsegmentator_dataset import PROCESSED_DATA_PATH as TS_DATA_PATH
from train import run, write_results, compute_mesh_metrics
from utils.detached_run import maybe_run_detached_cli
from utils.fissure_utils import binary_to_fissure_segmentation
from utils.image_ops import write_image
from visualization import visualize_with_overlay


def test(ds: ImageDataset, device, out_dir, show):
    print('\nTESTING MODEL ...\n')

    args = load_args_dict(os.path.join(out_dir, '..'))  # go to cv-run level
    if args['model'] == 'v1':
        model_class = MobileNetASPP
    elif args['model'] == 'v3':
        model_class = LRASPP_MobileNetv3_large_3d
    else:
        raise NotImplementedError()

    model = model_class.load(os.path.join(out_dir, 'model.pth'), device=device)
    model.to(device)
    model.eval()

    # get the non-binarized labels from the dataset
    dataset_binary = ds.binary
    ds.binary = False

    # directory for output predictions
    pred_dir = os.path.join(out_dir, 'test_predictions')
    mesh_dir = os.path.join(pred_dir, 'meshes')
    label_dir = os.path.join(pred_dir, 'labelmaps')
    plot_dir = os.path.join(pred_dir, 'plots')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # compute all predictions
    all_pred_meshes = []
    all_targ_meshes = []
    ids = []
    test_dice = torch.zeros(len(ds), ds.num_classes)
    test_recall = torch.zeros(len(ds))
    test_precision = torch.zeros_like(test_recall)
    softmax_thresholds = torch.linspace(0, 1, steps=21)
    recall_per_threshold = torch.zeros(len(ds), len(softmax_thresholds))
    precision_per_threshold = torch.zeros_like(recall_per_threshold)
    for i in range(2):#range(len(ds)):
        case, sequence = ds.get_id(i)
        ids.append((case, sequence))
        # TODO: train with more dilation? for better recall
        img, label = ds[i]
        img, label = ds.get_batch_collate_fn()([(img, label)])
        with torch.no_grad():
            softmax_pred = model.predict_all_patches(img.to(device), min_overlap=0.5, use_gaussian=True)
        label_pred = torch.argmax(softmax_pred, dim=1)

        if model.num_classes == 2:  # binary prediction
            # reconstruct left/right fissure
            label_pred = binary_to_fissure_segmentation(label_pred.squeeze(), ds.get_left_right_lung_mask(i),
                                                        resample_spacing=ds.resample_spacing).unsqueeze(0)

        label = label.to(device)
        test_dice[i] += batch_dice(label_pred, label, n_labels=ds.num_classes).squeeze().cpu()
        print(case, sequence, 'DICE:', test_dice[i])

        # write prediction as image
        label_img = ds.get_fissures(i)
        label_pred_img = write_image(
            label_pred, filename=os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'),
            meta_src_img=label_img, undo_resample_spacing=ds.resample_spacing, interpolator=sitk.sitkNearestNeighbor)

        # visualize a slice
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'{case} {sequence} (3D CNN prediction)')
        visualize_with_overlay(img.squeeze()[:, img.shape[-2]//2], label_pred.squeeze()[:, img.shape[-2]//2],
                               title='Predicted fissure segmentation', ax=ax[0])

        # measure precision and recall at different softmax thresholds
        threshold_found = False
        for t, thresh in enumerate(softmax_thresholds):
            fissure_points = torch.zeros_like(label_pred, device=device)
            for lbl in range(1, model.num_classes):
                fissure_points = torch.logical_or(fissure_points, softmax_pred[:, lbl] > thresh)

            recall_per_threshold[i, t] = binary_recall(prediction=fissure_points, target=label).squeeze().cpu()
            precision_per_threshold[i, t] = binary_precision(prediction=fissure_points, target=label).squeeze().cpu()

            # use the lowest threshold where not all pixels are background
            if not threshold_found and not torch.all(fissure_points):
                print(f'Threshold for point cloud: {thresh.item()}')
                threshold_found = True

                # visualize the positive fissure points
                visualize_with_overlay(img.squeeze()[:, img.shape[-2]//2],
                                       fissure_points[0, :, fissure_points.shape[-2]//2],
                                       title=f'Fissure points thresholded at {thresh.item():.1f}', ax=ax[1])
                if show:
                    plt.show()

                # save the metrics and the image
                test_recall[i] = recall_per_threshold[i, t]
                test_precision[i] = precision_per_threshold[i, t]
                print(f'Recall: {test_recall[i].item():.4f}, Precision: {test_precision[i].item():.4f}')

                write_image(fissure_points.long(),
                            filename=os.path.join(label_dir, f'{case}_fissures_thresh_{sequence}.nii.gz'),
                            meta_src_img=label_img, undo_resample_spacing=ds.resample_spacing,
                            interpolator=sitk.sitkNearestNeighbor)

                # TODO: measure with non-dilated fissures
                # TODO: test-time aug mirroring (for more uncertainty)

        fig.savefig(os.path.join(plot_dir, f'{case}_fissures_pred_{sequence}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        # reconstruct meshes from predicted labelmap
        _, predicted_meshes = poisson_reconstruction(label_pred_img, ds.get_lung_mask(i))
        for j, m in enumerate(predicted_meshes):
            # save reconstructed mesh
            o3d.io.write_triangle_mesh(os.path.join(mesh_dir, f'{case}_fissure{j + 1}_{sequence}.obj'), m)

        # remember meshes for evaluation
        meshes_target = ds.get_fissure_meshes(i)[:2 if ds.exclude_rhf else 3]
        all_targ_meshes.append(meshes_target)
        all_pred_meshes.append(predicted_meshes)

    mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing = compute_mesh_metrics(
        all_pred_meshes, all_targ_meshes, ids=ids, show=show, plot_folder=plot_dir)

    # restore previous setting
    ds.binary = dataset_binary

    # compute average metrics
    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    print(f'Test dice per class: {mean_dice} +- {std_dice}')
    print(f'ASSD per fissure: {mean_assd} +- {std_assd}')
    print(f'Mean recall: {test_recall.mean()}')
    print(f'Mean precision: {test_precision.mean()}')

    # plot mean precision-recall curve
    mean_recall_per_threshold = recall_per_threshold.mean(0)
    mean_precision_per_threshold = precision_per_threshold.mean(0)
    print('Softmax Thresholds used:', softmax_thresholds)
    print('Mean recall per threshold:', mean_recall_per_threshold)
    print('Mean precision per threshold:', mean_precision_per_threshold)
    plt.figure()
    plt.plot(mean_recall_per_threshold, mean_precision_per_threshold)
    plt.title('Mean Precision-Recall Curve for Binary Fissure Points\n(measured at different softmax-thresholds)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(plot_dir, f'precision_recall.png'), bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing,
                  mean_recall=test_recall.mean(0, keepdim=True),
                  mean_precision=test_precision.mean(0, keepdim=True), softmax_thresholds=softmax_thresholds,
                  mean_recall_per_threshold=mean_recall_per_threshold,
                  mean_precision_per_threshold=mean_precision_per_threshold)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95, percent_missing


if __name__ == '__main__':
    parser = get_seg_cnn_train_parser()
    args = parser.parse_args()
    maybe_run_detached_cli(args)

    if args.test_only:
        args = load_args_for_testing(from_dir=args.output, current_args=args)

    if args.ds == 'data':
        img_dir = '../data'
    elif args.ds == 'ts':
        img_dir = TS_DATA_PATH
    else:
        raise ValueError(f'No dataset named {args.ds}')

    ds = ImageDataset(img_dir, exclude_rhf=args.exclude_rhf, binary=args.binary, patch_size=(args.patch_size,)*3,
                      resample_spacing=args.spacing)

    # load the desired model
    if args.model == 'v1':
        model_class = MobileNetASPP
    elif args.model == 'v3':
        model_class = LRASPP_MobileNetv3_large_3d
    else:
        raise NotImplementedError()
    model = model_class(num_classes=ds.num_classes, patch_size=(args.patch_size,)*3)

    # save setup
    if not args.test_only:
        store_args(args=args, out_dir=args.output)

    run(ds, model, test, args)
