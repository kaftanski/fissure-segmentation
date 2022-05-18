import os
import SimpleITK as sitk
import torch

from cli.cl_args import get_seg_cnn_train_parser
from data import ImageDataset
from image_ops import write_image, tensor_to_sitk_image
from models.seg_cnn import MobileNetASPP
from train import run, batch_dice, compute_mesh_metrics, write_results


def test(ds, device, out_dir, show):
    print('\nTESTING MODEL ...\n')

    model = MobileNetASPP.load(os.path.join(out_dir, 'checkpoints', '49.pth'), device=device)
    model.to(device)
    model.eval()

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

    for i in range(len(ds)):
        case, sequence = ds.get_id(i)

        img, label = ds[i]
        img, label = ds.get_batch_collate_fn()([(img, label)])
        with torch.no_grad():
            softmax_pred = model.predict_all_patches(img.to(device), patch_size=(128, 128, 128), min_overlap=0.25)
        label_pred = torch.argmax(softmax_pred, dim=1)
        test_dice[i] += batch_dice(label_pred, label.to(device), n_labels=ds.num_classes).squeeze().cpu()
        print(case, sequence, 'DICE:', test_dice[i])

        # write prediction as image
        label_img = ds.get_fissures(i)
        label_pred_img = tensor_to_sitk_image(label_pred)
        label_pred_img.SetSpacing((ds.resample_spacing,)*3)
        label_pred_img = sitk.Resample(label_pred_img, referenceImage=label_img, interpolator=sitk.sitkNearestNeighbor)
        sitk.WriteImage(label_pred_img, os.path.join(label_dir, f'{case}_fissures_pred_{sequence}.nii.gz'))

    # compute average metrics
    mean_dice = test_dice.mean(0)
    std_dice = test_dice.std(0)

    mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95 = compute_mesh_metrics([[]], [[]])

    print(f'Test dice per class: {mean_dice} +- {std_dice}')
    print(f'ASSD per fissure: {mean_assd} +- {std_assd}')

    # output file
    write_results(os.path.join(out_dir, 'test_results.csv'), mean_dice, std_dice, mean_assd, std_assd, mean_sdsd,
                  std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95)

    return mean_dice, std_dice, mean_assd, std_assd, mean_sdsd, std_sdsd, mean_hd, std_hd, mean_hd95, std_hd95


if __name__ == '__main__':
    parser = get_seg_cnn_train_parser()
    args = parser.parse_args()

    ds = ImageDataset('../data', exclude_rhf=args.exclude_rhf)
    model = MobileNetASPP(num_classes=ds.num_classes)

    run(ds, model, test, args)

    # TODO: loss function L_CE+DICE
