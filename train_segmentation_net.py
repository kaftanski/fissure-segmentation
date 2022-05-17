from cli.cl_args import get_seg_cnn_train_parser
from data import ImageDataset
from models.seg_cnn import MobileNetASPP
from train import run

if __name__ == '__main__':
    parser = get_seg_cnn_train_parser()
    args = parser.parse_args()

    ds = ImageDataset('../data', exclude_rhf=args.exclude_rhf)
    model = MobileNetASPP(num_classes=ds.num_classes)

    run(ds, model, args)
