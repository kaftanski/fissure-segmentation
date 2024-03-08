from models.dgcnn import DGCNNSeg
from models.point_net import PointNetSeg


def get_point_seg_model_class(model_string):
    if model_string == 'DGCNN':
        return DGCNNSeg
    elif model_string == 'PointNet':
        return PointNetSeg
    else:
        raise NotImplementedError()


def get_point_seg_model_class_from_args(args):
    if 'model' not in args:
        return DGCNNSeg

    return get_point_seg_model_class(args.model)
