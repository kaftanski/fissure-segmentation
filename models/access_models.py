from models.dgcnn import DGCNNSeg
from models.point_net import PointNetSeg
from models.point_seg_net import PointSegmentationModelBase
from models.pointtransformer.seg_model import PointTransformerCompatibility


def get_point_seg_model_class(model_string) -> type(PointSegmentationModelBase):
    if model_string == 'DGCNN':
        return DGCNNSeg
    elif model_string == 'PointNet':
        return PointNetSeg
    elif model_string == 'PointTransformer':
        return PointTransformerCompatibility
    else:
        raise NotImplementedError()


def get_point_seg_model_class_from_args(args) -> type(PointSegmentationModelBase):
    if 'model' not in args:
        return DGCNNSeg

    return get_point_seg_model_class(args.model)
