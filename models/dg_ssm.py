import torch

from models.dgcnn import DGCNNReg
from models.modelio import LoadableModel, store_config_args
from shape_model.ssm import SSM


class DGSSM(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 ssm_alpha=3., ssm_targ_var=0.95):
        super(DGSSM, self).__init__()
        self.ssm = SSM(ssm_alpha, ssm_targ_var)
        self.dgcnn = DGCNNReg(k, in_features, 1,  # placeholder number of modes, has to be updated after training SSM
                              spatial_transformer, dynamic, image_feat_module)

    def forward(self, x):
        self.ssm.assert_trained()

        pred_weights = self.dgcnn(x)
        reconstructions = self.ssm.decode(pred_weights)
        return reconstructions

    def fit_ssm(self, shapes):
        self.ssm.fit(shapes)
