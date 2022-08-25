from types import SimpleNamespace

import torch
from torch import nn

from models.dgcnn_opensrc import DGCNN
from models.modelio import LoadableModel, store_config_args
from models.utils import init_weights
from shape_model.ssm import SSM, LSSM


class DGSSM(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 ssm_alpha=3., ssm_targ_var=0.95, ssm_modes=1, lssm=False):
        super(DGSSM, self).__init__()
        SSMClass = SSM if not lssm else LSSM
        self.ssm = SSMClass(ssm_alpha, ssm_targ_var)
        # self.dgcnn = DGCNNReg(k, in_features, dgcnn_out_features,  # placeholder number of modes, has to be updated after training SSM
        #                       spatial_transformer, dynamic, image_feat_module)
        dgcnn_args = SimpleNamespace(
            k=k,
            emb_dims=1024,  # length of global feature vector
            dropout=0.
        )
        self.dgcnn = DGCNN(dgcnn_args, output_channels=ssm_modes)

    def forward(self, x):
        self.ssm.assert_trained()

        coefficients = self.dgcnn(x)  # predict the coefficient multipliers for the eigenvalues
        pred_weights = coefficients.squeeze() * self.ssm.eigenvalues
        reconstructions = self.ssm.decode(pred_weights)
        # pred_points = self.dgcnn(x)
        # pred_weights = self.ssm(pred_points)
        # reconstructions = self.ssm.decode(pred_weights)
        return reconstructions, pred_weights.squeeze()

    def fit_ssm(self, shapes):
        self.ssm.fit(shapes)
        self.config['ssm_modes'] = self.ssm.num_modes.data.item()

        # # make the model regress the correct number of modes for the SSM
        # self.dgcnn.regression[-1] = SharedFullyConnected(256, self.ssm.num_modes, dim=1, last_layer=True).to(next(self.parameters()).device)
        # self.dgcnn.init_weights()
        self.dgcnn.linear3 = nn.Linear(256, self.config['ssm_modes'])
        self.dgcnn.apply(init_weights)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        ssm_state = {key.replace("ssm.", ""): value for key, value in checkpoint['model_state'].items() if "ssm." in key}
        model.ssm.register_parameters_from_state_dict(ssm_state)
        return model
