from types import SimpleNamespace

import torch
from torch import nn

from augmentations import compose_transform, transform_points_with_centering
from models.dgcnn_opensrc import DGCNN
from models.modelio import LoadableModel, store_config_args
from models.utils import init_weights
from shape_model.ssm import SSM, LSSM


class DGSSM(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 predict_affine_params=True, ssm_alpha=3., ssm_targ_var=0.95, ssm_modes=1, lssm=False):
        super(DGSSM, self).__init__()
        SSMClass = SSM if not lssm else LSSM
        self.predict_affine_params = predict_affine_params
        self.ssm = SSMClass(ssm_alpha, ssm_targ_var)
        # self.dgcnn = DGCNNReg(k, in_features, dgcnn_out_features,  # placeholder number of modes, has to be updated after training SSM
        #                       spatial_transformer, dynamic, image_feat_module)
        dgcnn_args = SimpleNamespace(
            k=k,
            emb_dims=1024,  # length of global feature vector
            dropout=0.
        )
        self.dgcnn = DGCNN(dgcnn_args, input_channels=in_features,
                           output_channels=ssm_modes + 6 if predict_affine_params else 0)

    def forward(self, x):
        self.ssm.assert_trained()

        coefficients = self.dgcnn(x)  # predict the coefficient multipliers for the eigenvalues
        if self.predict_affine_params:
            coefficients, so3_rotation, translation = self.split_prediction(coefficients)
        pred_weights = coefficients.squeeze() * self.ssm.eigenvalues
        reconstructions = self.ssm.decode(pred_weights)

        if self.predict_affine_params:
            reconstructions = transform_points_with_centering(
                reconstructions.transpose(1, 2), compose_transform(so3_rotation.squeeze(-1), translation.squeeze(-1)))
        return reconstructions.transpose(1, 2), torch.cat((pred_weights.squeeze(), so3_rotation, translation), dim=1)

    def fit_ssm(self, shapes):
        self.ssm.fit(shapes)
        self.config['ssm_modes'] = self.ssm.num_modes.data.item()

        # # make the model regress the correct number of modes for the SSM
        # self.dgcnn.regression[-1] = SharedFullyConnected(256, self.ssm.num_modes, dim=1, last_layer=True).to(next(self.parameters()).device)
        # self.dgcnn.init_weights()
        self.dgcnn.linear3 = nn.Linear(256, self.config['ssm_modes'] + 6 if self.predict_affine_params else 0)
        self.dgcnn.apply(init_weights)

    def split_prediction(self, dgcnn_pred):
        if self.predict_affine_params:
            return dgcnn_pred.split([self.ssm.num_modes.data, 3, 3], dim=1)
        else:
            bs = dgcnn_pred.shape[0]
            return dgcnn_pred, torch.zeros(bs, 3).to(dgcnn_pred), torch.zeros(bs, 3).to(dgcnn_pred)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        ssm_state = {key.replace("ssm.", ""): value for key, value in checkpoint['model_state'].items() if "ssm." in key}
        model.ssm.register_parameters_from_state_dict(ssm_state)
        return model
