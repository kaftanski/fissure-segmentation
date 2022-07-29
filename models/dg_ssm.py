import torch

from models.dgcnn import DGCNNReg, SharedFullyConnected
from models.modelio import LoadableModel, store_config_args
from shape_model.ssm import SSM


class DGSSM(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 ssm_alpha=3., ssm_targ_var=0.95, ssm_modes=1):
        super(DGSSM, self).__init__()
        self.ssm = SSM(ssm_alpha, ssm_targ_var)
        self.dgcnn = DGCNNReg(k, in_features, ssm_modes,  # placeholder number of modes, has to be updated after training SSM
                              spatial_transformer, dynamic, image_feat_module)

    def forward(self, x):
        self.ssm.assert_trained()

        pred_weights = self.dgcnn(x)
        reconstructions = self.ssm.decode(pred_weights)
        return reconstructions, pred_weights.squeeze()

    def fit_ssm(self, shapes):
        self.ssm.fit(shapes)
        self.config['ssm_modes'] = self.ssm.num_modes.data

        # make the model regress the correct number of modes for the SSM
        self.dgcnn.regression[-1] = SharedFullyConnected(256, self.ssm.num_modes, dim=1, last_layer=True).to(next(self.parameters()).device)
        self.dgcnn.init_weights()

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        ssm_state = {key.replace("ssm.", ""): value for key, value in checkpoint['model_state'].items() if "ssm." in key}
        model.ssm.register_parameters_from_state_dict(ssm_state)
        return model
