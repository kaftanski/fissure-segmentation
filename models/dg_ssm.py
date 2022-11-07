from types import SimpleNamespace

import torch
from torch import nn

from augmentations import compose_transform, transform_points
from models.dgcnn_opensrc import DGCNN
from models.modelio import LoadableModel, store_config_args
from models.utils import init_weights
from shape_model.ssm import SSM, LSSM


class RegressionHead(nn.Module):
    def __init__(self, in_channels, out_channel_list, dropout=0.):
        super(RegressionHead, self).__init__()
        out_channels = out_channel_list.pop(0)
        self.layers = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False)])
        for i, oc in enumerate(out_channel_list):
            self.layers.extend([nn.BatchNorm1d(out_channels),
                                nn.Dropout(p=dropout),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Linear(out_channels, oc, bias=not i == len(out_channel_list)-1)])
            out_channels = oc

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiHeadDGCNN(DGCNN):
    def __init__(self, dgcnn_args, input_channels, output_channels_main, other_heads_out):
        super(MultiHeadDGCNN, self).__init__(dgcnn_args, input_channels, output_channels_main)
        self.heads = nn.ModuleDict()
        self.head_active = {'main': True}
        for name, channels in other_heads_out.items():
            self.heads[name] = RegressionHead(dgcnn_args.emb_dims*2, channels, dgcnn_args.dropout)
            self.head_active[name] = True

        self.feat = {}
        self.linear1.register_forward_hook(create_in_feature_hook(self.feat, 'global_feature'))

    def forward(self, x):
        main_head_out = super(MultiHeadDGCNN, self).forward(x)
        global_feature = self.feat['global_feature'][0]
        other_heads_out = {}
        for name, head in self.heads.items():
            if self.head_active[name]:
                other_heads_out[name] = head(global_feature)
            else:
                if name == "scaling":
                    tensor_op = torch.ones
                else:
                    tensor_op = torch.zeros
                other_heads_out[name] = tensor_op(x.shape[0],
                    head.layers[-1].out_features, device=x.device, dtype=main_head_out.dtype)

        return main_head_out if \
                   self.head_active['main'] else \
                   torch.zeros_like(main_head_out, device=main_head_out.device),\
               other_heads_out

    def set_head_active(self, name, active=True):
        self.head_active[name] = active

    def predict_full_pointcloud(self, pc, sample_points=1024, n_runs_min=50):
        batch_size = pc.shape[0]
        coefficient_accumulation = torch.zeros(batch_size, self.linear3.out_features, 1, device=pc.device)
        transform_accumulation = {}
        for name in self.heads.keys():
            transform_accumulation[name] = torch.zeros(batch_size, self.heads[name].layers[-1].out_features, 1, device=pc.device)

        for i in range(n_runs_min):
            perm = torch.randperm(pc.shape[-1], device=pc.device)[:sample_points]
            coeff, transforms = self(pc[..., perm])
            coefficient_accumulation += coeff
            for name in self.heads.keys():
                transform_accumulation[name] += transforms[name]

        # TODO: not all points may have been seen.

        return coefficient_accumulation / n_runs_min, {name: val / n_runs_min for name, val in transform_accumulation.items()}


def create_in_feature_hook(feature_dict, name):
    def input_hook(model, input, output):
        feature_dict[name] = input
    return input_hook


class DGSSM(LoadableModel):
    @store_config_args
    def __init__(self, k, in_features, spatial_transformer=False, dynamic=True, image_feat_module=False,
                 predict_affine_params=True, ssm_alpha=3., ssm_targ_var=0.95, ssm_modes=1, lssm=False):
        super(DGSSM, self).__init__()
        if spatial_transformer:
            raise NotImplementedError()
        if not dynamic:
            raise NotImplementedError()
        if image_feat_module:
            raise NotImplementedError()

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
        # self.dgcnn = DGCNN(dgcnn_args, input_channels=in_features,
        #                    output_channels=ssm_modes + 9 if predict_affine_params else 0)
        self.dgcnn = MultiHeadDGCNN(dgcnn_args, input_channels=in_features, output_channels_main=ssm_modes,
                                    other_heads_out={
                                        'translation': [512, 50, 3],
                                        'rotation': [512, 50, 3],
                                        'scaling': [512, 50, 3]
                                    })

    def forward(self, x):
        self.ssm.assert_trained()

        x = self.dgcnn(x)
        coefficients, so3_rotation, translation, scaling = self.split_prediction(x)
        pred_weights = coefficients.squeeze() * self.ssm.eigenvalues  # coefficients are multipliers for the eigenvalues
        reconstructions = self.ssm.decode(pred_weights)

        if self.predict_affine_params:
            reconstructions = transform_points(
                reconstructions.transpose(1, 2), compose_transform(so3_rotation, translation, scaling))
        # else: affine params are the identity transform

        return reconstructions.transpose(1, 2), pred_weights, torch.cat((so3_rotation, translation, scaling), dim=1)

    def fit_ssm(self, shapes):
        self.ssm.fit(shapes)
        self.config['ssm_modes'] = self.ssm.num_modes.data.item()

        # # make the model regress the correct number of modes for the SSM
        # self.dgcnn.regression[-1] = SharedFullyConnected(256, self.ssm.num_modes, dim=1, last_layer=True).to(next(self.parameters()).device)
        # self.dgcnn.init_weights()
        self.dgcnn.linear3 = nn.Linear(256, self.config['ssm_modes'] if self.predict_affine_params else 0)
        self.dgcnn.apply(init_weights)

    def split_prediction(self, dgcnn_pred):
        if self.predict_affine_params:
            return dgcnn_pred[0], dgcnn_pred[1]['rotation'], dgcnn_pred[1]['translation'], dgcnn_pred[1]['scaling']
        else:
            bs = dgcnn_pred.shape[0]
            return dgcnn_pred, torch.zeros(bs, 3).to(dgcnn_pred), torch.zeros(bs, 3).to(dgcnn_pred), torch.zeros(bs, 3).to(dgcnn_pred)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        ssm_state = {key.replace("ssm.", ""): value for key, value in checkpoint['model_state'].items() if "ssm." in key}
        model.ssm.register_parameters_from_state_dict(ssm_state)
        return model

    def set_head_active(self, name, active=True):
        self.dgcnn.set_head_active(name, active)
