import os

import torch

from shape_model.ssm import LSSM, save_shape

if __name__ == '__main__':
    n_samples = 1000
    out_dir = 'results/ssm_points_sampled/'
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda:2'
    ssm = LSSM.load('results/corresponding_points/ssm.pth', device)
    samples = ssm.decode(ssm.random_samples(n_samples))
    samples = torch.stack([samples[:, :1024], samples[:, 1024:]], dim=1)
    for i, s in enumerate(samples):
        s = s.cpu().numpy()
        # visualize_point_cloud(np.concatenate([s[0], s[1]]), labels=torch.cat([torch.ones(1024), torch.ones(1024)+1]))
        save_shape(s, os.path.join(out_dir, f'SMPL{i:03d}_fixed.npz'))
