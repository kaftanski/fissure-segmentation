import matplotlib as mpl
import matplotlib.pyplot as plt

from thesis.utils import save_fig, textwidth_to_figsize

plt.style.use('seaborn-talk')

values = {
    'DGCNN (Förstner)': (1.1789+0.1723+0.0009, 3.54, mpl.cm.get_cmap('tab10').colors[1]),
    'DGCNN (Hessian)': (2.3048+34.5032+0.0009, 5.05, mpl.cm.get_cmap('tab10').colors[0]),
    'DGCNN (CNN)': (6.4817+0.3037+0.0009, 3.07, mpl.cm.get_cmap('tab10').colors[2]),
    'DGCNN+PC-AE (Förstner)': (0.2444+0.1723+0.0009, 7.44, mpl.cm.get_cmap('tab20').colors[3]),
    'DGCNN+PC-AE (Hessian)': (0.4746+34.5032+0.0009, 8.66, mpl.cm.get_cmap('tab20').colors[1]),
    'DGCNN+PC-AE (CNN)': (0.5644+0.3037+0.0009, 5.05, mpl.cm.get_cmap('tab20').colors[5]),
    'nnU-Net': (39.821, 2.39, mpl.cm.get_cmap('Dark2').colors[3]),
    # 'PointNet (Förstner)': (),
    # 'PointNet (CNN)': (),
}

fig = plt.figure(figsize=textwidth_to_figsize(0.5, aspect=2/3, presentation=True))
for model in values.keys():
    time, assd, color = values[model]
    plt.scatter(time, assd, c=color)
    if model == 'nnU-Net' or 'Hessian' in model:
        ha = 'right'
        x_adjust = -1
        y_adjust = 0
    else:
        ha = 'left'
        x_adjust = 1
        y_adjust = 0
    plt.annotate(model, (time + x_adjust, assd + y_adjust), ha=ha)
plt.xlabel('Inference Time [s]')
plt.ylabel('mean ASSD [mm]')
save_fig(fig, 'results/plots', 'time_performance_plot', pdf=False)
