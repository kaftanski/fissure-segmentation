import csv
import os.path

import torch
from thop import profile, clever_format

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEXT_WIDTH_INCH = 6.22404097223
# default pyplot figure size w=6.4, h=4.8
SLIDE_WIDTH_INCH = 13.3334646  # default wide screen power point slide

plt.style.use('seaborn-paper')


def param_and_op_count(model, input_shape, out_dir=None):
    input = torch.zeros(input_shape)
    macs, params = profile(model, (input, ))
    if out_dir is not None:
        with open(os.path.join(out_dir, 'op_count.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Parameters', 'FLOPs'])
            writer.writerow([params, macs])
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    return macs, params


def save_fig(fig, outdir, basename_without_extension, dpi=300, show=True, pdf=True, padding=False):
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    extension = '.png' if not pdf else '.pdf'
    path = os.path.join(outdir, basename_without_extension + extension)
    fig.savefig(path, bbox_inches='tight', dpi=dpi, **{} if padding else {'pad_inches': 0})
    if show:
        plt.show()
    else:
        plt.close(fig)


def legend_figure(labels, colors, outdir, basename):
    legend_figure = plt.figure(figsize=textwidth_to_figsize(0.2, 1 / 2))
    legend_figure.legend(handles=[Patch(facecolor=color, label=label) for label, color in zip(labels, colors)], loc='center')
    save_fig(legend_figure, outdir, basename, show=False)


def textwidth_to_figsize(w_frac, aspect=2/3, presentation=False):
    w = w_frac * (TEXT_WIDTH_INCH if not presentation else SLIDE_WIDTH_INCH)
    h = w * aspect
    return w, h