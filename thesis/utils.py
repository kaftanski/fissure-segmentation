import csv
import os.path

import torch
from thop import profile, clever_format

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torchinfo import summary
from ptflops import get_model_complexity_info


TEXT_WIDTH_INCH = 6.22404097223
# default pyplot figure size w=6.4, h=4.8
SLIDE_WIDTH_INCH = 13.3334646  # default wide screen power point slide (16:9)
SLIDE_HEIGHT_INCH = 7.5  # default 16:9 slide

plt.style.use('seaborn-v0_8-paper')  # TODO: test out newer seaborn styles


def param_and_op_count(model, input_shape, out_dir=None, fname='op_count.csv'):
    input = torch.zeros(input_shape, device=next(model.parameters()).device)
    macs_thop, params_thop = profile(model, (input, ))
    #stats_torchinfo = summary(model, input_size=input_shape)
    macs_ptflops, params_ptflops = get_model_complexity_info(model, input_shape[1:], as_strings=False,
                                                             print_per_layer_stat=False, verbose=True)
    if out_dir is not None:
        with open(os.path.join(out_dir, 'op_count.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Counting method', 'Parameters', 'MACs'])
            writer.writerow(['thop', params_thop, macs_thop])
            #writer.writerow(['torchinfo', stats_torchinfo.trainable_params, stats_torchinfo.total_mult_adds])
            writer.writerow(['ptflops', params_ptflops, macs_ptflops])

    macs_thop, params_thop = clever_format([macs_thop, params_thop], "%.3f")
    print(macs_thop, params_thop)
    return macs_thop, params_thop


def save_fig(fig, outdir, basename_without_extension, dpi=300, show=True, pdf=True, padding=False, bbox_inches='tight'):
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    extension = '.png' if not pdf else '.pdf'
    path = os.path.join(outdir, basename_without_extension + extension)

    if padding:
        pad_inches = 0.1
    else:
        if not pdf:
            # need a little padding for pngs
            pad_inches = 0.02
        else:
            pad_inches = 0

    fig.savefig(path, bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)
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