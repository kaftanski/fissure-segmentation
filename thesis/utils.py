import os.path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

TEXT_WIDTH_INCH = 6.22404097223
# default pyplot figure size w=6.4, h=4.8


plt.style.use('seaborn-paper')


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


def textwidth_to_figsize(w_frac, aspect=2/3):
    w = w_frac * TEXT_WIDTH_INCH
    h = w * aspect
    return w, h