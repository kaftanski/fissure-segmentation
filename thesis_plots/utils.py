import os.path

import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')


def save_fig(fig, outdir, basename_without_extension, dpi=300, show=True, pdf=True):
    os.makedirs(outdir, exist_ok=True)
    extension = '.png' if not pdf else '.pdf'
    path = os.path.join(outdir, basename_without_extension + extension)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
