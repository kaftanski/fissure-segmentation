from matplotlib import pyplot as plt
from torch import optim, nn

from thesis.utils import save_fig, textwidth_to_figsize


def plot_cosine_lrs(tmax, epochs, lr, wr=True, legend=False, fig=None):
    m = nn.Linear(1, 1)
    o = optim.SGD(m.parameters(), lr)
    if wr:
        s = optim.lr_scheduler.CosineAnnealingWarmRestarts(o, tmax, eta_min=lr * 0.05)
    else:
        s = optim.lr_scheduler.CosineAnnealingLR(o, epochs, eta_min=lr*0.05)

    lrs = []
    for i in range(epochs):
        s.step()
        # print(type(o.param_groups[0]['lr']))
        lrs.append(o.param_groups[0]['lr'])
    if fig is None:
        plt.figure(figsize=textwidth_to_figsize(0.6))
    plt.plot(lrs, label='with warm restarts' if wr else 'cosine annealing')
    if legend:
        plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')


if __name__ == '__main__':
    epochs = 1000
    lr = 0.001
    tmax = epochs // 4 + 1

    plot_cosine_lrs(tmax, epochs, lr, wr=False)
    save_fig(plt.gcf(), 'results/plots', 'cosine_annealing', show=True)

    plot_cosine_lrs(tmax, epochs, lr)
    save_fig(plt.gcf(), 'results/plots', 'cosine_annealing_warm_restarts', show=True)

    plot_cosine_lrs(tmax, epochs, lr, wr=False, legend=True)
    plot_cosine_lrs(tmax, epochs, lr, legend=True, fig=plt.gcf())
    save_fig(plt.gcf(), 'results/plots', 'cosine_annealing_both', show=True)

