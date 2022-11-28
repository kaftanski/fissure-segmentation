import os.path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from constants import KP_MODES, FEATURE_MODES
from thesis import textwidth_to_figsize
from thesis.utils import save_fig


def csv_to_df(csv_result_file):
    results_table = pd.read_csv(csv_result_file)
    results_table = results_table.set_index('Class').transpose()
    results_table = results_table[:-1]  # discard background
    results_table['Fissure'][:-1] = results_table['Fissure'][:-1].astype(int)
    results_table = results_table.set_index('Fissure')
    results_table = results_table.astype(float)

    def fix_dice(which='Mean'):
        results_table[f'{which} Dice'][1] = results_table[f'{which} Dice'][2]
        results_table[f'{which} Dice'][2] = results_table[f'{which} Dice'][3]
        results_table[f'{which} Dice'][3] = results_table[f'{which} Dice']['mean']
        results_table[f'{which} Dice']['mean'] = results_table[f'{which} Dice'][:-1].mean()

    fix_dice('Mean')
    fix_dice('StdDev')

    results_table = results_table.round(2)

    mod_table = pd.DataFrame(columns=['ASD_mean', 'ASD_std', 'SDSD_mean', 'SDSD_std', 'HD_mean', 'HD_std', 'missing'])
    mod_table['ASD_mean'] = results_table['Mean ASSD']
    mod_table['ASD_std'] = results_table['StdDev ASSD']
    mod_table['SDSD_mean'] = results_table['Mean SDSD']
    mod_table['SDSD_std'] = results_table['StdDev SDSD']
    mod_table['HD_mean'] = results_table['Mean HD']
    mod_table['HD_std'] = results_table['StdDev HD']
    mod_table['missing'] = results_table['proportion missing']
    return mod_table


def pm_table(table):
    mean_exits = []
    std_exits = []
    for id in map(str.strip, table.columns):
        if id.endswith('_mean'):
            mean_exits.append(id.removesuffix('_mean'))
        if id.endswith('_std'):
            std_exits.append(id.removesuffix('_std'))

    for id in sorted(set(mean_exits).intersection(set(std_exits))):
        m_column = id + '_mean'
        s_column = id + '_std'

        new_col_idx = next(i for i, col in enumerate(table.columns) if id in col)
        table.insert(new_col_idx, id.strip(), table[m_column].astype(str) + ' ± ' + table[s_column].astype(str))
        table = table.drop(columns=[m_column, s_column])

    return table


def get_all_tables(model='DGCNN'):
    tables = {}
    for kp in KP_MODES:
        tables[kp] = {}

        if kp == 'cnn':
            cur_feat = FEATURE_MODES + ['cnn']
        else:
            cur_feat = FEATURE_MODES

        for feat in cur_feat:
            folder = os.path.join('results', f'{model}_seg_{kp}_{feat}')
            result_file = os.path.join(folder, 'cv_results.csv')
            if os.path.isfile(result_file):
                table = csv_to_df(result_file)
                print(f"{kp}_{feat}")
                table['Fissure'] = [1, 2, 3, 'mean']
            else:
                table = pd.DataFrame(index=[0])
                print(f'missing experiment {kp}_{feat}')
                table['Fissure'] = None

            table['Keypoints'] = kp
            table['Features'] = feat

            tables[kp][feat] = table

    return tables


# def dgcnn_seg_table():
#     tables = get_all_tables()
#
#     combined_table = None
#     for kp in tables.keys():
#         for feat in tables[kp].keys():
#             table = pm_table(tables[kp][feat])
#             if combined_table is None:
#                 combined_table = table
#             else:
#                 combined_table = pd.concat((combined_table, table))
#
#     # reorder columns
#     # cols = list(combined_table.columns.values)
#     # cols.insert(0, cols.pop(-1))
#     # cols.insert(0, cols.pop(-1))
#     # combined_table = combined_table[cols]
#     combined_table = combined_table.set_index(['Keypoints', 'Features', 'Fissure'], drop=True)
#
#     print(combined_table.to_latex(multirow=True, multicolumn=True))


def dgcnn_seg_bar_plot(metric='ASD'):
    feat_modes = FEATURE_MODES + ['cnn']
    tables = get_all_tables()
    index = np.arange(len(tables.keys()))
    group_width = 0.7
    bar_width = group_width / len(feat_modes)
    cmap = mpl.cm.get_cmap('tab10')
    colors = {feat: cmap(i/10) for i, feat in enumerate(feat_modes)}
    fig = plt.figure(figsize=textwidth_to_figsize(0.7, 3/5))

    for i, kp in zip(index, tables.keys()):
        group = tables[kp]
        x = np.linspace(i - bar_width*len(group)/2, i + bar_width*len(group)/2, num=len(group))
        for j, feat in zip(x, group.keys()):
            if len(tables[kp][feat]) <= 1:
                continue
            plt.bar(j, height=tables[kp][feat][f'{metric}_mean']['mean'], width=bar_width,
                    yerr=tables[kp][feat][f'{metric}_std']['mean'], color=colors[feat])

    plt.xticks(index, labels=[kp.replace('oe', 'ö').replace('enhancement', 'hessian') for kp in tables.keys()])
    plt.ylabel(f'mean {metric} [mm]')
    save_fig(fig, 'results/plots', f'dgcnn_seg_{metric}')

    legend_figure = plt.figure(figsize=textwidth_to_figsize(0.2, 1/2))
    legend_figure.legend(handles=[Patch(
                         facecolor=colors[feat],
                         label=feat.capitalize().replace('Cnn', 'CNN').replace('Nofeat', 'None').replace('Mind_ssc', 'SSC').replace('Mind', 'MIND')) for feat in feat_modes],
        loc='center'
    )
    save_fig(legend_figure, 'results/plots', 'dgcnn_seg_legend')


def time_table(path='results/preproc_timing/timings.csv'):
    table = pd.read_csv(path)
    table = table.set_index(['Kind', 'Mode'])
    table = table.round(4)
    table = pm_table(table)
    print(table.to_latex(multirow=True, multicolumn=True))


def seg_table(model='DGCNN', only_one_feature: str=None):
    tables = get_all_tables(model)

    combined_table = None
    for kp in tables.keys():
        for feat in (tables[kp].keys() if only_one_feature is None else [only_one_feature]):
            table = pm_table(tables[kp][feat])
            if table.shape[0] == 1:
                continue
            if combined_table is None:
                combined_table = table
            else:
                combined_table = pd.concat((combined_table, table))

    combined_table = combined_table.set_index(['Keypoints', 'Features', 'Fissure'], drop=True)

    print(combined_table.to_latex(multirow=True, multicolumn=True))


def bar_plot(model):
    for metric in ['ASD', 'SDSD', 'HD']:
        feat_modes = FEATURE_MODES + ['cnn']
        tables = get_all_tables(model)
        index = np.arange(len(tables.keys()))
        group_width = 0.7
        bar_width = group_width / len(feat_modes)
        cmap = mpl.cm.get_cmap('tab10')
        colors = {feat: cmap(i/10) for i, feat in enumerate(feat_modes)}
        fig = plt.figure(figsize=textwidth_to_figsize(0.7, 3/5))

        for i, kp in zip(index, tables.keys()):
            group = tables[kp]
            x = np.linspace(i - bar_width*len(group)/2, i + bar_width*len(group)/2, num=len(group))
            for j, feat in zip(x, group.keys()):
                if len(tables[kp][feat]) <= 1:
                    continue
                plt.bar(j, height=tables[kp][feat][f'{metric}_mean']['mean'], width=bar_width,
                        yerr=tables[kp][feat][f'{metric}_std']['mean'], color=colors[feat])

        plt.xticks(index, labels=[kp.replace('oe', 'ö').replace('enhancement', 'hessian') for kp in tables.keys()])
        plt.ylabel(f'mean {metric} [mm]')
        save_fig(fig, 'results/plots', f'{model}_seg_{metric}')

        legend_figure = plt.figure(figsize=textwidth_to_figsize(0.2, 1/2))
        legend_figure.legend(handles=[Patch(
                             facecolor=colors[feat],
                             label=feat.capitalize().replace('Cnn', 'CNN').replace('Nofeat', 'None').replace('Mind_ssc', 'SSC').replace('Mind', 'MIND')) for feat in feat_modes],
            loc='center'
        )
        save_fig(legend_figure, 'results/plots', f'{model}_seg_legend')


def point_net_seg_table():
    seg_table('PointNet', 'image')


def dgcnn_seg_table():
    seg_table('DGCNN', None)


if __name__ == '__main__':
    KP_MODES.remove('noisy')
    FEATURE_MODES.remove('cnn')
    FEATURE_MODES = FEATURE_MODES + ['nofeat']

    # dgcnn_seg_table()
    # time_table()
    # point_net_seg_table()
    # bar_plot('DGCNN')
    seg_table('DGCNN', 'image')
