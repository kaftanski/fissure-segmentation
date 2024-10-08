import os.path
from collections import OrderedDict

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from constants import KP_MODES, FEATURE_MODES
from thesis.utils import save_fig, legend_figure, textwidth_to_figsize, SLIDE_WIDTH_INCH, SLIDE_HEIGHT_INCH


METRIC_RENAMER = {'ASSD_mean': 'ASSD', 'SDSD_mean': 'SDSD', 'HD_mean': 'HD'}


def removesuffix(text, suffix):
    """ for compatibility in Python 3.8 """
    if text.endswith(suffix):
        return text[:-len(suffix)]
    else:
        return text


def csv_to_df(csv_result_file):
    results_table = pd.read_csv(csv_result_file)
    if 'Class' in results_table.columns:
        results_table = results_table.set_index('Class').transpose()
    else:
        results_table = results_table.set_index('Fissure').transpose()
        results_table['Fissure'] = results_table.index

    if results_table.shape[0] > 4:
        results_table = results_table[:4]  # discard background of dice column

        def fix_dice(which='Mean'):
            results_table[f'{which} Dice'][1] = results_table[f'{which} Dice'][2]
            results_table[f'{which} Dice'][2] = results_table[f'{which} Dice'][3]
            results_table[f'{which} Dice'][3] = results_table[f'{which} Dice']['mean']
            results_table[f'{which} Dice']['mean'] = results_table[f'{which} Dice'][:-1].mean()

    else:
        def fix_dice(which=None):
            pass

    results_table.replace(float('NaN'), 0, inplace=True)
    results_table['Fissure'][results_table['Fissure']!='mean'] = results_table['Fissure'][results_table['Fissure']!='mean'].astype(int)
    results_table = results_table.set_index('Fissure')
    results_table = results_table.replace(',', '.', regex=True)  # fix comma as decimal delimiter
    results_table = results_table.astype(float)

    if 'Mean Dice' in results_table.columns:
        fix_dice('Mean')
        fix_dice('StdDev')

    results_table = results_table.round(2)

    mod_table = pd.DataFrame(columns=['ASSD_mean', 'ASSD_std', 'SDSD_mean', 'SDSD_std', 'HD_mean', 'HD_std', 'missing'])
    mod_table['ASSD_mean'] = results_table['Mean ASSD']
    mod_table['ASSD_std'] = results_table['StdDev ASSD']
    mod_table['SDSD_mean'] = results_table['Mean SDSD']
    mod_table['SDSD_std'] = results_table['StdDev SDSD']
    mod_table['HD_mean'] = results_table['Mean HD']
    mod_table['HD_std'] = results_table['StdDev HD']
    mod_table['missing'] = results_table['proportion missing']
    if mod_table.index[-1]==0:
        mod_table = mod_table.drop(index=0)
    return mod_table


def pm_table(table):
    mean_exits = []
    std_exits = []
    for id in map(str.strip, table.columns):
        if id.endswith('_mean'):
            mean_exits.append(removesuffix(id, '_mean'))
        if id.endswith('_std'):
            std_exits.append(removesuffix(id, '_std'))

    for id in sorted(set(mean_exits).intersection(set(std_exits))):
        m_column = id + '_mean'
        s_column = id + '_std'

        new_col_idx = next(i for i, col in enumerate(table.columns) if id in col)
        table.insert(new_col_idx, id.strip(), table[m_column].astype(str) + ' ± ' + table[s_column].astype(str))
        table = table.drop(columns=[m_column, s_column])

    return table


def get_all_tables(model='DGCNN_seg', cv=True, copd=False, exclude_rhf=False):
    tables = {}
    for kp in KP_MODES:
        tables[kp] = {}

        if kp == 'cnn':
            cur_feat = FEATURE_MODES + ['cnn']
        else:
            cur_feat = FEATURE_MODES

        for feat in cur_feat:
            folder = os.path.join('./results', f'{model}_{kp}_{feat}')
            if cv:
                # only take averaged cross-validation results
                table = get_table_from_folder(feat, kp, folder, f'cv_results{"_copd" if copd else ""}.csv', exclude_rhf=exclude_rhf)
            else:
                # join values from all folds in the table
                table = None
                for f in range(5):
                    fold_table = get_table_from_folder(feat, kp, os.path.join(folder, f'fold{f}'), f'test_results{"_copd" if copd else ""}.csv', exclude_rhf=exclude_rhf)
                    if table is None:
                        table = pd.DataFrame(columns=fold_table.columns)

                    try:
                        table.loc[f] = fold_table.loc['mean']
                    except KeyError:
                        # experiment is missing altogether
                        table = fold_table

            tables[kp][feat] = table

    return tables


def exclude_rhf_from_table(table):
    table = table.drop(index=3, inplace=False)
    table[table.index == 'mean'] = table[table.index != 'mean'].mean().round(2)
    return table


def get_table_from_folder(feat, kp, folder, filename, exclude_rhf=False):
    result_file = os.path.join(folder, filename)
    if os.path.isfile(result_file):
        table = csv_to_df(result_file)
        print(f"{kp}_{feat}")
        if exclude_rhf:
            table = exclude_rhf_from_table(table)

        table['Fissure'] = [*range(1, len(table)), 'mean']
    else:
        table = pd.DataFrame(index=[0])
        print(f'missing experiment {kp}_{feat}')
        table['Fissure'] = None
    table['Keypoints'] = kp
    table['Features'] = feat
    return table


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

#
# def dgcnn_seg_bar_plot(metric='ASD'):
#     feat_modes = FEATURE_MODES + ['cnn']
#     tables = get_all_tables()
#     index = np.arange(len(tables.keys()))
#     group_width = 0.7
#     bar_width = group_width / len(feat_modes)
#     cmap = mpl.cm.get_cmap('tab10')
#     colors = {feat: cmap(i/10) for i, feat in enumerate(feat_modes)}
#     fig = plt.figure(figsize=textwidth_to_figsize(0.7, 3/5))
#
#     for i, kp in zip(index, tables.keys()):
#         group = tables[kp]
#         x = np.linspace(i - bar_width*len(group)/2, i + bar_width*len(group)/2, num=len(group))
#         for j, feat in zip(x, group.keys()):
#             if len(tables[kp][feat]) <= 1:
#                 continue
#             plt.bar(j, height=tables[kp][feat][f'{metric}_mean']['mean'], width=bar_width,
#                     yerr=tables[kp][feat][f'{metric}_std']['mean'], color=colors[feat])
#
#     plt.xticks(index, labels=[kp.replace('oe', 'ö').replace('enhancement', 'hessian') for kp in tables.keys()])
#     plt.ylabel(f'mean {metric} [mm]')
#     save_fig(fig, 'results/plots', f'dgcnn_seg_{metric}')
#
#     legend_figure = plt.figure(figsize=textwidth_to_figsize(0.2, 1/2))
#     legend_figure.legend(handles=[Patch(
#                          facecolor=colors[feat],
#                          label=feat.capitalize().replace('Cnn', 'CNN').replace('Nofeat', 'None').replace('Mind_ssc', 'SSC').replace('Mind', 'MIND')) for feat in feat_modes],
#         loc='center'
#     )
#     save_fig(legend_figure, 'results/plots', 'dgcnn_seg_legend')


def time_table(path='results/preproc_timing/timings.csv'):
    table = pd.read_csv(path)
    table = table.set_index(['Kind', 'Mode'])
    table = table.round(4)
    table = pm_table(table)
    print(table.to_latex(multirow=True, multicolumn=True))


def seg_table(model='DGCNN_seg', only_one_feature: str=None, copd=False, exclude_rhf=False, pm=True):
    tables = get_all_tables(model, copd=copd, exclude_rhf=exclude_rhf)

    combined_table = None
    for kp in tables.keys():
        for feat in (tables[kp].keys() if only_one_feature is None else [only_one_feature]):
            table = pm_table(tables[kp][feat]) if pm else tables[kp][feat]
            if table.shape[0] == 1:
                continue
            if combined_table is None:
                combined_table = table
            else:
                combined_table = pd.concat((combined_table, table))

    combined_table = combined_table.set_index(['Keypoints', 'Features', 'Fissure'], drop=True)

    print(combined_table.to_latex(multirow=True, multicolumn=True))
    return combined_table


def bar_plot(model, presentation=False):
    tables = get_all_tables(model)

    for metric in ['ASSD', 'SDSD', 'HD']:

        cmap = mpl.cm.get_cmap('tab10')
        if not presentation:
            use_ssc = False  # unused in this application
            feat_modes = FEATURE_MODES + ['cnn']
            colors = {feat: cmap(i / 10) for i, feat in enumerate(feat_modes)}
            group_width = 0.75
        else:
            use_ssc = True
            plt.style.use("seaborn-v0_8-talk")
            feat_modes = ['image', 'nofeat']
            colors = {'image': cmap.colors[2], 'nofeat': 'gray'}
            if use_ssc:
                feat_modes.insert(1, 'mind_ssc')
                colors['mind_ssc'] = cmap.colors[1]
            group_width = 0.8

        index = np.arange(len(tables.keys()))
        bar_width = group_width / len(feat_modes)
        fig = plt.figure(figsize=textwidth_to_figsize(0.5, 3/4, presentation))

        for i, kp in zip(index, tables.keys()):
            group = tables[kp]
            cur_feat = list(group.keys())
            cur_feat = [f for f in cur_feat if f in feat_modes]
            x = np.linspace(i - group_width/2 + bar_width/2, i + group_width/2 - bar_width/2, num=len(cur_feat))
            for j, feat in zip(x, cur_feat):
                if len(tables[kp][feat]) <= 1:
                    continue

                value = tables[kp][feat][f'{metric}_mean']['mean']
                if not presentation:
                    # error bar
                    bar_kwargs = {'yerr': tables[kp][feat][f'{metric}_std']['mean']}
                else:
                    # no error bar
                    bar_kwargs = {}
                bar = plt.bar(j, height=value, width=bar_width*0.9, color=colors[feat], **bar_kwargs)

                if presentation:
                    # show value
                    plt.bar_label(bar)

        plt.xticks(index, labels=[kp.replace('oe', 'ö').replace('enhancement', 'hessian') for kp in tables.keys()])
        plt.ylabel(f'mean {metric} [mm]')
        save_fig(fig, 'results/plots', f'{model}_{metric}{"_presentation" if presentation else ""}{"_with_ssc" if use_ssc else ""}', pdf=not presentation)

        legend_figure = plt.figure(figsize=textwidth_to_figsize(0.1 if presentation else 0.2, 1/2, presentation))
        legend_figure.legend(handles=[Patch(facecolor=colors[feat],
                  label=feat.capitalize().replace('Cnn', 'CNN').replace('Nofeat', 'None').replace('Mind_ssc', 'SSC').replace('Mind', 'MIND')) for feat in feat_modes],
            loc='center'
        )
        save_fig(legend_figure, 'results/plots', f'{model}_{"presentation_" if presentation else ""}legend{"_with_ssc" if use_ssc else ""}', pdf=not presentation)


def bar_plot_pointnet_vs_dgcnn(presentation=False):
    dgcnn_tables = get_all_tables('DGCNN_seg')
    pointnet_tables = get_all_tables('PointNet_seg')
    feature = 'image'
    tables = {kp: {'DGCNN': dgcnn_tables[kp][feature], 'PointNet': pointnet_tables[kp][feature]} for kp in dgcnn_tables.keys()}

    for metric in ['ASSD', 'SDSD', 'HD']:

        cmap = mpl.cm.get_cmap('tab10')
        if not presentation:
            group_width = 0.75
        else:
            plt.style.use("seaborn-v0_8-talk")
            group_width = 0.8

        models = ['DGCNN', 'PointNet']
        colors = {'DGCNN': cmap.colors[2], 'PointNet': cmap.colors[0]}

        index = np.arange(len(tables.keys()))
        bar_width = group_width / len(models)
        fig = plt.figure(figsize=textwidth_to_figsize(0.5, 3 / 4, presentation))

        for i, kp in zip(index, tables.keys()):
            group = tables[kp]
            cur_feat = list(group.keys())
            cur_feat = [f for f in cur_feat if f in models]
            x = np.linspace(i - group_width / 2 + bar_width / 2, i + group_width / 2 - bar_width / 2, num=len(cur_feat))
            for j, feat in zip(x, cur_feat):
                if len(tables[kp][feat]) <= 1:
                    continue

                value = tables[kp][feat][f'{metric}_mean']['mean']
                if not presentation:
                    # error bar
                    bar_kwargs = {'yerr': tables[kp][feat][f'{metric}_std']['mean']}
                else:
                    # no error bar
                    bar_kwargs = {}
                bar = plt.bar(j, height=value, width=bar_width * 0.9, color=colors[feat], **bar_kwargs)

                if presentation:
                    # show value
                    plt.bar_label(bar)

        plt.xticks(index, labels=[kp.replace('oe', 'ö').replace('enhancement', 'hessian') for kp in tables.keys()])
        plt.ylabel(f'mean {metric} [mm]')
        save_fig(fig, 'results/plots', f'DGCNNvPointNet_{metric}{"_presentation" if presentation else ""}',
                 pdf=not presentation)

        legend_figure = plt.figure(figsize=textwidth_to_figsize(0.1 if presentation else 0.2, 1 / 2, presentation))
        legend_figure.legend(handles=[Patch(facecolor=colors[model], label=model) for model in models], loc='center')
        save_fig(legend_figure, 'results/plots', f'DGCNNvPointNet_{"presentation_" if presentation else ""}legend',
                 pdf=not presentation)


def comparative_bar_plot(tables_per_model, colors=None, rhf_excluded=False):
    index = np.arange(len(tables_per_model.keys()))
    models = list(tables_per_model.keys())
    group_width = 0.7
    bar_width = group_width / len(models)

    if colors is None:
        col = list(mpl.cm.get_cmap('Dark2').colors)
        col[4] = col[0]
        col[1] = col[5]
        col[0] = mpl.cm.get_cmap('tab10').colors[2]
        colors = {model: col[i] for i, model in enumerate(models)}
    else:
        colors = {model: c for model, c in zip(models, colors)}


    x = np.linspace(0, bar_width*len(models), num=len(models))
    for metric in ['ASSD', 'SDSD', 'HD']:

        fig = plt.figure(figsize=textwidth_to_figsize(0.3, 3/2))
        for i, model in zip(index, tables_per_model.keys()):
            plt.bar(x[i], height=tables_per_model[model][f'{metric}_mean']['mean'], width=bar_width,
                    yerr=tables_per_model[model][f'{metric}_std']['mean'], color=colors[model])

        plt.xticks([], [])
        plt.ylabel(f'mean {metric} [mm]')
        save_fig(fig, 'results/plots', f'comparison_{metric}{"_no_RHF" if rhf_excluded else ""}')

        legend_figure(labels=models, colors=[colors[model] for model in models],
                      outdir='results/plots', basename='comparison_legend')


def cross_val_swarm_plot(model, use_median_instead_of_mean=False, presentation=True, add_nnu_value=True, copd=False, exclude_rhf=False):
    tables = get_all_tables(model, cv=False, copd=copd, exclude_rhf=exclude_rhf)
    combined_table = pd.concat([tables[kp][feat] for kp in tables.keys() for feat in tables[kp].keys()])

    # fix the index (need ascending integers)
    combined_table = combined_table.set_index(np.arange(combined_table.shape[0]))

    # fix the spelling
    combined_table = combined_table.replace(FEATURE_MODES + ['cnn'], FEATURE_MODES_NORMALIZED + ['CNN'])
    combined_table = combined_table.replace(KP_MODES, KP_MODES_NORMALIZED)
    combined_table = combined_table.rename(columns=METRIC_RENAMER)

    print(combined_table)

    nnu = nnunet_table('voxels', cv=True, copd=copd, exclude_rhf=exclude_rhf)
    nnu = nnu.rename(columns=METRIC_RENAMER)

    # plotting
    cmap = mpl.cm.get_cmap('tab10')
    if not presentation:
        feat_modes = FEATURE_MODES_NORMALIZED + ['CNN']
        colors = {feat: cmap(i / 10) for i, feat in enumerate(feat_modes)}
    else:
        plt.style.use("seaborn-v0_8-talk")
        feat_modes = ['Image', 'SSC', 'None']
        colors = {'SSC': cmap.colors[1], 'Image': cmap.colors[2], 'None': 'gray'}
        combined_table = combined_table.drop(combined_table[~combined_table.Features.isin(feat_modes)].index)

    sns.set_theme()

    print(combined_table)

    for metric in ['ASSD', 'SDSD', 'HD']:
        # swarm plot in categories
        swarm_plot = sns.catplot(data=combined_table, x='Features', y=metric, col='Keypoints', hue='Features', kind='swarm', palette=colors,
                                 height=SLIDE_HEIGHT_INCH * 0.5, aspect=2/3, legend_out=False, legend='auto')

        # overlay mean-lines by adding boxplots without their boxes
        m_props = {'color': 'k', 'ls': '-', 'lw': 1.5}
        swarm_plot.map_dataframe(sns.boxplot, data=combined_table, x='Features', y=metric, zorder=10,
                                 showmeans=True, meanline=True,
                                 meanprops={'visible': not use_median_instead_of_mean, **m_props},
                                 medianprops={'visible': use_median_instead_of_mean, **m_props},
                                 whiskerprops={'visible': False},
                                 showfliers=False,
                                 showbox=False,
                                 showcaps=False,
                                 labels=['mean'])

        # add the nnu-net baseline value
        if add_nnu_value:
            if use_median_instead_of_mean:
                nnu_error_value = nnu[f'{metric}'].median()
            else:
                nnu_error_value = nnu[f'{metric}'].mean()
            print(nnu_error_value)
            swarm_plot.map(plt.axhline, y=nnu_error_value, ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3])

        swarm_plot.set_axis_labels(x_var='', y_var=f'mean {metric} [mm]')
        handles, labels = swarm_plot.axes[-1][-1].get_legend_handles_labels()
        handles = handles + [Line2D([],[],linestyle=''), Line2D([], [], color='k', lw=1.5, label='Mean' if not use_median_instead_of_mean else 'Median'),
                             Line2D([], [], ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3], label='nnU-Net')]
        swarm_plot.add_legend(title='Features:', handles=handles)
        swarm_plot.set_titles('{col_name} KPs')

        save_fig(swarm_plot.fig, 'results/plots', f'{model}_{metric}{"_copd" if copd else ""}{"_presentation" if presentation else ""}_swarmplot{"_nnu" if add_nnu_value else ""}{"_median" if use_median_instead_of_mean else ""}{"_no_RHF" if exclude_rhf else ""}', pdf=not presentation, bbox_inches='')
    plt.show()


def bvm_plot(copd=False):
    text_width_inch = 4.8041

    exclude_rhf = False
    model = 'DGCNN_seg'

    if copd:
        combined_table, nnu, dgcnn_table = copd_change_table()
    else:
        tables = get_all_tables(model, cv=False, copd=copd, exclude_rhf=exclude_rhf)
        combined_table = pd.concat([tables[kp][feat] for kp in tables.keys() for feat in tables[kp].keys()])

        # fix the index (need ascending integers)
        combined_table = combined_table.set_index(np.arange(combined_table.shape[0]))

        # fix the spelling
        combined_table = combined_table.replace(FEATURE_MODES + ['cnn'], FEATURE_MODES_NORMALIZED + ['CNN'])
        combined_table = combined_table.replace(KP_MODES, KP_MODES_NORMALIZED)
        combined_table = combined_table.rename(columns=METRIC_RENAMER)

        # load nnu table
        nnu = nnunet_table('voxels', cv=True, copd=copd, exclude_rhf=exclude_rhf)

    nnu = nnu.rename(columns=METRIC_RENAMER)

    # plotting
    cmap = mpl.cm.get_cmap('tab10')
    feat_modes = ['Image', 'SSC', 'None']
    colors = {'SSC': cmap.colors[1], 'Image': cmap.colors[0], 'None': 'gray'}
    markers = ['x', '.', '+']
    combined_table = combined_table.drop(combined_table[~combined_table.Features.isin(feat_modes)].index)
    tick_steps = {'ASSD': 1, 'SDSD': 0.5, 'HD': 2}
    tick_step_copd = 0.1

    sns.set_theme(context='paper', style='whitegrid', font_scale=0.75, rc={'axes.spines.right': True, 'axes.grid': True}), #'xtick.bottom': True})
    #plt.rc('font', size=8)  # controls default text sizes
    print(combined_table)

    for i, metric in enumerate(['ASSD', 'SDSD', 'HD']):
        add_legend = i == 1 and not copd

        # swarm plot in categories
        # point_plot = sns.catplot(data=combined_table, x='Keypoints', y=metric, hue='Features',
        #                          kind='point', palette=colors,
        #                          height=text_width_inch*0.5, aspect=1/2,
        #                          legend_out=i==2, legend='auto', markers=markers,
        #                          errorbar=None, linestyles="none", dodge=False)
        point_plot = sns.catplot(data=combined_table, y='Keypoints', x=metric, hue='Features',
                                 kind='point', palette=colors,
                                 height=text_width_inch/5, aspect=5/2,
                                 legend_out=add_legend, legend='auto', markers=markers,
                                 errorbar=None, linestyles="none", dodge=False)
        point_plot.ax.grid(True, 'both')
        # point_plot.set_xticklabels(rotation=60)

        # add the nnu-net baseline value
        nnu_error_value = nnu[metric].mean()
        # point_plot.refline(y=nnu_error_value, ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3], label='nnU-Net')
        point_plot.refline(x=nnu_error_value, ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3], label='nnU-Net')

        # y_label = f'mean {metric} [mm]' if not copd else f'relative {metric}'
        # point_plot.set_axis_labels(x_var='', y_var=y_label)
        x_label = f'mean {metric} [mm]' if not copd else f'relative {metric}'

        if not copd:
            x_tick_step = tick_steps[metric]
        else:
            x_tick_step = tick_step_copd

        point_plot.set(
            xlabel=x_label,
            ylabel='',
            #xticks=np.arange(np.round(x_min), np.round(x_max + x_tick_step), x_tick_step)
        )
        loc = ticker.MultipleLocator(base=x_tick_step)  # this locator puts ticks at regular intervals
        point_plot.ax.xaxis.set_major_locator(loc)
        point_plot.ax.xaxis.set_major_formatter(ticker.FuncFormatter(DecimalIfNecessaryFormatter()))

        # move axis tick labels closer to axis
        point_plot.ax.tick_params(axis='both', which='major', pad=0)

        # add legend with nnunet
        if add_legend:
            add_handles = [#Line2D([], [], linestyle='', label=''),
                           Line2D([], [], ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3], label='nnU-Net')]
            point_plot.fig.legend(handles=point_plot.legend.legendHandles + add_handles, #title='Features',
                                  loc='upper left', bbox_to_anchor=(0.77, 0.9))
            point_plot.legend.set_visible(False)

        save_fig(point_plot.fig, 'results/plots',
                 f'{model}_{metric}{"_copd" if copd else ""}_bvm{"_no_RHF" if exclude_rhf else ""}_horizontal',
                 pdf=True)

    plt.show()


class DecimalIfNecessaryFormatter:
    def __init__(self, decimal_places=1):
        self.decimal_places = decimal_places

    def __call__(self, x, pos):
        return '{:.{dec}f}'.format(x, dec=self.decimal_places) if int(x) != x else str(int(x))


def point_net_seg_table():
    seg_table('PointNet_seg', 'image')


def dgcnn_seg_table():
    seg_table('DGCNN_seg', None)


def pointtransformer_seg_table():
    seg_table('PointTransformer', None)


def nnunet_table(mode='surface_nodilate', cv=False, copd=False, exclude_rhf=False):
    assert mode in ['voxels', 'surface', 'surface_nodilate', 'subsample10000']

    if not copd:
        res_path = '../nnUNet/output/nnu_results/nnUNet/3d_fullres/Task503_FissuresTotalSeg/nnUNetTrainerV2_200ep__nnUNetPlansv2.1/'
    else:
        res_path = '../nnUNet/output/copd_pred'

    if not cv:
        file = f'cv_results_{mode}.csv'
        table = csv_to_df(os.path.join(res_path, file))
        if exclude_rhf:
            table = exclude_rhf_from_table(table)

        table['Fissure'] = table.index
    else:
        table = None
        for f in range(5):
            res_path_fold = os.path.join(res_path, f'fold_{f}', 'validation_mesh_reconstructions')
            file = f'test_results_{mode}.csv'
            fold_table = csv_to_df(os.path.join(res_path_fold, file))
            if exclude_rhf:
                fold_table = exclude_rhf_from_table(fold_table)
            fold_table['Fissure'] = fold_table.index
            if table is None:
                table = pd.DataFrame(columns=fold_table.columns)
            table.loc[f] = fold_table.loc['mean']
        print(table)

    return table


def v2m_table(exclude_rhf=False):
    res_path = '../voxel2mesh-master/resultsExperiment_003/cv_results.csv'
    table = csv_to_df(res_path)
    if exclude_rhf:
        table = exclude_rhf_from_table(table)
    table['Fissure'] = table.index
    return table


def model_comparison(exclude_rhf=False):
    dseg_tables = get_all_tables('DGCNN_seg', exclude_rhf=exclude_rhf)
    dseg_ae_tables = get_all_tables('DSEGAE_reg_aug_1024', exclude_rhf=exclude_rhf)
    dg_ssm_tables = get_all_tables('DG-SSM', exclude_rhf=exclude_rhf)
    tables = OrderedDict([
        ("DGCNN-Seg (cnn+image) + PSR", dseg_tables['cnn']['image']),
        ("DGCNN-Seg (cnn+image) + AE", dseg_ae_tables['cnn']['image']),
        ("GCN-SSM (cnn+image) (PC-to-mesh-SD)", dg_ssm_tables['cnn']['image']),
        # ("nnU-Net (SD)", nnunet_table('surface')),
        ("nnU-Net (label-to-mesh SD)", nnunet_table('voxels', exclude_rhf=exclude_rhf)),
        ("Voxel2Mesh", v2m_table(exclude_rhf=exclude_rhf))])

    colors = [mpl.cm.get_cmap('tab10').colors[2],
              mpl.cm.get_cmap('Dark2').colors[5],
              mpl.cm.get_cmap('Dark2').colors[2],
              mpl.cm.get_cmap('Dark2').colors[3],
              mpl.cm.get_cmap('Accent').colors[6]]

    comparative_bar_plot(tables, colors, rhf_excluded=exclude_rhf)

    # join dataframes for a latex table
    combined_table = None
    for model, table in tables.items():
        table = pm_table(table)
        table = table.drop(['Features', 'Keypoints'], errors="ignore", axis=1)
        table['Model'] = model
        if combined_table is not None:
            combined_table = pd.concat((combined_table, table))
        else:
            combined_table = table

    combined_table = combined_table.set_index(['Model', 'Fissure'])
    print(combined_table.to_latex(multirow=True, multicolumn=True))


def copd_comparison_table(model='DGCNN_seg', nnunet_mode="surface_nodilate"):
    dgcnn_tables = seg_table(model, exclude_rhf=True, pm=False)
    dgcnn_tables_copd = seg_table(model, copd=True, pm=False)
    nnu_table = nnunet_table(mode=nnunet_mode, exclude_rhf=True)
    nnu_table_copd = nnunet_table(mode=nnunet_mode, copd=True)

    nnu_table.insert(0,'Keypoints', "nnUNet")
    nnu_table.insert(1,'Features', "")
    nnu_table = nnu_table.set_index(['Keypoints', 'Features', 'Fissure'])

    nnu_table_copd.insert(0, 'Keypoints', "nnUNet")
    nnu_table_copd.insert(1, 'Features', "")
    nnu_table_copd = nnu_table_copd.set_index(['Keypoints', 'Features', 'Fissure'])

    concat_table = pd.concat([dgcnn_tables, nnu_table]).rename(columns={'ASSD_mean': 'ASSD', 'SDSD_mean': 'SDSD', 'HD_mean': 'HD'})
    concat_table_copd = pd.concat([dgcnn_tables_copd, nnu_table_copd]).rename(columns={'ASSD_mean': 'ASSD', 'SDSD_mean': 'SDSD', 'HD_mean': 'HD'})

    joint_table = concat_table.join(concat_table_copd, rsuffix='_copd')
    # exclude all but mean fissure
    # joint_table = joint_table[joint_table.index.droplevel([0,1]) == 'mean']

    for metric in ['ASSD', 'SDSD', 'HD']:
        # remove standard deviation col
        joint_table = joint_table.drop(columns=[f'{metric}_std', f'{metric}_std_copd'])

        # insert change factor
        column_index = joint_table.columns.get_loc(f'{metric}')
        joint_table.insert(column_index + 1, f'{metric}_change', joint_table[f'{metric}_copd'] / joint_table[f'{metric}'])

        # reorder columns
        col = joint_table.pop(f'{metric}_copd')
        joint_table.insert(column_index + 1, col.name, col)

    joint_table = joint_table.round(2)
    print(joint_table.to_latex(multirow=True, multicolumn=True))

    # slimmer version of the table
    joint_table_slim = joint_table.drop(columns=['ASSD', 'SDSD', 'HD', 'ASSD_copd', 'SDSD_copd', 'HD_copd', 'missing', 'missing_copd'])
    print()
    print(joint_table_slim.to_latex(multirow=True, multicolumn=True))

    # version of the table with only raw metrics
    joint_table_slim = joint_table.drop(columns=['ASSD_change', 'SDSD_change', 'HD_change', 'missing', 'missing_copd'])
    print()
    print(joint_table_slim.to_latex(multirow=True, multicolumn=True))
    return joint_table


def copd_change_table(model='DGCNN_seg'):
    combined_table = copd_comparison_table(model)
    combined_table = combined_table.drop(
        columns=['ASSD', 'SDSD', 'HD', 'ASSD_copd', 'SDSD_copd', 'HD_copd', 'missing', 'missing_copd'])
    combined_table = combined_table.rename(columns={'ASSD_change': 'ASSD', 'SDSD_change': 'SDSD', 'HD_change': 'HD'})
    combined_table = combined_table.reset_index()  # turn make index into columns

    # fix the spelling
    combined_table = combined_table.replace(FEATURE_MODES + ['cnn'], FEATURE_MODES_NORMALIZED + ['CNN'])
    combined_table = combined_table.replace(KP_MODES, KP_MODES_NORMALIZED)

    # use only mean fissure
    combined_table = combined_table[combined_table.Fissure == 'mean']

    # separate nnu and dgcnn-values out
    nnu_table = combined_table[combined_table.Keypoints == 'nnUNet']
    dgcnn_table = combined_table[combined_table.Keypoints != 'nnUNet']

    # fix the index (need ascending integers)
    combined_table = combined_table.set_index(np.arange(combined_table.shape[0]))
    return combined_table, nnu_table, dgcnn_table


def copd_relative_performance_plot(model='DGCNN_seg', presentation=True, add_nnu_value=True):
    combined_table, nnu_table, dgcnn_table = copd_change_table(model)

    print(combined_table)

    # plotting
    cmap = mpl.cm.get_cmap('tab10')
    if not presentation:
        feat_modes = FEATURE_MODES_NORMALIZED + ['CNN']
        colors = {feat: cmap(i / 10) for i, feat in enumerate(feat_modes)}
    else:
        plt.style.use("seaborn-v0_8-talk")
        feat_modes = ['Image', 'SSC', 'None']
        colors = {'SSC': cmap.colors[1], 'Image': cmap.colors[2], 'None': 'gray'}

    combined_table = combined_table.drop(combined_table[~combined_table.Features.isin(feat_modes)].index)
    sns.set_theme()

    print(combined_table)

    for metric in ['ASSD', 'SDSD', 'HD']:
        # swarm plot in categories
        bar_plot = sns.catplot(data=combined_table, x='Features', y=metric, col='Keypoints', hue='Features', kind='point', palette=colors,
                               height=SLIDE_HEIGHT_INCH * 0.5, aspect=2/3, legend_out=True, legend='auto')

        # add the nnu-net baseline value
        if add_nnu_value:
            nnu_error_value = nnu_table[metric].item()
            print(nnu_error_value)
            bar_plot.map(plt.axhline, y=nnu_error_value, ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3])

        bar_plot.set_axis_labels(x_var='', y_var=f'relative {metric}')
        if not presentation:
            bar_plot.set_xticklabels([])
        handles, labels = bar_plot.axes[-1][-1].get_legend_handles_labels()
        handles = handles + [Line2D([], [], linestyle=''),

                             Line2D([], [], ls='--', lw=1.5, c=mpl.cm.get_cmap('Dark2').colors[3], label='nnU-Net')]
        bar_plot.add_legend(title='Features:', handles=handles)
        bar_plot.set_titles('{col_name} KPs')

        save_fig(bar_plot.fig, 'results/plots', f'{model}_copd_relative_{metric}{"_presentation" if presentation else ""}{"_nnu" if add_nnu_value else ""}', pdf=not presentation, bbox_inches='')
    plt.show()


def feat_mode_normalizer(feat):
    return feat.lower().capitalize().replace('Cnn', 'CNN').replace('Nofeat', 'None').replace('Mind_ssc', 'SSC').replace('Mind', 'MIND').replace('Enhancement', 'Hessian')


def kp_mode_normalizer(kp):
    return kp.lower().replace('oe', 'ö').replace('enhancement', 'hessian').capitalize()


if __name__ == '__main__':
    KP_MODES.remove('noisy')
    KP_MODES_NORMALIZED = [kp_mode_normalizer(kp) for kp in KP_MODES]
    print(KP_MODES, KP_MODES_NORMALIZED)

    FEATURE_MODES.remove('cnn')
    FEATURE_MODES = FEATURE_MODES + ['nofeat']
    FEATURE_MODES_NORMALIZED = [feat_mode_normalizer(feat) for feat in FEATURE_MODES]

    # dgcnn_seg_table()
    # time_table()
    # point_net_seg_table()

    # bar_plot('DGCNN_seg', presentation=True)
    # bar_plot('DSEGAE_reg_aug_1024', presentation=True)
    # bar_plot_pointnet_vs_dgcnn(presentation=True)
    # seg_table('DGCNN', 'image')
    # model_comparison()
    # cross_val_swarm_plot("DGCNN_seg", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True)
    # cross_val_swarm_plot("DGCNN_seg", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True, copd=True)
    # seg_table("DGCNN_seg", copd=False, exclude_rhf=True)

    # cross_val_swarm_plot("DGCNN_seg", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True, exclude_rhf=True)
    # model_comparison(exclude_rhf=True)
    # copd_comparison_table()
    # copd_relative_performance_plot(presentation=True, add_nnu_value=True)

    # bvm_plot(copd=False)
    # bvm_plot(copd=True)

    # pointtransformer_seg_table()
    # bar_plot('PointTransformer', presentation=False)
    # cross_val_swarm_plot("PointTransformer", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True)
    # cross_val_swarm_plot("PointTransformer", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True, copd=True)
    # copd_relative_performance_plot('PointTransformer', presentation=True, add_nnu_value=True)
    copd_change_table('PointTransformer')

    # seg_table('PointNet_seg')
    # bar_plot('PointNet_seg', presentation=False)
    # cross_val_swarm_plot("PointNet_seg", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True)
    # cross_val_swarm_plot("PointNet_seg", presentation=True, use_median_instead_of_mean=False, add_nnu_value=True, copd=True)
    # copd_relative_performance_plot('PointNet_seg', presentation=True, add_nnu_value=True)
    copd_change_table('PointNet_seg')

    # bar_plot('DSEGAE_static', presentation=False)
    # bar_plot('DSEGAE_reg_aug_1024', presentation=False)
    # bar_plot('DSEGAE_n2048_k20_longer_train', presentation=False)
    # seg_table('DSEGAE_static')
    # seg_table('DSEGAE_reg_aug_1024')
    # seg_table('DSEGAE_n2048_k20_longer_train')

    copd_change_table("DGCNN_seg")
