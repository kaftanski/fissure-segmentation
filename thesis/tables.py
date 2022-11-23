import os.path

import pandas as pd

from constants import KP_MODES, FEATURE_MODES


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


def dgcnn_seg_table():
    combined_table = None
    for kp in KP_MODES:
        if kp == 'cnn':
            cur_feat = FEATURE_MODES + ['cnn']
        else:
            cur_feat = FEATURE_MODES

        for feat in cur_feat:
            folder = os.path.join('results', f'DGCNN_seg_{kp}_{feat}')
            result_file = os.path.join(folder, 'cv_results.csv')
            if os.path.isfile(result_file):
                table = csv_to_df(result_file)
                table = pm_table(table)
                print(f"{kp}_{feat}")
                table['Fissure'] = [1, 2, 3, 'mean']
            else:
                table = pd.DataFrame(index=[0])
                print(f'missing experiment {kp}_{feat}')
                table['Fissure'] = None

            table['Keypoints'] = kp
            table['Features'] = feat

            if combined_table is None:
                combined_table = table
            else:
                combined_table = pd.concat((combined_table, table))

    # reorder columns
    # cols = list(combined_table.columns.values)
    # cols.insert(0, cols.pop(-1))
    # cols.insert(0, cols.pop(-1))
    # combined_table = combined_table[cols]
    combined_table = combined_table.set_index(['Fissure', 'Keypoints', 'Features'], drop=True)

    print(combined_table.to_latex(multirow=True, multicolumn=True))


if __name__ == '__main__':
    KP_MODES.remove('noisy')
    FEATURE_MODES.remove('cnn')
    FEATURE_MODES = FEATURE_MODES + ['no_feat']
    dgcnn_seg_table()
