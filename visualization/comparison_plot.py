import itertools
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from global_config import home
from visualization.fold_result_visualization import get_5_fold_performance_df, get_statistical_significance


def plot_single_plot(df, shifts, variable):
    sub_df = df[[column for column in df.columns if (variable in column)]]
    for column in sub_df.columns:
        sub_df.loc[:, column] = sub_df[column].astype(float)
    sns.boxplot(data=sub_df, palette="coolwarm")
    ticks, labels = plt.xticks()
    plt.xticks(ticks=ticks, labels=shifts)
    plt.tick_params('y', labelsize=13)
    plt.tick_params('x', labelrotation=75, labelsize=13)
    plt.ylabel('Correlation coefficient', size=13)
    plt.xlabel('Shift with respect to receptive field centre (in milliseconds)', size=13)
    plt.grid()
    plt.tight_layout()

    plt.savefig(f'{home}/results/test_results/graphs/{variable}_performance_comparison.png')
    plt.show()




def plot_quadruple_plot(df: pandas.DataFrame, prefixes: list, variable: str, titles: list,
                        chance_level: pandas.DataFrame, file_prefix: str):
    """
    Function to plotting boxplots for columns in the df.
    :param df: pandas.DataFrame where the columns hold average performances for each of the
    12 patients for the different models
    :param prefixes: prefixes of the models which are to be plotted
    :param variable: 'vel' or 'absVel' based on if we want to plot results for velocity or absolute velocity
    :param titles: 'Titles of the subplots'
    :param chance_level: a pandas.DataFrame containing chance-level decoding values for
    :param file_prefix: prefix under which the plot should be saved
    :return: None
    """
    fig, ax = plt.subplots(2, 5, sharey='row', figsize=(len(titles) * 4 , (int(len(titles)*2)+6)),
                           gridspec_kw={'width_ratios': [1, 1, 0.1, 1, 1]})
    # assert len(prefixes) == int(len(titles))
    # indices = [x for x in itertools.product([0,1], [x for x in range(int(len(titles)))])]
    # indices = [x for x in range(len(prefixes))]
    letters = ['A', 'B', 'C', 'C', 'D', 'E', 'F', 'G', 'G', 'H']
    # letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    columns = [x.replace('k_k', 'k') if 'k_k' in x else x for x in df.columns]
    chance_columns = [x.replace('k_k', 'k') if 'k_k' in x else x for x in chance_level.columns]
    columns = [x.replace('k1_d3', 'k1') if 'k1_d3' in x else x for x in columns]
    chance_columns = [x.replace('k1_d3', 'k1') if 'k1_d3' in x else x for x in chance_columns]
    df.columns = columns
    chance_level.columns = chance_columns
    for i in range(len(prefixes)):
        if (i == 2) or (i == 7):
            ax[indices[i]].set_visible(False)
            continue
        if (i == 0) or (i == 5):
            ax[indices[i]].set_ylabel('Correlation coefficient', size=30)
        sub_df = df[[column for column in df.columns if
                     (variable in column) and (column.startswith(prefixes[i])) and ('k4' not in column) and (
                             'sbp1' not in column)]]
        col_names = []
        print(prefixes[i])
        for col in sub_df.columns:
            switch_back = False
            col = col.replace(prefixes[i], '')

            if (prefixes[i] == 'm_') or (prefixes[i] == 'sm_'):
                chance_level_col = f'{prefixes[i]}{variable}_k3_d3'
                chance_level_values = sub_df[chance_level_col]
            elif (prefixes[i] == 'lp_m_') or (prefixes[i] == 'lp_sm_'):
                chance_level_col = '{}_{}'.format(prefixes[i].split('_')[1], col)
                chance_level_values = df[chance_level_col]
            # if 'pw_' in prefixes[i]:
            #     chance_level_col = prefixes[i].replace('pw_', '')
            else:
                if 'k3_d3' in col:
                    chance_level_col = f'shuffled_m_{variable}_k3_d3'
                else:
                    chance_level_col = f'shuffled_{prefixes[i]}{col}'
                    chance_level_values = chance_level[chance_level_col]
            # chance_level_col = prefixes[i].replace('pw_', '')
            # chance_level_values = df[f'{chance_level_col}{col}']

            # if ((prefixes[i] != 'm_') and (prefixes[i] != 'sm_')) or (
            #         f'{prefixes[i]}{col}' != f'{prefixes[i]}{variable}_k3_d3'):
            #     if (prefixes[i] == 'lp_m_') or (prefixes[i] == 'lp_sm_'):
            #         alternative = 'less'
            #     else:
            #         alternative = 'greater'
                # if switch_back:
                #     prefixes[i] = f'pw_{prefixes[i]}'
            if ('h' in prefixes[i]) or (prefixes[i] == 'sm_') or (prefixes[i] == 'm_'):
                alternative = 'greater'
            else:
                alternative = 'less'
            # if 'pw_' not in prefixes[i]:
            if ('k3_d3' in col) and ((prefixes[i] == 'm_') or (prefixes[i] == 'sm_')):
                significance_string = ''

            else:
                print(prefixes[i])
                significance_string = get_statistical_significance(chance_level_values.tolist(),
                                                                   sub_df[f'{prefixes[i]}{col}'].tolist(),
                                                                   alternative=alternative)
            if 'k3_d3' in col:
                col = f'{col}_sbp0'
            col = f'{col}{significance_string}'
            col_names.append(col)
        # col_names = [col.replace(prefixes[i], '') for col in sub_df.columns]
        ax[indices[i]].axhline(0, color='k', linestyle='--')
        # ax[indices[i]].text(-0.2, 1.05, letters[i],
        #                     size=30, weight='bold')
        if indices[i][0] == 0:
            ax[indices[i]].text(-0.2, 1.0, letters[i],
                                size=30, weight='bold')
        else:
            ax[indices[i]].text(-0.2, 0.69, letters[i],
                                size=30, weight='bold')
        # ax[indices[i]].boxplot(sub_df, labels=col_names)
        sns.boxplot(data=sub_df, ax=ax[indices[i]])
        ax[indices[i]].set_title(titles[i % int(len(titles))], size=30, loc='center', pad=10)
        ax[indices[i]].tick_params('x', labelrotation=75, labelsize=30)
        ax[indices[i]].tick_params('y', labelsize=30)
        ax[indices[i]].set_xticklabels(labels=col_names)
        ax[indices[i]].get_xticklabels()[2].set_color("red")
        ax[indices[i]].grid(which='major', axis='y', linestyle='--')
        if any("sbp1" in s for s in sub_df.columns):
            ax[indices[i]].get_xticklabels()[3].set_color("green")
        # if (i == 1) or (i == int(len(titles))+1):
        # if indices[i] == 0:
        #     text = 'Non-shifted'
        # else:
        #     text = 'Shifted'
        # ax[indices[i]].annotate(text, xy=(1.2, 1), xytext=(15, 20),
        #             xycoords='axes fraction', textcoords='offset points',
        #             size='15', ha='center', va='baseline')

    # plt.ylabel('Correlation coefficient')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.86)
    plt.savefig(
        f'{home}/results/performances_5_results/graphs/{file_prefix}_{variable}_performance_comparison.pdf', dpi=200)
    plt.show()


def get_big_df(files, dir=None):
    if dir is None:
        dir = {}
    big_df = pandas.DataFrame()
    for file in files:
        df = pandas.read_csv(file, sep=';', index_col=0)
        matched_prefixes = [d for d in dir.keys() if d in file]
        assert len(matched_prefixes) < 2
        if len(matched_prefixes) == 1:
            prefix = dir[matched_prefixes[0]]
        else:
            prefix = ''
        for column in df.columns:
            big_df[f'{prefix}{column}'] = df[column]
    return big_df


def plot_one_file_results(file_df, file, variable):
    sub_df = file_df[[column for column in file_df.columns if variable in column]]
    columns = [x.replace('k_k', 'k') if 'k_k' in x else x for x in sub_df.columns]

    columns = [x.replace('hpv_lpt_strong_m', 'hpv_lpt_m') for x in columns]
    columns = [x.replace('hpv_lpt_strong_sm', 'hpv_lpt_sm') for x in columns]

    # columns = [x.replace('k1_d3', 'k1') if 'k1_d3' in x else x for x in columns]
    sub_df.columns = columns
    col_order = ['m', 'pw_m', 'sm', 'pw_sm',
                 'hp_m', 'hp_sm', 'pw_hp_m', 'pw_hp_sm',
                 'hpv_m', 'hpv_sm', 'hpv_lpt_m',
                 'hpv_lpt_sm', 'shuffled_m']
    new_col_names = ['w0_full-full_s0', 'w1_full-full_s0', 'w0_full-full_s1', 'w1_full-full_s1',
                     'w0_high-high_s0', 'w1_high-high_s0', 'w0_high-high_s1', 'w1_high-high_s1',
                     'w0_full-high_s0', 'w0_full-high_s1', 'w0_low-high_s0', 'w0_low-high_s1', 'shuffled']
    sub_df = sub_df[[f'{col}_{variable}_{file}' for col in col_order]]
    sub_df = sub_df.loc[:, ~sub_df.columns.duplicated()]
    significant_cols = []
    for i, col in enumerate(sub_df.columns[:-1]):
        str = get_statistical_significance(sub_df[f'shuffled_m_{variable}_{file}'], sub_df[col])
        significant_cols.append(f'{new_col_names[i]}{str}')
    significant_cols.append(f'shuffled_m')
    # sub_df.columns = significant_cols
    plt.boxplot(sub_df, labels=significant_cols)
    plt.ylabel('Correlation coefficient')
    plt.tick_params('x', labelrotation=75, labelsize=12)
    plt.grid(which='major', axis='y', linestyle='- -', )
    plt.tight_layout()
    plt.savefig(f'{home}/results/test_results/graphs/{variable}_one_file_comparison.png')
    plt.show()


if __name__ == '__main__':

    variable = 'absVel'
    # prefixes = ['m_', 'pw_m_', 'lp_m_', 'pw_lp_m_', 'hp_m_', 'pw_hp_m_', 'hpv_m_', 'pw_hpv_m_']
    prefixes = ['m_', 'sm_', 'lp_m_', 'lp_sm_', 'hp_m_', 'hp_sm_', 'hpv_m_', 'hpv_sm_']

    prefixes2 = ['m_', 'lp_m_', 'hp_m_', 'hpv_m_', 'lpt_hpv_m_']
    titles = ['Full training & validation', 'Full training & low-pass \nvalidation',
              'High-pass training & validation \n15th order Butterworth',
              'Full training & high-pass \n validation 15 order Butterworth',
              'Low-pass training & high-pass \n validation 15 order Butterworth']
    titles2 = ['Full training & validation',
               'Whitened- full training & validation',
               '',
               'Full training & low-pass validation\n15 order Butterworth',
               'Whitened - full training & low-pass validation\n15 order Butterworth',
               'High-pass training & validation\n15 order Butterworth',
               'Whitened - high-pass training & validation\n15 order Butterworth',
               '',
               'Full training & high-pass validation\n15 order Butterworth',
               'Whitened - full training & high-pass validation\n15 order Butterworth'
               ]
    titles_s = ['Full training & validation',
               'Shifted - full training & validation',
               '',
               'Full training & low-pass validation\n15 order Butterworth',
               'Shifted - full training & low-pass validation\n15 order Butterworth',
               'High-pass training & validation\n15 order Butterworth',
               'Shifted - high-pass training & validation\n15 order Butterworth',
               '',
               'Full training & high-pass validation\n15 order Butterworth',
               'Shifted - full training & high-pass validation\n15 order Butterworth'
               ]
    big_df = get_5_fold_performance_df(variable, prefixes)
    chance_level_df = get_5_fold_performance_df(variable, prefixes=[f'shuffled_{prefix}' for prefix in prefixes if
                                                                    (prefix != 'lp_m_') and (prefix != 'lp_sm_') and (not 'pw_' in prefix)])
    # chance_level_df = pandas.read_csv(f'{home}/results/test_results/random_performances.csv', sep=';', index_col=0)
    # plot_quadruple_plot(big_df, ['m_', 'lp_m_', 'hp_m_', 'hpv_m_', 'lpt_hpv_m_'], 'vel', ['Full training & validation',
    #                                                                                       'Full training & low-pass validation\n15th order Butterworth',
    #                                                                                       'High-pass training & validation \n15th order Butterworth',
    #                                                                                       'Full training & high-pass validation\n15 order Butterworth',
    #                                                                                       'Low-pass training & high-pass validation \n15 order Butterworth',
    #                                                                                      ], chance_level_df)

    # prefixes = ['m_', 'pw_m_', '', 'lp_m_', 'pw_lp_m_', 'hp_m_', 'pw_hp_m_', '', 'hpv_m_', 'pw_hpv_m_']
    prefixes = ['m_', 'sm_', '', 'lp_m_', 'lp_sm_', 'hp_m_', 'hp_sm_', '', 'hpv_m_', 'hpv_sm_']
    # prefixes = [f'pw_{prefix}' if prefix != '' else '' for prefix in prefixes]
    plot_quadruple_plot(big_df, prefixes, variable, titles_s, chance_level_df, 'original_vs_shifted')
    """
    
    files1 = [f'{home}/results/test_results/initial_performances.csv',
              f'{home}/results/test_results/hp_performances.csv',
              f'{home}/results/test_results/hp_valid_performances.csv',
              f'{home}/results/test_results/lp_valid_performances.csv',
              f'{home}/results/test_results/lp_shifted_performances.csv',
              f'{home}/results/test_results/lpt_hpv_performances.csv',
              f'{home}/results/test_results/lpt_hpv_strong_performances.csv',
              f'{home}/results/test_results/shifted_performances.csv',
              f'{home}/results/test_results/hp_shifted_performances.csv',
              f'{home}/results/test_results/hp_valid_shifted_performances.csv',
              f'{home}/results/test_results/lpt_hpv_strong_shifted_performances.csv']
    # files = [f'{home}/results/initial_performance.csv', f'{home}/results/lp_performance.csv',
    #          f'{home}/results/shifted2_performance.csv', f'{home}/results/lp_shifted_performance.csv']
    big_df = get_big_df(files1)
    plot_quadruple_plot(big_df,
                        ['m_', 'lpv_m_', 'hp_m_', 'hpv_m_', 'hpv_lpt_strong_m_', 'sm_', 'lp_sm_', 'hp_sm_', 'hpv_sm_',
                         'hpv_lpt_strong_sm_'],
                        'absVel',
                        ['Full training & validation', 'Low-pass training & full validation',
                         'High-pass training & validation \n15th order Butterworth',
                         'Full training & high-pass validation \n15 order Butterworth',
                         'Low-pass training & high-pass validation \n15 order Butterworth'], chance_level_df)
    files2 = [f'{home}/results/test_results/initial_performances.csv',
              f'{home}/results/pre_whitened/initial_performances.csv',
              f'{home}/results/test_results/initial_shifted_performances.csv',
              f'{home}/results/pre_whitened/initial_shifted_performances.csv',
              f'{home}/results/test_results/hp_performances.csv',
              f'{home}/results/pre_whitened/hp_performances.csv',
              f'{home}/results/test_results/hp_shifted_performances.csv',
              f'{home}/results/pre_whitened/hp_shifted_performances.csv',
              f'{home}/results/test_results/random_performances.csv']
    # files = files1 + files2
    file = 'k3_d3_sbp0'
    big_df = get_big_df(files2, dir={'pre_whitened': 'pw_'})
    file_df = big_df[[column for column in big_df.columns if file in column]]
    plot_one_file_results(file_df, file, variable='absVel')

    # plot_quadruple_plot(big_df, ['m_', 'pw_m_', 'sm_', 'pw_sm_', 'hp_m_', 'pw_hp_m_', 'hp_sm_', 'pw_hp_sm_'], 'vel',
    #                     ['Non-shifted full training & validation', 'Non-shifted pre-whitened full training & validation',
    #                      'Shifted full training & validation', 'Shifted pre-whitened full training & validation',
    #                      'Non-shifted high-pass training & validation \n15th order Butterworth',
    #                      'Non-shifted pre-whitened high-pass training & validation \n15th order Butterworth',
    #                      'Shifted high-pass training & validation\n15th order Butterworth',
    #                      'Shifted pre-whitened high-pass training & validation\n15th order Butterworth'],
    #                     chance_level_df)"""

    # shifts = [x for x in range(-250, 251, 25)] + [261]
    # shifts.remove(261)
    # shifts = [int(1000*(x/250)) for x in shifts]
    # shift_df = pandas.read_csv(f'{home}/results/shifted_window_performances.csv', sep=';', index_col=0)
    # shift_df['vel_shift_by_261'] = big_df['m_vel_k3_d3']
    # plot_single_plot(shift_df, shifts, 'absVel', None)
