import itertools
import pandas
import matplotlib.pyplot as plt
from scipy.stats import binom_test, wilcoxon

from global_config import home


def plot_single_plot(df, shifts, variable, title):
    sub_df = df[[column for column in df.columns if (variable in column)]]
    plt.boxplot(sub_df, labels=shifts)
    # plt.title(title)
    plt.tick_params('x', labelrotation=75, labelsize=12)
    plt.ylabel('Correlation coefficient')
    plt.xlabel('Shift with respect to receptive field centre (in milliseconds)')
    plt.tight_layout()

    plt.savefig(f'{home}/results/test_results/graphs/{variable}_performance_comparison.png')
    plt.show()


def get_statistical_significance(random_correlations, correlations):
    success = [1 if corr > random_corr else 0 for corr, random_corr in zip(correlations, random_correlations)]
    # p = binom_test(x=sum(success), n=len(success), p=0.05, alternative='greater')
    value, p = wilcoxon(correlations, random_correlations,  alternative='greater')
    string = ''
    if p < 0.05:
        string = ' *'
    if p < 0.01:
        string = ' **'
    return string


def plot_quadruple_plot(df, prefixes, variable, titles, chance_level):
    fig, ax = plt.subplots(2, int(len(titles)), sharey='row', figsize=(len(titles)*4, int(len(titles))*2))
    assert len(prefixes) == int(len(titles)*2)
    indices = [x for x in itertools.product([0, 1], [x for x in range(int(len(titles)))])]
    # indices = [x for x in range(len(titles))]
    # indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    for i in range(len(prefixes)):
        if (i == 0) or (i == int(len(titles))):
            ax[indices[i]].set_ylabel('Correlation coefficient', size=12)
        sub_df = df[[column for column in df.columns if (variable in column) and (column.startswith(prefixes[i])) and ('k4' not in column) and ('sbp1' not in column)]]
        columns = [x.replace('k_k', 'k') if 'k_k' in x else x for x in sub_df.columns]
        chance_columns = [x.replace('k_k', 'k') if 'k_k' in x else x for x in chance_level.columns]
        columns = [x.replace('k1_d3', 'k1') if 'k1_d3' in x else x for x in columns]
        chance_columns = [x.replace('k1_d3', 'k1') if 'k1_d3' in x else x for x in chance_columns]

        sub_df.columns = columns
        chance_level.columns = chance_columns
        col_names = []
        print(prefixes[i])
        for col in sub_df.columns:
            col = col.replace(prefixes[i], '')
            if 'k3_d3' in col:
                chance_level_col = f'shuffled_m_{variable}_k3_d3_sbp0'
            else:
                chance_level_col = f'shuffled_m_{col}'
            significance_string = get_statistical_significance(chance_level[chance_level_col].tolist(), sub_df[f'{prefixes[i]}{col}'].tolist())
            col = f'{col}{significance_string}'
            col_names.append(col)
        # col_names = [col.replace(prefixes[i], '') for col in sub_df.columns]
        ax[indices[i]].boxplot(sub_df, labels=col_names)
        ax[indices[i]].set_title(titles[i%int(len(titles))])
        ax[indices[i]].tick_params('x', labelrotation=75, labelsize=12)
        ax[indices[i]].get_xticklabels()[2].set_color("red")
        ax[indices[i]].grid(which='major', axis='y', linestyle='--')
        if any("sbp1" in s for s in sub_df.columns):
            ax[indices[i]].get_xticklabels()[3].set_color("green")
        if (i == 1) or (i == int(len(titles))+1):
            if indices[i][0] == 0:
                text = 'Non-shifted'
            else:
                text = 'Shifted'
            ax[indices[i]].annotate(text, xy=(1.2, 1), xytext=(15, 20),
                        xycoords='axes fraction', textcoords='offset points',
                        size='15', ha='center', va='baseline')

    # plt.ylabel('Correlation coefficient')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.92)
    plt.savefig(f'{home}/results/test_results/graphs/shortened_{variable}_performance_comparison.png')
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
                     'w0_high-high_s0',  'w1_high-high_s0', 'w0_high-high_s1', 'w1_high-high_s1',
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
    plt.grid(which='major', axis='y', linestyle='- -',)
    plt.tight_layout()
    plt.savefig(f'{home}/results/test_results/graphs/{variable}_one_file_comparison.png')
    plt.show()


if __name__ == '__main__':
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
    chance_level_df = pandas.read_csv(f'{home}/results/test_results/random_performances.csv', sep=';', index_col=0)
    big_df = get_big_df(files1)
    plot_quadruple_plot(big_df, ['m_', 'lpv_m_', 'hp_m_', 'hpv_m_', 'hpv_lpt_strong_m_', 'sm_', 'lp_sm_', 'hp_sm_', 'hpv_sm_', 'hpv_lpt_strong_sm_'],
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
    #                     chance_level_df)
    # shifts = [x for x in range(-250, 251, 25)] + [261]
    # shifts.remove(261)
    # shifts = [int(1000*(x/250)) for x in shifts]
    # shift_df = pandas.read_csv(f'{home}/results/test_results/complete_performances.csv', sep=';', index_col=0)
    # plot_single_plot(shift_df, shifts, 'absVel', None)
