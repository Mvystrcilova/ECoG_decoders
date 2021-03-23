import itertools
import pandas
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

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
    value, p = wilcoxon(correlations, random_correlations, alternative='greater')
    string = ''
    if p < 0.05:
        string = ' *'
    if p < 0.01:
        string = ' **'
    return string


def plot_quadruple_plot(df, prefixes, variable, titles, chance_level):
    fig, ax = plt.subplots(2, len(titles), sharey='row', figsize=(len(titles)*4.5, int(len(titles)*1.7)))
    assert len(prefixes) == int(len(titles)*2)
    indices = [x for x in itertools.product([0, 1], [x for x in range(int(len(titles)))])]
    # indices = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    for i in range(len(prefixes)):
        if (i == 0) or (i == int(len(titles))):
            ax[indices[i]].set_ylabel('Correlation coefficient', size=12)
        sub_df = df[[column for column in df.columns if (variable in column) and (column.startswith(prefixes[i])) and ('k4' not in column)]]
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
            if 'sbp1' not in col:
                chance_level_col = f'shuffled_m_{col}'
            else:
                chance_level_col = f'shuffled_m_{variable}_k3_d3_sbp0'
            significance_string = get_statistical_significance(chance_level[chance_level_col].tolist(), sub_df[f'{prefixes[i]}{col}'].tolist())
            col = f'{col}{significance_string}'
            col_names.append(col)
        # col_names = [col.replace(prefixes[i], '') for col in sub_df.columns]
        ax[indices[i]].boxplot(sub_df, labels=col_names)
        ax[indices[i]].set_title(titles[i%int(len(titles))])
        ax[indices[i]].tick_params('x', labelrotation=75, labelsize=12)
        ax[indices[i]].get_xticklabels()[2].set_color("red")
        if any("sbp1" in s for s in sub_df.columns):
            ax[indices[i]].get_xticklabels()[3].set_color("green")
        if (i == 2) or (i == int(len(titles))+2):
            if indices[i][0] == 0:
                text = 'Non-shifted'
            else:
                text = 'Shifted'
            ax[indices[i]].annotate(text, xy=(1.2, 1), xytext=(15, 20),
                        xycoords='axes fraction', textcoords='offset points',
                        size='15', ha='center', va='baseline')

    # plt.ylabel('Correlation coefficient')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)
    plt.savefig(f'{home}/results/test_results/graphs/lpthp_valid_{variable}_performance_comparison.png')
    plt.show()


def get_big_df(files):
    big_df = pandas.DataFrame()
    for file in files:
        df = pandas.read_csv(file, sep=';', index_col=0)
        for column in df.columns:
            big_df[column] = df[column]
    return big_df


if __name__ == '__main__':
    files = [f'{home}/results/test_results/initial_performances.csv', f'{home}/results/test_results/hp3_performances.csv',
             f'{home}/results/test_results/hp_performances.csv', f'{home}/results/test_results/hpv3_performances.csv',
             f'{home}/results/test_results/hp_valid_performances.csv', f'{home}/results/test_results/lpt_hpv_performances.csv',
             f'{home}/results/test_results/lpt_hpv_strong_performances.csv',
             f'{home}/results/test_results/shifted_performances.csv', f'{home}/results/test_results/hp3_shifted_performances.csv',
             f'{home}/results/test_results/hp_shifted_performances.csv', f'{home}/results/test_results/hpv3_shifted_performances.csv',
             f'{home}/results/test_results/hp_valid_shifted_performances.csv',
             f'{home}/results/test_results/lpt_hpv_shifted_performances.csv',
             f'{home}/results/test_results/lpt_hpv_strong_shifted_performances.csv']
    # files = [f'{home}/results/initial_performance.csv', f'{home}/results/lp_performance.csv',
    #          f'{home}/results/shifted2_performance.csv', f'{home}/results/lp_shifted_performance.csv']
    chance_level_df = pandas.read_csv(f'{home}/results/test_results/random_performances.csv', sep=';', index_col=0)
    big_df = get_big_df(files)
    plot_quadruple_plot(big_df, ['m_', 'hp3_m_', 'hp_m_', 'hpv3_m_', 'hpv_m_', 'hpv_lpt_m_', 'hpv_lpt_strong_m_', 'sm_', 'hp3_sm_', 'hp_sm_', 'hpv3_sm_', 'hpv_sm_', 'hpv_lpt_sm_', 'hpv_lpt_strong_sm_'],
                        'vel',
                        ['Full training & validation', 'High-pass training & validation \n3rd order Butterworth',
                         'High-pass training & validation \n15th order Butterworth',
                         'Full training & high-pass validation \n3rd order Butterworth',
                         'Full training & high-pass validation \n15 order Butterworth',
                         'Low-pass training & high-pass validation \n3rd order Butterworth',
                         'Low-pass training & high-pass validation \n15 order Butterworth'], chance_level_df)
    # plot_quadruple_plot(big_df, ['m_', 'lp_m_', 's2_m_', 'lp_sm_'], 'vel',
    #                     ['Non-shifted performance', 'Non-shifted low-pass validation performance',
    #                      'Shifted performance', 'Shifted low-pass validation performance'])
    shifts = [x for x in range(-250, 251, 25)] + [261]
    # shifts.remove(261)
    # shifts = [int(1000*(x/250)) for x in shifts]
    # shift_df = pandas.read_csv(f'{home}/results/test_results/complete_performances.csv', sep=';', index_col=0)
    # plot_single_plot(shift_df, shifts, 'absVel', None)
