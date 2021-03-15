import itertools
import pandas
import matplotlib.pyplot as plt

from global_config import home


def plot_single_plot(df, shifts, variable, title):
    sub_df = df[[column for column in df.columns if (variable in column)]]
    plt.boxplot(sub_df, labels=shifts)
    # plt.title(title)
    plt.tick_params('x', labelrotation=75, labelsize=12)
    plt.ylabel('Correlation coefficient')
    plt.xlabel('Shift with respect to receptive field centre (in milliseconds)')
    plt.tight_layout()

    plt.savefig(f'{home}/results/graphs/{variable}_hp_window_shifting_performance_comparison.png')
    plt.show()


def plot_quadruple_plot(df, prefixes, variable, titles):
    fig, ax = plt.subplots(2, 4, sharey='row', figsize=(19, 11))
    assert len(prefixes) == len(titles)
    indices = [x for x in itertools.product(range(2), repeat=2)]
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    for i in range(len(prefixes)):
        if (i == 0) or (i == 4):
            ax[indices[i]].set_ylabel('Correlation coefficient', size=12)
        sub_df = df[[column for column in df.columns if (variable in column) and (column.startswith(prefixes[i])) and ('k4' not in column)]]
        col_names = [col.replace(prefixes[i], '') for col in sub_df.columns]
        ax[indices[i]].boxplot(sub_df, labels=col_names)
        ax[indices[i]].set_title(titles[i])
        ax[indices[i]].tick_params('x', labelrotation=75, labelsize=12)
        ax[indices[i]].get_xticklabels()[2].set_color("red")
        if (i != len(prefixes)-1) and (i != 3):
            ax[indices[i]].get_xticklabels()[3].set_color("green")
        if (i == 1) or (i == 5):
            if indices[i][0] == 0:
                text = 'Initial'
            else:
                text = 'Shifted'
            ax[indices[i]].annotate(text, xy=(1.2, 1), xytext=(15, 20),
                        xycoords='axes fraction', textcoords='offset points',
                        size='15', ha='center', va='baseline')

    # plt.ylabel('Correlation coefficient')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6)
    plt.savefig(f'{home}/results/graphs/lpthp_valid_{variable}_performance_comparison.png')
    plt.show()


def get_big_df(files):
    big_df = pandas.DataFrame()
    for file in files:
        df = pandas.read_csv(file, sep=';', index_col=0)
        for column in df.columns:
            big_df[column] = df[column]
    return big_df


if __name__ == '__main__':
    files = [f'{home}/results/initial_performance.csv', f'{home}/results/hp_performance.csv',
             f'{home}/results/hp_valid_performance.csv', f'{home}/results/lpthv_performance.csv',
             f'{home}/results/shifted2_performance.csv',
             f'{home}/results/hp_shifted2_performance.csv', f'{home}/results/hp_shifted_valid_performance.csv',
             f'{home}/results/lpthv_shifted_performance.csv']
    # files = [f'{home}/results/initial_performance.csv', f'{home}/results/lp_performance.csv',
    #          f'{home}/results/shifted2_performance.csv', f'{home}/results/lp_shifted_performance.csv']
    # big_df = get_big_df(files)
    # plot_quadruple_plot(big_df, ['m_', 'hp_m_', 'hpv_m_', 'lpthv_m_', 's2_m_', 'hp_sm2_', 'hpv_sm_', 'lpthv_sm_'], 'vel',
    #                     ['Full training & validation', 'High-pass training & validation',
    #                      'Full training & high-pass validation',
    #                      'Low-pass training & high-pass validation', 'Full training & validation', 'High-pass training & validation',
    #                      'Full training & high-pass validation',
    #                      'Low-pass training & high-pass validation'])
    # plot_quadruple_plot(big_df, ['m_', 'lp_m_', 's2_m_', 'lp_sm_'], 'vel',
    #                     ['Initial performance', 'Initial low-pass validation performance',
    #                      'Shifted performance', 'Shifted low-pass validation performance'])
    shifts = [x for x in range(-250, 251, 25)] + [261]
    shifts.remove(261)
    shifts = [int(1000*(x/250)) for x in shifts]
    shift_df = pandas.read_csv(f'{home}/results/shifted_hp_window_performances.csv', sep=';', index_col=0)
    plot_single_plot(shift_df, shifts, 'absVel', None)
