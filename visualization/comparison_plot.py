import itertools
import pandas
import matplotlib.pyplot as plt

from global_config import home


def plot_quadruple_plot(df, prefixes, variable, titles):
    fig, ax = plt.subplots(1, 6, sharey='row', figsize=(25, 5))
    assert len(prefixes) == len(titles)
    indices = [x for x in itertools.product(range(2), repeat=2)]
    for i in range(len(prefixes)):
        sub_df = df[[column for column in df.columns if (variable in column) and (column.startswith(prefixes[i]))]]
        ax[i].boxplot(sub_df, labels=sub_df.columns)
        ax[i].set_title(titles[i])
        ax[i].tick_params('x', labelrotation=90)
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0.45)
    plt.savefig(f'{home}/results/graphs/hp_valid_{variable}_performance_comparison.png')
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
             f'{home}/results/hp_valid_performance.csv', f'{home}/results/shifted2_performance.csv',
             f'{home}/results/hp_shifted2_performance.csv', f'{home}/results/hp_shifted_valid_performance.csv']
    big_df = get_big_df(files)
    plot_quadruple_plot(big_df, ['m_', 'hp_m', 'hpv_m', 's2_m', 'hp_sm2', 'hpv_sm'], 'vel',
                        ['Initial performance', 'Initial high-pass performance',
                         'Initial high-pass validation performance', 'Shifted performance',
                         'Shifted high-pass performance', 'Shifted high-pass validation performance'])
