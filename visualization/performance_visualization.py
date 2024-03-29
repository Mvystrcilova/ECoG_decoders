import matplotlib.pyplot as plt
import seaborn as sns
from global_config import home
import numpy as np
import pandas


def load_results_file(file_path):
    """
    Function reading deprecated result saving formats
    """
    file = open(file_path, 'r')
    lines = file.readlines()
    results = {}
    for i, line in enumerate(lines):
        if i % 2 == 1:
            res_line = line.split(':')
            key = res_line[0].split(' ')[0]
            value = [float(x) for x in res_line[1].split(';')]
            results[key] = value

    df = pandas.DataFrame(data=results)
    # df.columns = x_ticks
    return df


def plot_df_boxplot(df, title, saving_path=None):
    """
    Shows and potentialy saves a boxplot from a pandas.DataFrame
    :param df: pandas.DataFrame with values to be plotted
    :param title: graph title
    :param saving_path: path where to save the graph
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.boxplot(x=df, labels=df.columns)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()


if __name__ == '__main__':
    # vel_results = load_results_file(home + '/outputs/vel_avg_best_results.txt')
    # absVel_results = load_results_file(home + '/outputs/absVel_avg_best_results.txt')
    # plot_df_boxplot(vel_results, home + '/results/vel_performance_graph.png')
    # plot_df_boxplot(absVel_results, home + '/results/abs_vel_performance_graph.png')
    """
    Code to visualize preliminary results
    """
    df = pandas.read_csv(f'{home}/results/hp_performance.csv', sep=';', index_col=0)
    plot_df_boxplot(df, 'high-pass performance')
    df2 = pandas.read_csv(f'{home}/results/hp_shifted2_performance.csv', sep=';', index_col=0)
    plot_df_boxplot(df2, 'high-pass shifted performance')
    df3 = pandas.read_csv(f'{home}/results/shifted2_performance.csv', sep=';', index_col=0)
    plot_df_boxplot(df3, 'shifted_performance')