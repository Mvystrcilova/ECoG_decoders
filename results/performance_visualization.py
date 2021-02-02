import matplotlib.pyplot as plt
import seaborn as sns
from global_config import home
import numpy as np
import pandas


def load_results_file(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    results = {}
    for i, line in enumerate(lines):
        if i % 2 == 1:
            res_line = line.split(':')
            key = res_line[0].split(' ')[0]
            value = [float(x) for x in res_line[1].split(';')]
            results[key] = value
    x_ticks = ['speed initial model', 'speed dilation 1', 'speed dilation i^2',
               'velocity initial model', 'velocity dilation 1', 'velocity dilation i^2']
    df = pandas.DataFrame(data=results)
    # df.columns = x_ticks
    return df


def plot_df_boxplot(df, saving_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.boxplot(x=df, labels=df.columns)
    plt.xticks(rotation=60)
    plt.tight_layout()
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()


if __name__ == '__main__':
    vel_results = load_results_file(home + '/outputs/avg_best_results.txt')
    absVel_results = load_results_file(home + '/outputs/absVel_avg_best_results.txt')
    plot_df_boxplot(vel_results, home + '/results/vel_performance_graph.png')
    plot_df_boxplot(absVel_results, home + '/results/abs_vel_performance_graph.png')
