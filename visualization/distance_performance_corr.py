import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas
from global_config import input_time_length, home
from layer_passes import get_num_of_predictions


def get_kernels_from_name(name):
    kernels, dilations = None, None

    if 'k3' in name:
        kernels = [3, 3, 3, 3]
    elif 'k2' in name:
        kernels = [2, 2, 2, 2]
    elif 'k1' in name:
        kernels = [1, 1, 1, 1]
    elif 'k4' in name:
        kernels = [4, 4, 4, 4]
    if ('d1' in name) or ('d' not in name):
        dilations = [1, 1, 1, 1]
    elif 'd2' in name:
        dilations = [2, 4, 8, 16]
    elif ('d3' in name):
        dilations = [3, 9, 27, 81]
    assert kernels is not None
    assert dilations is not None
    return kernels, dilations


def get_distance_to_performance(df, variable):
    distance_performance_dict = {}
    for column in df.columns:
        if ('Unnamed' not in column) and (variable in column):
            print(column)
            kernels, dilations = get_kernels_from_name(column)
            max_k, _ = get_num_of_predictions(kernels, dilations)
            distance = int((input_time_length - max_k + 1)/2)
            if 'sbp0' in column:
                distance = int(522/2)
            performance = df[column].to_numpy()
            distance_performance_dict[distance] = [np.mean(performance), scipy.stats.sem(performance)]
    ordered_distance_performance_dict = collections.OrderedDict(sorted(distance_performance_dict.items()))

    return ordered_distance_performance_dict


def get_nums_from_dicts(distance_performance_dict):
    ys = list(distance_performance_dict.keys())
    ys = [y/250 for y in ys]
    xs = [l[0] for l in list(distance_performance_dict.values())]
    sem = [l[1] for l in list(distance_performance_dict.values())]
    return ys, xs, sem


def plot_distance_peformance(distance_performance_dicts, labels, title, output_file=None):
    colors = ['steelblue', 'orange', 'limegreen', 'hotpink', 'turquoise', 'gold', 'hotpink']
    sem_colors = ['lightsteelblue', 'navajowhite', 'mistyrose', 'honeydew', 'lemonchiffon', 'azure', 'lavenderblush']
    plt.figure(figsize=(12, 6))
    for i, dp_dict in enumerate(distance_performance_dicts):
        ys, xs, sem = get_nums_from_dicts(dp_dict)
        plt.plot(ys, xs, label=labels[i], color=colors[i])
        plt.fill_between(ys, [x-s for x, s in zip(xs, sem)], [x+s for x, s in zip(xs, sem)], alpha=0.15)
    plt.plot([314/250, 314/250], [0, 0.8], color='red', label='Deep4Net sbp1 window size')
    plt.plot([261/250, 261/250], [0, 0.8], color='green', label='Deep4Net sbp0 window size')

    plt.xlabel('Half of the input window size giving one prediction (in seconds)', size='12')
    plt.ylabel('Correlation coefficient', size='12')
    # plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    files = [f'{home}/results/test_results/initial_performances.csv', f'{home}/results/test_results/hp_performances.csv',
             f'{home}/results/test_results/hp_valid_performances.csv', f'{home}/results/test_results/lp_valid_performances.csv',
             f'{home}/results/test_results/lpt_hpv_performances.csv']
    # files = [f'{home}/results/test_results/shifted_performances.csv', f'{home}/results/test_results/hp_shifted_performances.csv',
    #          f'{home}/results/test_results/hp_valid_shifted_performances.csv', f'{home}/results/test_results/lpt_hpv_shifted_performances.csv',
    #          f'{home}/results/test_results/lpt_hpv_shifted_performances.csv']
    dicts = []
    variable = 'absVel'
    for file in files:
        df = pandas.read_csv(file, sep=';', index_col=[0])
        df_dict = get_distance_to_performance(df, variable)
        dicts.append(df_dict)
    title = f'Non-shifted {variable}'
    output_file = f'{home}/results/test_results/graphs/distance_performance_{variable}.png'
    plot_distance_peformance(dicts, ['Full train & valid', 'Hp train & valid', 'Full train & hp valid',
                                     'Full train & lp valid', 'Lp train & hp valid'], title, output_file)





