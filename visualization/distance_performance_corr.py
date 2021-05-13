import collections

import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas
from global_config import input_time_length, home
from layer_passes import get_num_of_predictions
from visualization.fold_result_visualization import get_5_fold_performance_df


def get_kernels_from_name(name):
    """based on the name of the model returns its max-pool layer's kernel sizes
    and dilations """
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
    """
    plots the
    :param df: pandas.DataFrame where the columns are performances of models on the different patients
    :param variable: 'vel' if we are plotting velocity, else 'absVel' for absolute velocity
    :return: dictionary where the performances are ordered based on the size of the receptive field of the networks
    """
    distance_performance_dict = {}
    for column in df.columns:
        if 'k4_d2' not in column:
            if ('Unnamed' not in column) and (variable in column):
                print(column)
                kernels, dilations = get_kernels_from_name(column)
                max_k, _ = get_num_of_predictions(kernels, dilations)
                distance = int((input_time_length - max_k + 1)/2)
                if ('k3_d3' in column) and ('sbp1' not in column):
                    distance = int(522/2)
                performance = df[column].to_numpy()
                print(column, distance, np.median(performance))

                distance_performance_dict[distance] = [np.median(performance), scipy.stats.sem(performance)]
    ordered_distance_performance_dict = collections.OrderedDict(sorted(distance_performance_dict.items()))

    return ordered_distance_performance_dict


def get_nums_from_dicts(distance_performance_dict):
    ys = list(distance_performance_dict.keys())
    ys = [y/250 for y in ys]
    xs = [l[0] for l in list(distance_performance_dict.values())]
    sem = [l[1] for l in list(distance_performance_dict.values())]
    return ys, xs, sem


def plot_distance_peformance(distance_performance_dicts, labels, title, output_file=None):
    """ plots the dependence of the performance on the size of the receptive field"""
    colors = ['steelblue', 'orange', 'limegreen', 'hotpink', 'turquoise', 'gold', 'hotpink']
    sem_colors = ['lightsteelblue', 'navajowhite', 'mistyrose', 'honeydew', 'lemonchiffon', 'azure', 'lavenderblush']
    models = ['k1', 'k2_d1', 'k3_d1', 'k4_d1', 'k2_d2', 'k3_d2', 'k2_d3', 'k3_d3', 'k3_d3_sbp1', 'k4_d3']
    plt.figure(figsize=(12, 6))
    for i, dp_dict in enumerate(distance_performance_dicts):
        ys, xs, sem = get_nums_from_dicts(dp_dict)
        plt.plot(ys, xs, label=labels[i], color=colors[i], marker='o')
        if i == 0:
            for j, model in enumerate(models):
                plt.annotate(text=model, xy=(ys[j], xs[j]+0.01) )
        plt.fill_between(ys, [x-s for x, s in zip(xs, sem)], [x+s for x, s in zip(xs, sem)], alpha=0.15)
    # plt.plot([341/250, 341/250], [0, 0.8], color='red', label='Deep4Net sbp1 window size')



    plt.xlabel('Half of the input window size giving one prediction (in seconds)', size='12')
    plt.ylabel('Correlation coefficient', size='12')
    # plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.grid(axis='y')
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    # files = [f'{home}/results/test_results/initial_performances.csv', f'{home}/results/test_results/hp_performances.csv',
    #          f'{home}/results/test_results/hp_valid_performances.csv', f'{home}/results/test_results/lp_valid_performances.csv',
    #          f'{home}/results/test_results/lpt_hpv_performances.csv']
    # files = [f'{home}/results/test_results/shifted_performances.csv', f'{home}/results/test_results/hp_shifted_performances.csv',
    #          f'{home}/results/test_results/hp_valid_shifted_performances.csv', f'{home}/results/test_results/lpt_hpv_shifted_performances.csv',
    #          f'{home}/results/test_results/lpt_hpv_shifted_performances.csv']
    files = [f'{home}/results/test_results/initial_performances.csv', f'{home}/results/test_results/lp_valid_performances.csv']
    dicts = []
    prefixes = ['m_', 'lp_m_']
    variable = 'absVel'
    # for file in files:

    big_df = get_5_fold_performance_df(variable)
    for i, prefix in enumerate(prefixes):
        sub_df = big_df[[column for column in big_df.columns if column.startswith(prefix)]]

        df = pandas.read_csv(files[i], sep=';', index_col=[0])
        wide_net_performances = df[[column for column in df.columns if (variable in column) and (('k4' in column) or ('sbp1' in column))]]
        sub_df = pandas.concat([sub_df, wide_net_performances], axis=1)
        df_dict = get_distance_to_performance(sub_df, variable)
        dicts.append(df_dict)
    if variable == 'vel':
        variable_name = 'velocity'
    else:
        variable_name = 'absolute velocity'
    title = f'Non-shifted {variable_name}'
    output_file = f'{home}/results/test_results/graphs/distance_shifted_performance_{variable}.png'
    plot_distance_peformance(dicts, ['Full train & valid', 'Full train & lp valid'], title, output_file)





