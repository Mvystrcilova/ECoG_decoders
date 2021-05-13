from pathlib import Path

import numpy as np
import pandas
import seaborn as sns
import torch
from braindecode.util import np_to_var

from global_config import home, input_time_length
from layer_passes import get_num_of_predictions
import matplotlib.pyplot as plt

from models.Model import load_model
from visualization.distance_performance_corr import get_kernels_from_name


def get_gradient_for_shift(shift, layer, gradient_kind, file, prefix):
    gradients = None
    print('shift:', shift, 'layer:', layer, 'gradient_kind:', gradient_kind)
    if shift != 261:
        gradients = np.load(
            f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/{prefix}/amps/{layer}/amps_avg_{file}_trained_train_{gradient_kind}.npy')
        return gradients
    # elif shift == 0:
    #     if 'vel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/vel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
    #     elif 'absVel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/absVel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')

    elif shift == 261:
        if 'vel' in file:
            gradients = np.load(
                f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/{prefix}/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
        elif 'absVel' in file:
            gradients = np.load(
                f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/{prefix}/amps/{layer}/{layer}/amps_avg_absVel_k_3333_trained_train_{gradient_kind}.npy')

    assert gradients is not None
    return gradients


def get_gradients_for_all_shifts(shifts, layer, gradient_kind, file, prefix):
    gradients = pandas.DataFrame()
    for shift in shifts:
        gradient = get_gradient_for_shift(shift, layer, gradient_kind, file, prefix)
        gradients[shift] = np.mean(gradient, axis=(0, 1))
    return gradients


def set_gradient_df_index(gradient_df, layer, file, shifts=False):
    kernel_size, dilations = get_kernels_from_name(file)
    max_k, max_l = get_num_of_predictions(kernel_size, dilations, layer=None)
    print(file, max_k)
    if 'k3_d3' in file:
        max_k = 521
        shape = 521 * 2
    else:
        shape = min((input_time_length - max_k) * 2, 1200)
    # shape = 522
    y = np.around(np.fft.rfftfreq(shape, 1 / 250.0), 0)
    # index = np.linspace(0, 125, len(y))
    if shifts:
        new_columns = [int(1000*(int(column)/250)) for column in gradient_df.columns]
        gradient_df.columns = new_columns
    # y = [str(ypsilon).split('.')[0] for ypsilon in y]
    gradient_df = gradient_df.set_index(pandas.Index(y), drop=True)

    return gradient_df


def plot_gradient_heatmap(gradient_df, title, output, xlabel, ax):
    # sns.color_palette("vlag", as_cmap=True)
    minimum = min(min(gradient_df[0].min()), min(gradient_df[1].min()), min(gradient_df[2].min()))
    maximum = max(max(gradient_df[0].max()), max(gradient_df[1].max()), max(gradient_df[2].max()))
    for i, a in enumerate(ax):
        coef = 0.2
        sns.heatmap(gradient_df[i].iloc[::-1], cmap='coolwarm', center=0, cbar_kws={'label': 'Gradients'},
                    ax=a, vmin=minimum+minimum*-1*coef, vmax=maximum-maximum*coef)
        if i == 0:
            a.set_ylabel('Frequency Hz', size=14)
        a.set_xlabel(xlabel, size=14)
    # if gradient_df.shape[0] > 7:
        locs, labels = plt.yticks()
        labels = np.arange(0, 126, 25)
        locs = np.linspace(min(locs), max(locs), len(labels))
        a.set_title(title[i], size=16)
        # plt.yticks(np.arange(0, 125, 1))
        plt.yticks(locs, labels=reversed(labels))
        # plt.title(title)
        a.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True, rotation=75, labelsize=15)
        a.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False, labelsize=15)


def get_gradient_title(layer, gradient_kind):
    gradient_title_dict = {'conv_spat': 'Spatial convolution', 'conv_2': '\nFirst convolutional layer',
                           'conv_3': '\nSecond convolutional layer', 'conv_4': '\nThird convolutional layer',
                           'conv_classifier': 'Output layer'}
    if gradient_kind == 'MCH':
        gradient_string = 'Motor channels'
    elif gradient_kind == 'NCH':
        gradient_string = 'Non-motor channels'
    else:
        gradient_string = 'All channels'
    # return f' {gradient_string} - {gradient_title_dict[layer]}'
    return f' {gradient_string} - {layer}'


def get_gradient_for_file(file, layer, gradient_kind, variable, prefix, gradient_dir):
    gradient = np.load(
        f'{home}/outputs/{gradient_dir}/{file}/{prefix}/amps/{layer}/amps_avg_{file}_trained_train_{gradient_kind}.npy')
    gradient = np.mean(gradient, axis=(0, 1))
    return gradient


def create_shift_gradient_heatmap(prefix, variable):
    """
    Plotting gradients of the Deep4Net sbp0 architecture during the gradual shifting

    :param prefix: the setting specified by
    :param variable: 'vel' for velocity, 'absVel' for absolute velocity gradients
    :return: None
    """
    shifts = [x for x in range(-250, 251, 25)] + [261]
    # shifts = [-250]
    shifts.remove(261)
    # shifts.remove(250)
    # shifts.remove(75)
    # shifts = [int(1000 * (x / 250)) for x in shifts]
    file = f'{variable}_k3_d3'
    layers = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    for layer in layers:
        gradient_dfs = []
        titles = []
        fig, ax = plt.subplots(1, 3, sharey='row', figsize=(15, 4))
        output_dir = f'{home}/outputs/shift_gradients/{file}/{prefix}'
        output = f'{output_dir}/{prefix}_shift_gradients_{layer}_all_kinds.png'

        for i, gradient_kind in enumerate(['ALLCH', 'MCH', 'NCH']):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            gradient_df = get_gradients_for_all_shifts(shifts, layer, gradient_kind, file, prefix)
            gradient_df = set_gradient_df_index(gradient_df, layer, file, shifts=True)
            title = get_gradient_title(layer, gradient_kind)
            gradient_dfs.append(gradient_df)
            titles.append(title)

        plot_gradient_heatmap(gradient_dfs, titles, output,
                                  xlabel='Shift with respect to receptive \nfield centre (in milliseconds)',
                                  ax=ax)
        plt.tight_layout()
        plt.savefig(output)
        plt.show()


def extend_second_list(long_list: list, long_index_list: list, shorter_list: list, shorter_index_list):
    longer_short_list = []
    short_list_index = 0
    for i, item in enumerate(long_index_list):
        longer_short_list.append(shorter_list[short_list_index])
        if long_index_list[i] == shorter_index_list[short_list_index]:
            if (short_list_index+1) < len(shorter_list):
                # if short_list_index == 480:
                #     print('stop')
                short_list_index += 1

    return longer_short_list, long_index_list


def model_gradients_heatmap(files, layers, variable, prefix, gradient_dir, saved_models_dir='lr_0.001'):
    for layer in layers:
        output_dir = f'{home}/outputs/{gradient_dir}/{layer}/{prefix}/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, sharey='row', figsize=(15, 6))
        gradient_dfs = []
        titles = []

        for i, gradient_kind in enumerate(['ALLCH', 'MCH', 'NCH']):
            gradient_dict = {}
            index_dict = {}
            for file in files:
                gradient_df = pandas.DataFrame()
                gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix, gradient_dir)
                gradient_dict[file] = gradient
                gradient_df[file] = gradient
                if 'sbp1' in file:
                    shape = min((input_time_length - 519) * 2, 1200)
                else:
                    model = load_model(f'/models/saved_models/{saved_models_dir}/{prefix}_{file}/{prefix}_{file}_p_1/last_model')
                    with torch.no_grad():
                        in_channels = list(model.parameters())[2].shape[3]
                        test_out = model.double()(np_to_var(np.zeros([1, in_channels, 1200])).cuda())
                    shape = min((input_time_length - test_out.shape[1]) * 2, 1200)
                y = np.around(np.fft.rfftfreq(shape, 1 / 250.0), 0)
                gradient_df = gradient_df.set_index(pandas.Index(y), drop=True)
                index_dict[file] = list(gradient_df.index.values)
            sorted_gradient_dict = {k: v for k, v in sorted(gradient_dict.items(), key=lambda item: len(item[1]), reverse=True)}
            longest_k = list(sorted_gradient_dict.keys())[0]
            gradient_df = pandas.DataFrame()
            gradient_df[longest_k] = sorted_gradient_dict[longest_k]
            gradient_df.set_index(pandas.Index(index_dict[longest_k]), drop=True)
            for k, v in sorted_gradient_dict.items():
                if k != longest_k:
                    gradient, new_index = extend_second_list(long_list=sorted_gradient_dict[longest_k],
                                                             long_index_list=index_dict[longest_k],
                                                             shorter_list=v, shorter_index_list=index_dict[k])
                    gradient_df[k] = gradient
            # print(gradient_df[f'{variable}_k1_d3'].tolist())
            gradient_df = gradient_df.reindex([f'{variable}_k1_d3', f'{variable}_k2_d3',
                                               f'{variable}_k3_d3',
                                               f'{variable}_k2_d1',  f'{variable}_k3_d1',
                                               f'{variable}_k2_d2', f'{variable}_k3_d2'], axis=1)
            gradient_df = gradient_df.rename({f'{variable}_k3_d3': f'{variable}_k3_d3_sbp0',
                                             f'{variable}_k1_d3': f'{variable}_k1'}, axis=1)
            # print(gradient_df[f'{variable}_k1'].tolist())

            title = get_gradient_title(layer, gradient_kind)
            gradient_dfs.append(gradient_df)
            titles.append(title)
        output = f'{output_dir}/{variable}_model_gradients_all_kinds.png'
        plot_gradient_heatmap(gradient_dfs, titles, output, xlabel='Models', ax=ax)
        plt.tight_layout()
        plt.savefig(output)
        # plt.show()


def plot_one_file_heatmap(df, file, variable, layers, gradient_kinds, prefix):
    initial_mins = [1000, 10000, 1000, 1000, 1000]
    initial_maxes = [-1000, -10000, -1000, -1000, -1000]

    for grad_kind_df in df:
        mins = [min(layer_df.min()) for layer_df in grad_kind_df]
        maxes = [max(layer_df.max()) for layer_df in grad_kind_df]
        initial_mins = [min(initial, new) for initial, new in zip(initial_mins, mins)]
        initial_maxes = [max(initial, new) for initial, new in zip(initial_maxes, maxes)]
    print('initial mins:', initial_mins, 'initial_maxes:', initial_maxes)
    for j, grad_kind_df in enumerate(df):
        fig, ax = plt.subplots(1, 5, sharey='row', figsize=(25, 5))
        for i, layer_df in enumerate(grad_kind_df):

            print('seting gradient_df_index', layers[i])
            layer_df = set_gradient_df_index(layer_df, layers[i], file, shifts=False)
            print('creating heatmap')
            min_coef = 1
            if 'h' in prefix:
                if i == 0:
                    coef = 0.99
                elif i <= 2:
                    coef = 0.5
                else:
                    coef = 0.75
            else:
                if i == 0:
                    coef = 0.99
                else:
                    coef = 0.6
            sns.heatmap(layer_df, cmap='coolwarm', center=0, cbar_kws={'label': 'Gradients'},
                        ax=ax[i], vmin=initial_mins[i]+-1*initial_mins[i]*coef, vmax=initial_maxes[i]-initial_maxes[i]*coef)
            ax[i].plot([0, layer_df.shape[0]], [0, 0])
            if i == 0:
                ax[i].set_ylabel('Frequency Hz')
            ax[i].set_title(layers[i], size=15)
            locs, labels = plt.yticks()
            labels = np.arange(0, 126, 25)
            locs = np.linspace(min(locs), max(locs), len(labels))
            plt.yticks(locs, labels=labels)
            ax[i].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=True, rotation=75, labelsize=15)
        print('saving to:', f'{home}/results/one_file_resutls/{file}/{variable}/heatmaps.png')
        plt.tight_layout()
        Path(f'{home}/results/one_file_resutls/{file}/{variable}').mkdir(exist_ok=True, parents=True)
        plt.savefig(f'{home}/results/one_file_resutls/{file}/{variable}/{prefix}_{gradient_kinds[j]}_heatmaps_full_grads.png')
        plt.show()


def plot_one_file_results(file, variable):
    grad_kind_dfs = []
    grad_kind_dfs2 = []

    layers = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    for gradient_kind in ['ALLCH', 'MCH', 'NCH']:
        layer_dfs1 = []
        layer_dfs2 = []

        for layer in layers:
            normal_gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix='m', gradient_dir='all_layer_gradients2')
            pw_gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix='m', gradient_dir='all_layer_grads_pw')
            shifted_gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix='sm', gradient_dir='all_layer_gradients2')
            pw_shifted_gradient = get_gradient_for_file(file, layer, gradient_kind, variable=variable, prefix='sm', gradient_dir='all_layer_grads_pw')
            hp_gradient = get_gradient_for_file(file, layer, gradient_kind, variable=variable, prefix='hp_m', gradient_dir='all_layer_gradients2')
            pw_hp_gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix='hp_m', gradient_dir='all_layer_grads_pw')
            shifted_hp_gradient = get_gradient_for_file(file, layer, gradient_kind, variable=variable, prefix='hp_sm', gradient_dir='all_layer_gradients2')
            pw_shifted_hp_gradient = get_gradient_for_file(file, layer, gradient_kind, variable=variable, prefix='hp_sm', gradient_dir='all_layer_grads_pw')
            prefixes1 = ['w0_s0',  'w1_s0', 'w0_s1', 'w1_s1']
            prefixes2 = ['w0_s0',  'w1_s0', 'w0_s1', 'w1_s1']

            cols1 = [normal_gradient, pw_gradient, shifted_gradient, pw_shifted_gradient]
            cols_2 = [hp_gradient, pw_hp_gradient, shifted_hp_gradient, pw_shifted_hp_gradient]
            df = pandas.DataFrame()
            df2 = pandas.DataFrame()
            for prefix, col in zip(prefixes1, cols1):
                df[prefix] = col
            for prefix, col in zip(prefixes2, cols_2):
                df2[prefix] = col
            layer_dfs1.append(df)
            layer_dfs2.append(df2)
        grad_kind_dfs.append(layer_dfs1)
        grad_kind_dfs2.append(layer_dfs2)
    # plot_one_file_heatmap(grad_kind_dfs, file=file, variable=variable, layers=layers, gradient_kinds=['ALLCH', 'MCH', 'NCH'], prefix='full')
    plot_one_file_heatmap(grad_kind_dfs2, file, variable=variable, layers=layers, gradient_kinds=['ALLCH', 'MCH', 'NCH'], prefix='hp')


if __name__ == '__main__':
    """
    Plots the gradient heatmaps of gradient saved via the gradient_inspection.py script.
    """
    variable = 'vel'
    files = [f'{variable}_k1_d3', f'{variable}_k2_d3', f'{variable}_k3_d3',
             f'{variable}_k2_d1', f'{variable}_k3_d1',
             f'{variable}_k2_d2', f'{variable}_k3_d2'
              ]
    prefix = ['sbp0_m', 'sbp0_hp_m']
    prefixes = ['m', 'sm', 'hp_m', 'hp_sm']
    prefixes2 = ['sm']

    # plotting gradients of all architectures on the different datasets
    for prefix in prefixes:
    #     model_gradients_heatmap(files, ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier'], variable, prefix,
    #                             'all_layer_gradients2')
        model_gradients_heatmap(files, ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier'], variable, prefix,
                            'all_layer_grads_pw', saved_models_dir='pre_whitened')

    # plotting gradient changes during the gradual shift of the predicted time-point
    # for p in prefix:
    #     create_shift_gradient_heatmap(p, variable)

    # plotting all experiments for only one architecture
    # for variable in ['vel', 'absVel']:
    #     plot_one_file_results(f'{variable}_k3_d3', variable)
