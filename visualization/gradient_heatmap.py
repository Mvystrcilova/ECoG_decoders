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


def get_gradient_for_shift(shift, layer, gradient_kind, file):
    gradients = None
    print('shift:', shift, 'layer:', layer, 'gradient_kind:', gradient_kind)
    if shift != 261:
        gradients = np.load(
            f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/amps_avg_{file}_trained_train_{gradient_kind}.npy')
        return gradients
    # elif shift == 0:
    #     if 'vel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/vel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
    #     elif 'absVel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/absVel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')

    elif shift == 261:
        if 'vel' in file:
            gradients = np.load(
                f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
        elif 'absVel' in file:
            gradients = np.load(
                f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/{layer}/amps_avg_absVel_k_3333_trained_train_{gradient_kind}.npy')

    assert gradients is not None
    return gradients


def get_gradients_for_all_shifts(shifts, layer, gradient_kind, file):
    gradients = pandas.DataFrame()
    for shift in shifts:
        gradient = get_gradient_for_shift(shift, layer, gradient_kind, file)
        gradients[shift] = np.mean(gradient, axis=(0, 1))
    return gradients


def set_gradient_df_index(gradient_df, layer, file):
    kernel_size, dilations = get_kernels_from_name(file)
    max_k, max_l = get_num_of_predictions(kernel_size, dilations, layer=None)
    print(file, max_k)
    if 'k3_d3' in file:
        max_k = 679
    shape = min((input_time_length - max_k) * 2, 1200)
    # shape = 522
    y = np.around(np.fft.rfftfreq(shape, 1 / 250.0), 0)
    # index = np.linspace(0, 125, len(y))
    # new_columns = [int(1000*(int(column)/250)) for column in gradient_df.columns]
    # gradient_df.columns = new_columns
    # y = [str(ypsilon).split('.')[0] for ypsilon in y]
    gradient_df = gradient_df.set_index(pandas.Index(y), drop=True)

    return gradient_df


def plot_gradient_heatmap(gradient_df, title, output, xlabel, ax):
    # sns.color_palette("vlag", as_cmap=True)
    minimum = min(min(gradient_df[0].min()), min(gradient_df[1].min()), min(gradient_df[2].min()))
    maximum = max(max(gradient_df[0].max()), max(gradient_df[1].max()), max(gradient_df[2].max()))
    for i, a in enumerate(ax):
        sns.heatmap(gradient_df[i], cmap='coolwarm', center=0, cbar_kws={'label': 'Gradients'},
                    ax=a, vmin=minimum, vmax=maximum)
    # if gradient_df.shape[0] > 7:
        locs, labels = plt.yticks()
        labels = np.arange(0, 126, 25)
        locs = np.linspace(min(locs), max(locs), len(labels))
        a.set_title(title[i])
        # plt.yticks(np.arange(0, 125, 1))
        plt.yticks(locs, labels=labels)
        # plt.title(title)
        a.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True, rotation=75, labelsize=12)
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency Hz')


def get_gradient_title(layer, gradient_kind):
    gradient_title_dict = {'conv_spat': 'Spatial convolution', 'conv_2': 'First convolutional layer',
                           'conv_3': 'Second convolutional layer', 'conv_4': 'Third convolutional layer',
                           'conv_classifier': 'Output layer'}
    if gradient_kind == 'MCH':
        gradient_string = 'Motor channels'
    elif gradient_kind == 'NCH':
        gradient_string = 'Non-notor channels'
    else:
        gradient_string = 'All channels'
    return f' {gradient_string} - {gradient_title_dict[layer]}'


def get_gradient_for_file(file, layer, gradient_kind, variable, prefix):
    gradient = np.load(
        f'{home}/outputs/all_layer_gradients2/{file}/{prefix}/amps/{layer}/amps_avg_{file}_trained_train_{gradient_kind}.npy')
    gradient = np.mean(gradient, axis=(0, 1))
    return gradient


def create_shift_gradient_heatmap():
    variable = 'vel'
    shifts = [x for x in range(-250, 251, 25)] + [261]
    # shifts = [-250]
    shifts.remove(261)
    # shifts.remove(75)
    # shifts = [int(1000 * (x / 250)) for x in shifts]
    file = f'{variable}_k3_d3'
    layers = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    for layer in layers:
        for gradient_kind in ['ALLCH', 'MCH', 'NCH']:
            output_dir = f'{home}/outputs/shift_gradients/{file}/'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output = f'{output_dir}/shift_gradients_{layer}_{gradient_kind}.png'
            gradient_df = get_gradients_for_all_shifts(shifts, layer, gradient_kind, file)
            gradient_df = set_gradient_df_index(gradient_df, layer)
            title = get_gradient_title(layer, gradient_kind)
            plot_gradient_heatmap(gradient_df, title, output,
                                  xlabel='Shift with respect to receptive field centre (in milliseconds)')


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


def model_gradients_heatmap(files, layers, variable, prefix, saved_models_dir='lr_0.001'):
    for layer in layers:
        output_dir = f'{home}/outputs/all_model_gradients/{layer}/{prefix}/'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 3, sharey='row', figsize=(18, 6))
        gradient_dfs = []
        titles = []

        for i, gradient_kind in enumerate(['ALLCH', 'MCH', 'NCH']):
            gradient_dict = {}
            index_dict = {}
            for file in files:
                gradient_df = pandas.DataFrame()
                gradient = get_gradient_for_file(file, layer, gradient_kind, variable, prefix)
                gradient_dict[file] = gradient
                gradient_df[file] = gradient
                if 'sbp1' in file:
                    shape = min((input_time_length - 519) * 2, 1200)
                else:
                    model = load_model(f'/models/saved_models/{saved_models_dir}/{prefix}_{file}/{prefix}_{file}_p_1/last_model')
                    with torch.no_grad():
                        in_channels = list(model.parameters())[2].shape[3]
                        test_out = model(np_to_var(np.zeros([1, in_channels, 1200])))
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
            print(gradient_df[f'{variable}_k1_d3'].tolist())
            gradient_df = gradient_df.reindex([f'{variable}_k1_d3', f'{variable}_k2_d3',
                                               f'{variable}_k3_d3', f'{variable}_sbp1_k3_d3',
                                               f'{variable}_k2_d1',  f'{variable}_k3_d1',
                                               f'{variable}_k2_d2', f'{variable}_k3_d2'], axis=1)
            gradient_df = gradient_df.rename({f'{variable}_k3_d3': f'{variable}_k3_d3_sbp0',
                                              f'{variable}_sbp1_k3_d3': f'{variable}_k3_d3_sbp1',
                                             f'{variable}_k1_d3': f'{variable}_k1'}, axis=1)
            print(gradient_df[f'{variable}_k1'].tolist())

            title = get_gradient_title(layer, gradient_kind)
            gradient_dfs.append(gradient_df)
            titles.append(title)
        output = f'{output_dir}/{variable}_model_gradients_all_kinds.png'
        plot_gradient_heatmap(gradient_dfs, titles, output, xlabel='Models', ax=ax)
        plt.tight_layout()
        plt.savefig(output)
        plt.show()

if __name__ == '__main__':
    variable = 'absVel'
    files = [f'{variable}_k1_d3', f'{variable}_k2_d3', f'{variable}_k3_d3',
             f'{variable}_sbp1_k3_d3',
             f'{variable}_k2_d1', f'{variable}_k3_d1',
             f'{variable}_k2_d2', f'{variable}_k3_d2'
              ]
    for prefix in ['m', 'sm', 'hp_m', 'hp_sm']:
        model_gradients_heatmap(files, ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier'], variable, prefix)
