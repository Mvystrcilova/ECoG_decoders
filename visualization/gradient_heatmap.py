from pathlib import Path

import numpy as np
import pandas
import seaborn as sns
from global_config import home, input_time_length
from layer_passes import get_num_of_predictions
import matplotlib.pyplot as plt

def get_gradient_for_shift(shift, layer, gradient_kind, file):
    gradients = None
    print('shift:', shift, 'layer:', layer, 'gradient_kind:', gradient_kind)
    if shift != 261:
        gradients = np.load(f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/amps_avg_{file}_trained_train_{gradient_kind}.npy')
        return gradients
    # elif shift == 0:
    #     if 'vel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/vel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
    #     elif 'absVel' in file:
    #         gradients = np.load(f'{home}/outputs/all_layer_gradients/absVel_k_3333/sbp1_sm/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')

    elif shift == 261:
        if 'vel' in file:
            gradients = np.load(f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/amps_avg_vel_k_3333_trained_train_{gradient_kind}.npy')
        elif 'absVel' in file:
            gradients = np.load(f'{home}/outputs/all_layer_gradients/{file}/shift_{shift}/sbp0_m/amps/{layer}/{layer}/amps_avg_absVel_k_3333_trained_train_{gradient_kind}.npy')

    assert gradients is not None
    return gradients


def get_gradients_for_all_shifts(shifts, layer, gradient_kind, file):
    gradients = pandas.DataFrame()
    for shift in shifts:
        gradient = get_gradient_for_shift(shift, layer, gradient_kind, file)
        gradients[shift] = np.mean(gradient, axis=(0, 1))
    return gradients


def set_gradient_df_index(gradient_df, layer):
    max_k, max_l = get_num_of_predictions([3, 3, 3, 3], [3, 9, 27, 81], layer=layer)
    shape = input_time_length - max_k + 1
    y = np.around(np.fft.rfftfreq(shape, 1 / 250.0), 0)
    # index = np.linspace(0, 125, len(y))
    new_columns = [int(1000*(int(column)/250)) for column in gradient_df.columns]
    gradient_df.columns = new_columns
    y = [str(ypsilon).split('.')[0] for ypsilon in y]
    gradient_df = gradient_df.set_index(pandas.Index(y), drop=True)
    return gradient_df


def plot_gradient_heatmap(gradient_df, title, output):
    # sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(gradient_df, cmap='coolwarm', center=0, cbar_kws={'label': 'Gradients'})
    plt.title(title)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True)
    plt.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        right=False )
    plt.xlabel('Shift with respect to receptive field centre (in milliseconds)')
    plt.ylabel('Frequency Hz')
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


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


if __name__ == '__main__':
    variable = 'absVel'
    shifts = [x for x in range(-250, 251, 25)] + [261]
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
            plot_gradient_heatmap(gradient_df, title, output)



