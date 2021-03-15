import argparse
import gc
import itertools
from pathlib import Path

import scipy

from Interpretation.interpretation import get_outs
from Interpretation.manual_manipulation import prepare_for_gradients
from global_config import output_dir, home, input_time_length
import matplotlib.pyplot as plt
import numpy as np

from layer_passes import get_num_of_predictions


def create_multi_graph(amp_grads, amp_grads_sems, batch_X, ax, title):
    y = np.mean(np.abs(amp_grads[0]), axis=(0, 1))
    mch = np.mean(np.abs(amp_grads[1]), axis=(0, 1))
    nch = np.mean(np.abs(amp_grads[2]), axis=(0, 1))
    ax.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), y, color='steelblue', label='All channels')
    ax.fill_between(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), y - amp_grads_sems[0], y + amp_grads_sems[0], alpha=0.2, color='steelblue')
    ax.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), mch, color='limegreen', label='Motor channels')
    ax.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), nch, color='lightcoral', label='Non-motor channels')

    ax.fill_between(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), mch - amp_grads_sems[1], mch + amp_grads_sems[1], color='limegreen', alpha=0.2)
    ax.fill_between(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), nch - amp_grads_sems[2], nch + amp_grads_sems[2], color='lightcoral', alpha=0.2)
    ax.set_title(title)


def get_kernel_and_dilation_from_long_name(file):
    kernels, dilations = None, None
    if ('3333' in file) or ('k3' in file):
        kernels = [3, 3, 3, 3]
    elif '2222' in file:
        kernels = [2, 2, 2, 2]
    elif 'k_1111' in file:
        kernels = [1, 1, 1, 1]
    elif 'k_4444' in file:
        kernels = [4, 4, 4, 4]

    if 'dilations_1111' in file:
        dilations = [1, 1, 1, 1]
    elif 'dilations_24' in file:
        dilations = [2, 4, 8, 16]
    else:
        dilations = [3, 9, 27, 81]

    return kernels, dilations


def plot_multiple_gradients_from_ndarrays(trained_mode, eval_mode, prefixes, file, titles, grad_type):
    plt.clf()
    fig_s, ax_s = plt.subplots(2, 2, sharey='row', figsize=(20, 10))
    output = f'{output_dir}/multigraphs/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}_ALLCH_sw.png'
    kernels = None
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    kernels, dilations = get_kernel_and_dilation_from_long_name(file)

    for i, prefix in enumerate(prefixes):
        gradients = np.load(f'{home}/outputs/all_layer_gradients/{file}/{prefix}/{grad_type}/conv_classifier/{grad_type}_avg_{file}_{train_mode}_{eval_mode}_ALLCH.npy')
        gradients_mch = np.load(f'{home}/outputs/all_layer_gradients/{file}/{prefix}/{grad_type}/conv_classifier/{grad_type}_avg_{file}_{train_mode}_{eval_mode}_MCH.npy')
        gradients_nch = np.load(f'{home}/outputs/all_layer_gradients/{file}/{prefix}/{grad_type}/conv_classifier/{grad_type}_avg_{file}_{train_mode}_{eval_mode}_NCH.npy')
        gradients_sem =  np.mean(scipy.stats.sem(np.abs(gradients), axis=1), axis=0)
        gradients_mch_sem = np.mean(scipy.stats.sem(np.abs(gradients_mch), axis=1), axis=0)
        gradients_nch_sem = np.mean(scipy.stats.sem(np.abs(gradients_nch), axis=1), axis=0)
        output_shape = get_num_of_predictions(kernels, dilations)
        batch = np.zeros([1, 1, input_time_length - output_shape[0] + 1, 1])
        create_multi_graph([gradients, gradients_mch, gradients_nch], [gradients_sem, gradients_mch_sem, gradients_nch_sem], batch, ax_s[indices[i]], titles[i])
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.tight_layout()
    print('saving figures:', output)
    fig_s.savefig(output)
    plt.show()
    plt.close(fig_s)


def plot_multiple_gradients(patient_index, trained_mode, eval_mode, prefixes, file, titles, trajectory_index,
                            grad_type='amps', motor_channels=None):
    plt.clf()
    fig, ax = plt.subplots(2, 2, sharey='row', figsize=(30, 50))
    fig_s, ax_s = plt.subplots(2, 2, sharey='row', figsize=(30, 15))
    shift = [False, True, False, True]
    high_pass = [False, False, True, True]
    # if motor_channels is not None:
    #     if motor_channels:
    #         m_ch_str = '_MCH'
    #     else:
    #         m_ch_str = '_NCH'
    # else:
    #     m_ch_str = ''
    m_ch_str = '_ALLCH'
    print('motor string:', m_ch_str)
    indices = [x for x in itertools.product(range(2), repeat=2)]
    output = f'{output_dir}/multigraphs/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}{m_ch_str}.png'
    output_s = f'{output_dir}/multigraphs/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}{m_ch_str}_sw.png'
    print('output:', output)

    Path(f'{output_dir}/multigraphs/{file}/{grad_type}/').mkdir(parents=True, exist_ok=True)
    for i, prefix in enumerate(prefixes):
        gradients, smaller_gradients = [], []
        gradients_mch, smaller_gradients_mch = [], []
        gradients_nch, smaller_gradients_nch = [], []

        correlations = []
        X_reshaped, small_window = None, None
        for patient_index in range(1, 13):
            model_name = f'{prefix}_{file}'
            corrcoef, new_model, X_reshaped, small_window, _, motor_channel_indices, non_motor_channel_indices = prepare_for_gradients(patient_index,
                                                                                     f'lr_0.001/{model_name}/',
                                                                                     trained_mode, eval_mode,
                                                                                     shift=shift[i],
                                                                                     high_pass=high_pass[i],
                                                                                     trajectory_index=trajectory_index,
                                                                                     motor_channels=motor_channels)
            # print(new_model)
            print('starting gradients')
            amp_grads, outs = get_outs(X_reshaped[:1], new_model, None, grad_type=grad_type)
            amp_grads_mch = np.take(amp_grads, motor_channel_indices.astype(int), axis=1)
            amp_grads_nch = np.take(amp_grads, non_motor_channel_indices.astype(int), axis=1)

            small_grads, outs = get_outs(X_reshaped[:1, :, :small_window], new_model, None, grad_type=grad_type)
            small_grads_mch = np.take(small_grads, motor_channel_indices.astype(int), axis=1)
            small_grads_nch = np.take(small_grads, non_motor_channel_indices.astype(int), axis=1)

            gradients.append(amp_grads)
            gradients_mch.append(amp_grads_mch)
            gradients_nch.append(amp_grads_nch)

            smaller_gradients.append(small_grads)
            smaller_gradients_mch.append(small_grads_mch)
            smaller_gradients_nch.append(small_grads_nch)
            correlations.append(corrcoef)
            gc.collect()

        corrcoef = sum(correlations) / len(correlations)
        amp_grads = np.concatenate(gradients, axis=1)
        amp_grads_mch = np.concatenate(gradients_mch, axis=1)
        amp_grads_nch = np.concatenate(gradients_nch, axis=1)

        Path(f'{output_dir}/gradients/{file}/{grad_type}/').mkdir(parents=True, exist_ok=True)
        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}{m_ch_str}', amp_grads)
        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}_MCH', amp_grads_mch)
        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}_NCH', amp_grads_nch)

        amp_grads_sd = np.std(np.abs(amp_grads), axis=(0, 1))
        small_amp_grads = np.concatenate(smaller_gradients, axis=1)
        small_amp_grads_mch = np.concatenate(smaller_gradients_mch, axis=1)
        small_amp_grads_nch = np.concatenate(smaller_gradients_nch, axis=1)

        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}{m_ch_str}_sw', small_amp_grads)
        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}_MCH_sw', small_amp_grads_mch)
        np.save(f'{home}/outputs/gradients/{file}/{grad_type}/{grad_type}_avg_{file}_{trained_mode}_{eval_mode}_NCH_sw', small_amp_grads_nch)

        small_amp_grads_sd = np.std(np.abs(small_amp_grads), axis=(0, 1))
        print(output)
        create_multi_graph([amp_grads, amp_grads_mch, amp_grads_nch], amp_grads_sd, X_reshaped,
                           ax[indices[i]], title=f'{eval_mode} {titles[i]} corr: {corrcoef:.2f}')
        create_multi_graph([small_amp_grads, small_amp_grads_mch, small_amp_grads_nch], small_amp_grads_sd,
                           X_reshaped[:1, :, :small_window, :], ax_s[indices[i]],
                           title=f'{eval_mode} {titles[i]} corr: {corrcoef:.2f}')
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.tight_layout()
    print('saving figures:', output, output_s)
    fig.savefig(output)
    fig_s.savefig(output_s)
    plt.show()
    plt.close(fig)
    plt.close(fig_s)


trained_modes = ['trained']
eval_modes = ['train', 'validation']

parser = argparse.ArgumentParser()
parser.add_argument("--grad_type", default='amps', type=str)
parser.add_argument('--variable', default=0, type=int)
parser.add_argument('--channels', default=None, type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.channels is not None:
        if args.channels == 'MCH':
            motor_channels = True
        else:
            motor_channels = False
    else:
        motor_channels = None
    print('motor channels,', motor_channels)
    trajectory_index = args.variable
    if trajectory_index == 1:
        variable = 'absVel'
    else:
        variable = 'vel'
    grad_type = args.grad_type

    files = [f'{variable}_k_3333_dilations_1111', f'{variable}_k_4444',
             f'{variable}_k_3333', f'{variable}_k_3333_dilations_24816',
             f'{variable}_k_4444_dilations_1111']


    files2 = [f'{variable}_k_4444_dilations_24816', f'{variable}_k_1111', f'{variable}_k_2222_dilations_1111', f'{variable}_k_2222',
              f'{variable}_k_2222_dilations_24816', f'{variable}_k_1111']
    # files3 = [f'{variable}_k_1111']

    files = files + files2

    # files = files
    # prefixes = ['m', 's2_m', 'hp_m', 'hp_sm2']
    prefixes = ['sbp1_m', 'sbp1_sm', 'sbp1_hp', 'sbp1_hps']

    titles = ['Initial setting', 'Shifted shifted', 'Initial high-pass setting',
              'Shifted high-pass setting']
    for file in files:
        for train_mode in trained_modes:
            for eval_mode in eval_modes:
                # plot_multiple_gradients(4, train_mode, eval_mode, prefixes, file, titles, trajectory_index,
                #                         grad_type=grad_type, motor_channels=motor_channels)
                plot_multiple_gradients_from_ndarrays(train_mode, eval_mode, prefixes, file, titles, grad_type)