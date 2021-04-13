import argparse
import logging
from pathlib import Path

import matplotlib
import scipy
from braindecode.util import np_to_var, var_to_np

from Interpretation.interpretation import calculate_phase_and_amps
from Interpretation.manual_manipulation import prepare_for_gradients
from global_config import output_dir, cuda, input_time_length, home
from gradient_multigraphs import get_kernel_and_dilation_from_long_name
from layer_passes import get_num_of_predictions

log = logging.getLogger()
log.setLevel('DEBUG')
import sys
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
from matplotlib import pyplot as plt

matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14
import seaborn
import torch

seaborn.set_style('darkgrid')


def plot_all_module_gradients(titles, batch_X, gradients, output_file, shift_by):
    fig, ax = plt.subplots(2, 2, sharey='row', figsize=(17, 12))
    if shift_by is not None:
        titles = [f'{title} Shifted by: {shift_by}' for title in titles]
    # indices = [0, 1, 2, 3]
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, gradient in enumerate(gradients[0]):
        sem = np.mean(scipy.stats.sem(np.abs(gradient), axis=1), axis=0)
        mch_sem = np.mean(scipy.stats.sem(np.abs(gradients[1][i]), axis=1), axis=0)
        nch_sem = np.mean(scipy.stats.sem(np.abs(gradients[2][i]), axis=1), axis=0)
        y = np.mean(np.abs(gradient), axis=(0, 1))
        mch = np.mean(np.abs(gradients[1][i]), axis=(0, 1))
        nch = np.mean(np.abs(gradients[2][i]), axis=(0, 1))
        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), y, color='steelblue', label='All channels')
        ax[indices[i]].fill_between(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), y - sem, y + sem, alpha=0.2, color='steelblue')

        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), mch, color='limegreen',
                            label='Motor channels')
        ax[indices[i]].fill_between(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), mch - mch_sem, mch + mch_sem, alpha=0.2, color='limegreen')

        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), nch, color='lightcoral',
                            label='Non-motor channels')
        ax[indices[i]].fill_between(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), nch - nch_sem, nch + nch_sem, color='lightcoral',
                                    alpha=0.2)

        ax[indices[i]].set_title(titles[i])

    plt.legend()
    plt.tight_layout()
    # plt.savefig(output_file)
    plt.show()


def get_module_gradients(model, module_name, X_reshaped, small_window):
    print("Module {:s}...".format(module_name))
    ## Create new model

    new_model = torch.nn.Sequential()
    found_selected = False
    for name, child in model.named_children():
        new_model.add_module(name, child)
        if name == module_name:
            found_selected = True
            break
    assert found_selected
    # Batch through X for GPU memory reasons
    print("Computing gradients...")
    with torch.no_grad():
        if cuda:
            test_out = new_model(np_to_var(X_reshaped[:2]).cuda())
        else:
            test_out = new_model(np_to_var(X_reshaped[:2]))

    n_filters = test_out.shape[1]
    n_preds = test_out.shape[2]
    print('test out shape:', test_out.shape)
    if small_window is None:
        small_window = min((input_time_length - n_preds)*2, 1200)
    new_X_reshaped = X_reshaped[:, :, :small_window, :]

    # filters x windows x channels x freqs
    all_amp_grads = np.ones(
        (n_filters,) + new_X_reshaped.shape[:2] + (len(np.fft.rfftfreq(new_X_reshaped.shape[2], d=1 / 250.0)),),
        dtype=np.float32) * np.nan
    # all_phases_grads = np.ones(
    #     (n_filters,) + new_X_reshaped.shape[:2] + (len(np.fft.rfftfreq(new_X_reshaped.shape[2], d=1 / 250.0)),),
    #     dtype=np.float32) * np.nan

    i_start = 0
    print('small window:', small_window)
    for batch_X in np.array_split(new_X_reshaped, 5):
        iffted, amps_th, phases_th = calculate_phase_and_amps(batch_X)

        if cuda:
            outs = new_model(iffted.double().cuda())
        else:
            outs = new_model(iffted.double())
        assert outs.shape[1] == n_filters
        print('model outputs shape:', outs.shape)
        for i_filter in range(n_filters):
            mean_out = torch.mean(outs[:, i_filter])
            mean_out.backward(retain_graph=True)
            amp_grads = var_to_np(amps_th.grad)
            # print(amp_grads.shape)
            all_amp_grads[i_filter, i_start:i_start + len(amp_grads)] = amp_grads.squeeze(-1)
            phases_grads = var_to_np(phases_th.grad)
            # print(phases_grads.shape)
            # all_phases_grads[i_filter, i_start:i_start + len(phases_grads)] = phases_grads.squeeze(-1)
            amps_th.grad.zero_()
            phases_th.grad.zero_()

        i_start += len(amp_grads)

    del amp_grads  # just make sure I don't use it accidentally now
    del phases_grads  # just make sure I don't use it accidentally now
    assert i_start == all_amp_grads.shape[1]
    assert not np.any(np.isnan(all_amp_grads))
    # assert i_start == all_phases_grads.shape[1]
    # assert not np.any(np.isnan(all_phases_grads))
    # mean across windows
    meaned_amp_grads = np.mean(all_amp_grads, axis=1)
    # meaned_phase_grads = np.mean(all_phases_grads, axis=1)
    meaned_phase_grads = None
    # phase_grads_list.append(meaned_phase_grads)
    # amp_grads_list.append(meaned_amp_grads)
    print('grads shape:', meaned_amp_grads.shape)
    return meaned_amp_grads, meaned_phase_grads, small_window


def get_gradients_for_intermediate_layers(select_modules, prefix, file, shift, high_pass, trajectory_index, motor_channels,
                                          low_pass, shift_by, saved_model_dir, whiten, gradient_save_dir):
    amp_gradient_dict = {module_name: [] for module_name in select_modules}
    amp_gradient_dict_mch = {module_name: [] for module_name in select_modules}
    amp_gradient_dict_nch = {module_name: [] for module_name in select_modules}

    # phase_gradient_dict_mch = {module_name: [] for module_name in select_modules}
    # phase_gradient_dict_nch = {module_name: [] for module_name in select_modules}
    # phase_gradient_dict = {module_name: [] for module_name in select_modules}

    X_reshaped_list, X_reshaped_list_mch, X_reshaped_list_nch = [], [], []
    for patient_index in range(1, 13):
        print('prefix index', i)
        model_name = f'{prefix}_{file}'
        print(patient_index, model_name)
        corrcoef, new_model, X_reshaped, small_window, _, motor_channel_indices, non_motor_channel_indices = prepare_for_gradients(
            patient_index, f'{saved_model_dir}/{model_name}', train_mode, eval_mode,
            shift=shift, high_pass=high_pass, trajectory_index=trajectory_index, multi_layer=True,
            motor_channels=motor_channels, low_pass=low_pass, shift_by=shift_by, whiten=whiten,
            saved_model_dir=saved_model_dir)

        # amp_grads_list, amp_grads_list_mch, amp_grads_list_nch = [], [], []
        # phase_grads_list, phase_grads_list_mch, phase_grads_list_nch = [], [], []
        # X_reshaped = X_reshaped[:, :, :small_window]
        for j, module_name in enumerate(select_modules):
            amp_grads, phase_grads, module_filters = get_module_gradients(new_model, module_name,
                                                                          X_reshaped, small_window=small_window)
            # amp_grads_mch, phase_grads_mch = np.take(amp_grads, motor_channel_indices.astype(int),
            #                                          axis=1), np.take(phase_grads,
            #                                                           motor_channel_indices.astype(int),
            #                                                           axis=1)
            amp_grads_mch, phase_grads_mch = np.take(amp_grads, motor_channel_indices.astype(int),
                                                     axis=1), None
            # amp_grads_nch, phase_grads_nch = np.take(amp_grads, non_motor_channel_indices.astype(int),
            #                                          axis=1), np.take(phase_grads,
            #                                                           non_motor_channel_indices.astype(
            #                                                               int), axis=1)

            amp_grads_nch, phase_grads_nch = np.take(amp_grads, non_motor_channel_indices.astype(int), axis=1), None
            amp_gradient_dict[module_name].append(amp_grads)
            amp_gradient_dict_mch[module_name].append(amp_grads_mch)
            amp_gradient_dict_nch[module_name].append(amp_grads_nch)

            # phase_gradient_dict[module_name].append(phase_grads)
            # phase_gradient_dict_mch[module_name].append(phase_grads_mch)
            # phase_gradient_dict_nch[module_name].append(phase_grads_nch)

            if len(X_reshaped_list) < 4:
                X_reshaped_list.append(X_reshaped[:, :, :module_filters])

    amp_grads_list, amp_grads_list_mch, amp_grads_list_nch = [], [], []
    amp_grads_std = []
    # phase_grads_list, phase_grads_list_mch, phase_grads_list_nch = [], [], []
    # phase_grads_std = []
    if shift_by is not None:
        shift_string = f'/shift_{shift_by}/'
    else:
        shift_string = ''
    for module_name in select_modules:
        amp_grads = np.concatenate(amp_gradient_dict[module_name], axis=1)
        amp_grads_mch = np.concatenate(amp_gradient_dict_mch[module_name], axis=1)
        amp_grads_nch = np.concatenate(amp_gradient_dict_nch[module_name], axis=1)
        # Path(f'{output_dir}/{gradient_save_dir}/{file}/{shift_string}/{prefix}/phase/{module_name}/').mkdir(parents=True,
        #                                                                                      exist_ok=True)
        Path(f'{output_dir}/{gradient_save_dir}/{file}/{shift_string}/{prefix}/amps/{module_name}/').mkdir(parents=True,
                                                                                            exist_ok=True)

        print('saving gradients to:', f'{home}/outputs/{gradient_save_dir}/{file}/{shift_string}/{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_ALLCH')
        np.save(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_string}/{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_ALLCH',
            amp_grads)
        np.save(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_string}/{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_MCH',
            amp_grads_mch)
        np.save(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_string}/{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_NCH',
            amp_grads_nch)

        print('concatenated grads shape:', amp_grads.shape)
        # phase_grads = np.concatenate(phase_gradient_dict[module_name], axis=1)
        # phase_grads_mch = np.concatenate(phase_gradient_dict_mch[module_name], axis=1)
        # phase_grads_nch = np.concatenate(phase_gradient_dict_nch[module_name], axis=1)

        # np.save(
        #     f'{home}/outputs/all_layer_gradients/{file}/shift_{shift_by}/{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_ALLCH',
        #     phase_grads)
        # np.save(
        #     f'{home}/outputs/all_layer_gradients/{file}/shift_{shift_by}/{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_MCH',
        #     phase_grads_mch)
        # np.save(
        #     f'{home}/outputs/all_layer_gradients/{file}/shift_{shift_by}/{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_NCH',
        #     phase_grads_nch)

        amp_grads_list.append(amp_grads)
        amp_grads_list_mch.append(amp_grads_mch)
        amp_grads_list_nch.append(amp_grads_nch)
        amp_grads_std.append(np.std(np.abs(amp_grads), axis=(0, 1)))

        # phase_grads_list.append(phase_grads)
        # phase_grads_std.append(np.std(np.abs(phase_grads), axis=(0, 1)))
        # phase_grads_list_mch.append(phase_grads_mch)
        # phase_grads_list_nch.append(phase_grads_nch)
    phase_grads_list, phase_grads_list_mch, phase_grads_list_nch, phase_grads_std = None, None, None, None
    return X_reshaped_list, amp_grads_list, amp_grads_list_mch, amp_grads_list_nch, amp_grads_std, phase_grads_list, phase_grads_list_mch, phase_grads_list_nch, phase_grads_std


def get_gradients_for_intermediate_layers_from_np_arrays(file, prefix, modules, train_mode, eval_mode, shift_by=None):
    amp_grads, amp_grads_mch, amp_grads_nch, phase_grads, phase_grads_mch, phase_grads_nch = [], [], [], [], [], []
    kernel_size, dilation = get_kernel_and_dilation_from_long_name(file)
    X_reshaped_list = []
    for module_name in modules:
        if shift_by is not None:
            shift_by_str = f'/shift_{shift_by}/'
        else:
            shift_by_str = ''
        max_k, max_l = get_num_of_predictions(kernel_size, dilation, layer=module_name)
        X_reshaped_list.append(np.zeros([1, 1, input_time_length - max_k + 1, 1]))
        print(module_name, prefix)
        amp_grads.append(np.load(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_by_str}{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_ALLCH.npy'))
        amp_grads_mch.append(np.load(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_by_str}{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_MCH.npy'))
        amp_grads_nch.append(np.load(
            f'{home}/outputs/{gradient_save_dir}/{file}/{shift_by_str}{prefix}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_NCH.npy'))
        # phase_grads.append(np.load(
        #     f'{home}/outputs/all_layer_gradients/{file}/{shift_by_str}{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_ALLCH.npy'))
        # phase_grads_mch.append(np.load(
        #     f'{home}/outputs/all_layer_gradients/{file}/{shift_by_str}{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_MCH.npy'))
        # phase_grads_nch.append(np.load(
        #     f'{home}/outputs/all_layer_gradients/{file}/{shift_by_str}{prefix}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_NCH.npy'))
    return X_reshaped_list, amp_grads, amp_grads_mch, amp_grads_nch, phase_grads, phase_grads_mch, phase_grads_nch

parser = argparse.ArgumentParser()
parser.add_argument('--variable', default='vel', type=str)
parser.add_argument('--channels', default=None, type=str)
parser.add_argument('--shifts', default=None, type=int, nargs=4)

if __name__ == '__main__':
    select_modules = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    args = parser.parse_args()
    print(cuda)
    variable = args.variable
    if variable == 'vel':
        trajectory_index = 0
    else:
        trajectory_index = 1

    files = [f'{variable}_k3_d3', f'{variable}_k2_d3',
             f'{variable}_k1_d3', ]
    files2 = [f'{variable}_k2_d2', f'{variable}_k3_d1',
              ]
    files3 =[f'{variable}_k3_d3']
    whiten = False
    saved_model_dir = 'lr_0.001'
    shift_string = ''
    gradient_save_dir = 'all_layer_gradients'
    if whiten:
        saved_model_dir = 'pre_whitened'
        gradient_save_dir = 'all_layer_grads_pw'

    prefixes = ['sbp0_m']

    # shifts6 = [-250, -225, -200, -175, -150, -125]
    # shifts = [-100, -75, -50]
    # shifts2 = [-25, 0, 25, 50, 75]
    # shifts3 = [100, 125, 150]
    # shifts4 = [175, 200, 225, 250]
    shifts = args.shifts
    if shifts == [150, 175, 200, 225]:
        shifts += [250]
    print('shifts:', shifts)
    shift = [True, True]
    # high_pass = [False, True]
    high_pass = [False]
    low_pass = [False, False, False, False]

    if args.channels is not None:
        if args.channels == 'MCH':
            motor_channels = True
        else:
            motor_channels = False
    else:
        motor_channels = None

    trained_modes = ['trained']
    eval_modes = ['train']
    if motor_channels is not None:
        if motor_channels:
            m_ch_note = '_MCH'
        else:
            m_ch_note = '_NCH'
    else:
        m_ch_note = ''
    print(prefixes)
    for i, prefix in enumerate(prefixes):
        for file in files3:
            for s in shifts:
                for train_mode in trained_modes:
                    for eval_mode in eval_modes:
                        Path(f'{output_dir}/{gradient_save_dir}/{file}/shift_{s}/{prefix}').mkdir(parents=True, exist_ok=True)
                        X_reshaped_list, amp_grads_list, amp_grads_list_mch, amp_grads_list_nch, amp_grads_std, phase_grads_list, phase_grads_list_mch, phase_grads_list_nch, phase_grads_std = get_gradients_for_intermediate_layers(select_modules, prefix, file=file, shift=shift[i], high_pass=high_pass[i], trajectory_index=trajectory_index, motor_channels=motor_channels, low_pass=low_pass[i], shift_by=s, saved_model_dir=saved_model_dir, whiten=whiten, gradient_save_dir=gradient_save_dir)
                        # X_reshaped_list, amp_grads_list, amp_grads_list_mch, amp_grads_list_nch, phase_grads_list, phase_grads_list_mch, phase_grads_list_nch = get_gradients_for_intermediate_layers_from_np_arrays(file, prefix, modules=select_modules, train_mode=train_mode, eval_mode=eval_mode, shift_by=s)
                        print(prefix, file, train_mode, eval_mode)
                        print('shift:', s)
                        # plot_all_module_gradients(select_modules, X_reshaped_list,
