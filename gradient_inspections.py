import argparse
import logging
from pathlib import Path

import matplotlib
from braindecode.util import np_to_var, var_to_np

from Interpretation.interpretation import calculate_phase_and_amps
from Interpretation.manual_manipulation import prepare_for_gradients
from global_config import output_dir, cuda, input_time_length, home

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


def plot_all_module_gradients(titles, batch_X, gradients, gradient_stds, output_file):
    fig, ax = plt.subplots(2, 2, sharey='row', figsize=(15, 11))
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, gradient in enumerate(gradients[0]):
        y = np.mean(np.abs(gradient), axis=(0, 1))
        mch = np.mean(np.abs(gradients[1][i]), axis=(0, 1))
        nch = np.mean(np.abs(gradients[2][i]), axis=(0, 1))
        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), y, color='steelblue', label='All channels')
        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), mch, color='limegreen',
                            label='Motor channels')
        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), nch, color='lightcoral',
                            label='Non-motor channels')

        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), y - gradient_stds[i],
                            color='lightsteelblue', label='All channel std')
        ax[indices[i]].plot(np.fft.rfftfreq(batch_X[i].shape[2], 1 / 250.0), y + gradient_stds[i],
                            color='lightsteelblue')
        ax[indices[i]].set_title(titles[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def get_module_gradients(model, module_name, X_reshaped):
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
    small_window = input_time_length - n_preds + 1
    new_X_reshaped = X_reshaped[:, :, :small_window, :]

    # filters x windows x channels x freqs
    all_amp_grads = np.ones(
        (n_filters,) + new_X_reshaped.shape[:2] + (len(np.fft.rfftfreq(new_X_reshaped.shape[2], d=1 / 250.0)),),
        dtype=np.float32) * np.nan
    all_phases_grads = np.ones(
        (n_filters,) + new_X_reshaped.shape[:2] + (len(np.fft.rfftfreq(new_X_reshaped.shape[2], d=1 / 250.0)),),
        dtype=np.float32) * np.nan

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
            all_phases_grads[i_filter, i_start:i_start + len(phases_grads)] = phases_grads.squeeze(-1)
            amps_th.grad.zero_()
            phases_th.grad.zero_()

        i_start += len(amp_grads)

    del amp_grads  # just make sure I don't use it accidentally now
    del phases_grads  # just make sure I don't use it accidentally now
    assert i_start == all_amp_grads.shape[1]
    assert not np.any(np.isnan(all_amp_grads))
    assert i_start == all_phases_grads.shape[1]
    assert not np.any(np.isnan(all_phases_grads))
    # mean across windows
    meaned_amp_grads = np.mean(all_amp_grads, axis=1)
    meaned_phase_grads = np.mean(all_phases_grads, axis=1)
    # phase_grads_list.append(meaned_phase_grads)
    # amp_grads_list.append(meaned_amp_grads)
    print('grads shape:', meaned_phase_grads.shape)
    return meaned_amp_grads, meaned_phase_grads, small_window


parser = argparse.ArgumentParser()
parser.add_argument('--variable', default='vel', type=str)
parser.add_argument('--channels', default=None, type=str)

if __name__ == '__main__':
    select_modules = ['conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    args = parser.parse_args()
    variable = args.variable
    if variable == 'vel':
        trajectory_index = 0
    else:
        trajectory_index = 1

    files = [f'{variable}_k_3333', f'{variable}_k_1111', f'{variable}_k_2222_dilations_1111', f'{variable}_k_2222',
             f'{variable}_k_2222_dilations_24816']
    files2 = [f'{variable}_k_3333_dilations_1111',
              f'{variable}_k_3333_dilations_24816', f'{variable}_k_4444', f'{variable}_k_4444_dilations_1111',
              f'{variable}_k_4444_dilations_24816']

    # files = [f'{variable}_k_4444_dilations_24816']
    prefixes = ['m', 's2_m', 'hp_m', 'hp_sm2']

    titles = ['Initial performance', 'Shifted performance', 'Initial high-pass performance',
              'Shifted high-pass performance']
    shift = [False, True, False, True]
    high_pass = [False, False, True, True]

    if args.channels is not None:
        if args.channels == 'MCH':
            motor_channels = True
        else:
            motor_channels = False
    else:
        motor_channels = None

    trained_modes = ['trained']
    eval_modes = ['train', 'validation']

    if motor_channels is not None:
        if motor_channels:
            m_ch_note = '_MCH'
        else:
            m_ch_note = '_NCH'
    else:
        m_ch_note = ''
    for file in files2:
        for i, prefix in enumerate(prefixes):
            for train_mode in trained_modes:
                for eval_mode in eval_modes:
                    output_amp = f'{output_dir}/all_layer_grads/{file}/{prefix}/amps_avg_{prefix}_{file}_{train_mode}_{eval_mode}{m_ch_note}_sw.png'
                    output_phase = f'{output_dir}/all_layer_grads/{file}/{prefix}/phase_avg_{prefix}_{file}_{train_mode}_{eval_mode}{m_ch_note}_sw.png'

                    Path(f'{output_dir}/all_layer_grads/{file}/{prefix}').mkdir(parents=True, exist_ok=True)

                    amp_gradient_dict = {module_name: [] for module_name in select_modules}
                    amp_gradient_dict_mch = {module_name: [] for module_name in select_modules}
                    amp_gradient_dict_nch = {module_name: [] for module_name in select_modules}

                    phase_gradient_dict_mch = {module_name: [] for module_name in select_modules}
                    phase_gradient_dict_nch = {module_name: [] for module_name in select_modules}
                    phase_gradient_dict = {module_name: [] for module_name in select_modules}

                    X_reshaped_list, X_reshaped_list_mch, X_reshaped_list_nch = [], [], []
                    for patient_index in range(1, 13):
                        model_name = f'{prefix}_{file}'
                        print(patient_index, model_name)
                        corrcoef, new_model, X_reshaped, small_window, _, motor_channel_indices, non_motor_channel_indices = prepare_for_gradients(
                            patient_index,
                            f'lr_0.001/{model_name}',
                            train_mode, eval_mode,
                            shift=shift[i],
                            high_pass=high_pass[i],
                            trajectory_index=trajectory_index,
                            multi_layer=True,
                            motor_channels=motor_channels)

                        amp_grads_list, amp_grads_list_mch, amp_grads_list_nch = [], [], []
                        phase_grads_list, phase_grads_list_mch, phase_grads_list_nch = [], [], []
                        # X_reshaped = X_reshaped[:, :, :small_window]
                        for i, module_name in enumerate(select_modules):
                            amp_grads, phase_grads, module_filters = get_module_gradients(new_model, module_name,
                                                                                          X_reshaped)
                            amp_grads_mch, phase_grads_mch = np.take(amp_grads, motor_channel_indices.astype(int),
                                                                     axis=1), np.take(phase_grads,
                                                                                      motor_channel_indices.astype(int),
                                                                                      axis=1)
                            amp_grads_nch, phase_grads_nch = np.take(amp_grads, non_motor_channel_indices.astype(int),
                                                                     axis=1), np.take(phase_grads,
                                                                                      non_motor_channel_indices.astype(
                                                                                          int), axis=1)

                            amp_gradient_dict[module_name].append(amp_grads)
                            amp_gradient_dict_mch[module_name].append(amp_grads_mch)
                            amp_gradient_dict_nch[module_name].append(amp_grads_nch)

                            phase_gradient_dict[module_name].append(phase_grads)
                            phase_gradient_dict_mch[module_name].append(phase_grads_mch)
                            phase_gradient_dict_nch[module_name].append(phase_grads_nch)

                            if len(X_reshaped_list) < 4:
                                X_reshaped_list.append(X_reshaped[:, :, :module_filters])

                    amp_grads_list, amp_grads_list_mch, amp_grads_list_nch = [], [], []
                    amp_grads_std = []
                    phase_grads_list, phase_grads_list_mch, phase_grads_list_nch = [], [], []
                    phase_grads_std = []
                    for module_name in select_modules:
                        amp_grads = np.concatenate(amp_gradient_dict[module_name], axis=1)
                        amp_grads_mch = np.concatenate(amp_gradient_dict_mch[module_name], axis=1)
                        amp_grads_nch = np.concatenate(amp_gradient_dict_nch[module_name], axis=1)
                        Path(f'{output_dir}/all_layer_gradients/{file}/phase/{module_name}/').mkdir(parents=True,
                                                                                                    exist_ok=True)
                        Path(f'{output_dir}/all_layer_gradients/{file}/amps/{module_name}/').mkdir(parents=True,
                                                                                                   exist_ok=True)

                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_ALLCH',
                            amp_grads)
                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_MCH',
                            amp_grads_mch)
                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/amps/{module_name}/amps_avg_{file}_{train_mode}_{eval_mode}_NCH',
                            amp_grads_nch)

                        print('concatenated grads shape:', amp_grads.shape)
                        phase_grads = np.concatenate(phase_gradient_dict[module_name], axis=1)
                        phase_grads_mch = np.concatenate(phase_gradient_dict_mch[module_name], axis=1)
                        phase_grads_nch = np.concatenate(phase_gradient_dict_nch[module_name], axis=1)

                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_ALLCH',
                            phase_grads)
                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_MCH',
                            phase_grads_mch)
                        np.save(
                            f'{home}/outputs/all_layer_gradients/{file}/phase/{module_name}/phase_avg_{file}_{train_mode}_{eval_mode}_NCH',
                            phase_grads_nch)

                        amp_grads_list.append(amp_grads)
                        amp_grads_list_mch.append(amp_grads_mch)
                        amp_grads_list_nch.append(amp_grads_nch)
                        amp_grads_std.append(np.std(np.abs(amp_grads), axis=(0, 1)))

                        phase_grads_list.append(phase_grads)
                        phase_grads_std.append(np.std(np.abs(phase_grads), axis=(0, 1)))
                        phase_grads_list_mch.append(phase_grads_mch)
                        phase_grads_list_nch.append(phase_grads_nch)

                    plot_all_module_gradients(select_modules, X_reshaped_list,
                                              [amp_grads_list, amp_grads_list_mch, amp_grads_list_nch], amp_grads_std,
                                              output_file=output_amp)
                    plot_all_module_gradients(select_modules, X_reshaped_list,
                                              [phase_grads_list, phase_grads_list_mch, phase_grads_list_nch],
                                              phase_grads_std,
                                              output_file=output_phase)
