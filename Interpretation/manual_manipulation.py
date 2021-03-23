from pathlib import Path

import numpy as np
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var, var_to_np
from matplotlib import pyplot as plt  # equiv. to: import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm.autonotebook import tqdm
from cycler import cycler
from Interpretation.interpretation import get_corr_coef, reshape_Xs, calculate_phase_and_amps, plot_correlation, \
    plot_gradients
from models.Model import load_model, create_new_model
import sys
import matplotlib
import seaborn
from global_config import home, input_time_length, output_dir, random_seed, cuda
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from data.pre_processing import Data, get_num_of_channels
from torchsummary import summary
import random

torch.manual_seed(random_seed)
random.seed(random_seed)

matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14

seaborn.set_style('darkgrid')


def plot_individual_gradients(batch_X, amp_grads_per_crop, setname, coef, output_file):
    print('Plotting individual gradients')
    plt.figure(figsize=(18, 5))
    plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), np.mean(amp_grads_per_crop, axis=(1)).T, lw=0.25,
             color=seaborn.color_palette()[0])
    plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), np.mean(amp_grads_per_crop, axis=(0, 1)), lw=1,
             color=seaborn.color_palette()[1])
    plt.axhline(y=0, color='black')

    plt.xlim(80, 90)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.title("{:s} Amplitude Individual Crop Gradients (Corr {:.2f}%)".format(setname, corrcoef * 100))
    plt.tight_layout()
    plt.savefig(f'{output_file}/individual_gradients_80_90.png')

    plt.show()
    plt.figure(figsize=(18, 5))
    plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), np.mean(amp_grads_per_crop, axis=(1)).T, lw=0.25,
             color=seaborn.color_palette()[0])
    plt.axhline(y=0, color='black')

    plt.xlim(20, 30)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.title("{:s} Amplitude Gradients (Corr {:.2f}%)".format(setname, corrcoef * 100))
    plt.tight_layout()
    plt.savefig(f'{output_file}/individual_gradients_20_30.png', format='png')
    plt.show()


def plot_colored_amps_per_crops(corrcoef, setname, amp_grads_per_crop):
    with plt.rc_context(rc={'axes.prop_cycle': cycler(
            color=cm.coolwarm(np.linspace(0, 1, len(amp_grads_per_crop))))}):
        plt.figure(figsize=(18, 5))
        plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), np.mean(amp_grads_per_crop, axis=(1)).T, lw=0.25, );
        plt.axhline(y=0, color='black')
        plt.axvline(x=np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0)[344], color='black')
        plt.axvline(x=np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0)[349], color='black')
        plt.axvline(x=np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0)[351], color='black')

        plt.xlim(80, 90)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.title("{:s} Amplitude Gradients (Corr {:.2f}%)".format(setname, corrcoef * 100))


def look_at_interesting_crops(interesting_crops, ):
    pass


def manually_manipulate_signal(X_reshaped, output, model, maxpool_model=False, white_noise=True):
    if white_noise:
        freqs = ['white_noise']
    else:
         freqs = (250 / 3, 60, 40, 250 / (3 ** 2), 250 / (3 ** 3))
    for freq in freqs:
        with torch.no_grad():
            batch_X = X_reshaped[:1]
            if maxpool_model:
                outs = model(np_to_var(batch_X))
                outs = np.mean(np.asarray(outs), axis=1)
            else:
                outs = model(np_to_var(batch_X))
            if white_noise:
                sine = np.random.normal(0, 1, batch_X.shape[2])
            else:
                sine = np.sin(np.linspace(0, freq * np.pi * 2 * batch_X.shape[2] / 250,
                                      batch_X.shape[2]))
            changed_X = batch_X + sine[None, None, :, None] * 0.5
            if maxpool_model:
                changed_outs = np.mean(np.asarray(model(np_to_var(changed_X, dtype=np.float32).double())), axis=1)
            else:
                changed_outs = model(np_to_var(changed_X))

        plt.figure(figsize=(16, 4))
        plt.plot(batch_X[0, 0, 1:250, 0], )
        plt.plot(changed_X[0, 0, 1:250, 0])
        plt.legend(("Original X", "Changed X"))
        plt.xlabel("Timestep")
        if not white_noise:
            freq = f'{freq:.2f}'
        plt.title(f"Frequency {freq} Hz")
        plt.tight_layout()
        plt.savefig(f'{output}/mm_original_and_changed_{freq}Hz.png', format='png')
        plt.show()

        plt.figure(figsize=(16, 4))
        if maxpool_model:
            plt.plot(np.squeeze(outs))
            plt.plot(np.squeeze(changed_outs))
        else:
            plt.plot(var_to_np(outs.squeeze()))
            plt.plot(var_to_np(changed_outs.squeeze()))
        plt.legend(("Original out", "Changed out"))
        plt.xlabel("Timestep")
        plt.title(f"Frequency {freq} Hz")

        plt.tight_layout()
        plt.savefig(f'{output}/mm_original_out_and_changed_out_{freq}Hz.png')
        plt.show()

        plt.figure(figsize=(16, 4))
        if maxpool_model:
            plt.plot(np.squeeze(changed_outs) -np.squeeze(outs))

        else:
            plt.plot(var_to_np(changed_outs.squeeze()) - var_to_np(outs.squeeze()))
        plt.plot(sine[-len(outs.squeeze()):] * 0.4)
        plt.legend(("Out diff", "Added sine last part"))
        plt.title(f"Frequency {freq} Hz")

        plt.tight_layout()
        plt.savefig(f'{output}/mm_output_original_difference_time{freq}Hz.png')
        plt.xlabel("Timestep")
        plt.show()

        plt.figure(figsize=(16, 4))
        if maxpool_model:
            plt.plot(np.fft.rfftfreq(len(np.squeeze(outs)), 1 / 250.0),
                     np.abs(np.fft.rfft(np.squeeze(changed_outs) - np.squeeze(outs))))
            # plt.plot(np.fft.rfftfreq(len(np.squeeze(changed_outs)), 1 / 250.0)[1:],
            #          np.abs(np.fft.rfft(sine[-len(np.squeeze(outs)):] * 0.4))[1:])
        else:
            plt.plot(np.fft.rfftfreq(len(np.squeeze(outs)), 1 / 250.0),
                     np.abs(np.fft.rfft(np.squeeze(changed_outs) - np.squeeze(outs))))
            # plt.plot(np.fft.rfftfreq(len(np.squeeze(changed_outs)), 1 / 250.0)[1:],
            #          np.abs(np.fft.rfft(sine[-len(np.squeeze(outs)):] * 0.4))[1:])
        plt.legend(("Out diff", "Added sine last part"))
        plt.xlabel("Frequency [Hz]")
        plt.title(f"Frequency {freq} Hz")
        plt.tight_layout()
        plt.savefig(f'{output}/mm_output_original_difference_time_frequency_{freq}Hz.png')

        plt.show()


def get_amp_grads_per_crops(outs, X_reshaped):
    amp_grads_per_crop = []
    for i_time in tqdm(range(outs.shape[2])):
        batch_X = X_reshaped[:1]
        iffted, amps_th, phases_th = calculate_phase_and_amps(batch_X)
        outs = new_model(iffted.double())
        assert outs.shape[1] == 1
        mean_out = torch.mean(outs[:, :, i_time, ])
        mean_out.backward(retain_graph=True)
        amp_grads = var_to_np(amps_th.grad).squeeze(-1)
        amp_grads_per_crop.append(amp_grads)
    amp_grads_per_crop = np.array(amp_grads_per_crop)
    print(amp_grads_per_crop.shape)
    amp_grads_per_crop = amp_grads_per_crop.squeeze()
    print(amp_grads_per_crop.shape)

    return amp_grads_per_crop

variable = 'absVel'
model_prefix = 'm'

trained_modes = ['trained', 'untrained']
eval_modes = ['train', 'validation']
cropped = False


def prepare_for_gradients(patient_index, model_name, trained_mode, eval_mode, saved_model_dir, model_file=None, shift=False, high_pass=False, trajectory_index=0,
                          multi_layer=False, motor_channels=None, low_pass=False, shift_by=None, whiten=False):
    if shift_by is not None:
        shift_str = f'shift_{shift_by}'
        model_name_list = model_name.split('/')
        model_name_list = [model_name_list[0], shift_str, model_name_list[1]]
        model_name = '/'.join(model_name_list)
        index = 2
        random_valid = False
    else:
        index = 1
        shift_str = f''
        random_valid = True
    if model_file is None:
        if '/' in model_name:
            other_model_name = model_name.split('/')[index] + f'_p_{patient_index}'
        else:
            other_model_name = f'{model_name}_p_{patient_index}'
        if trained_mode == 'untrained':

            model_file = f'/models/saved_models/{model_name}/{other_model_name}/initial_{other_model_name}'
        else:
            # model_file = f'/models/saved_models/{model_name}/{other_model_name}/last_model'
            model_file = f'/models/saved_models/{model_name}/{other_model_name}/best_model_split_0'
    output = f'{output_dir}/hp_graphs/{model_name}/{eval_mode}/{trained_mode}/'
    # Path(output).mkdir(parents=True, exist_ok=True)
    model = load_model(model_file)
    print(model_file)
    print('motor channels:', motor_channels)

    in_channels = get_num_of_channels(home + f'/previous_work/P{patient_index}_data.mat')
    n_preds_per_input = get_output_shape(model, in_channels, 1200)[1]
    shift_window = input_time_length - n_preds_per_input + 1
    small_window = min((input_time_length - n_preds_per_input)*2, 1200)
    print('small window:', small_window, model_name)
    print('shift window:', shift_window, shift)
    if shift_by is None:
        shift_index = int(shift_window / 2)
    else:
        shift_index = int((shift_window / 2) - shift_by)

    data = Data(home + f'/previous_work/P{patient_index}_data.mat', -1, low_pass=low_pass, trajectory_index=trajectory_index,
                shift_data=shift, high_pass=high_pass, shift_by=shift_index, pre_whiten=whiten, random_valid=random_valid)

    data.cut_input(input_time_length, n_preds_per_input, False)
    train_set, test_set = data.train_set, data.test_set
    corrcoef = get_corr_coef(train_set, model)
    num_channels = None

    if eval_mode == 'validation':
        train_set = test_set

    X_reshaped = np.asarray(train_set.X)
    print(X_reshaped.shape)
    X_reshaped = reshape_Xs(input_time_length, X_reshaped)
    # summary(model.float(), input_size=(data.in_channels, 683, 1))
    if not multi_layer:
        new_model = create_new_model(model, 'conv_classifier', input_channels=num_channels)
    else:
        new_model = model
    # with torch.no_grad():
    #     test_out = new_model(np_to_var(X_reshaped[:2]).double())
    new_model.eval()
    # n_filters = test_out.shape[1]

    return corrcoef, new_model, X_reshaped, small_window, output, data.motor_channels, data.non_motor_channels

model_names = []
if __name__ == '__main__':
    if cropped:
        model_string = 'cropped_model'
    else:
        model_string = 'model'
    # model_name = f'{model_string}_strides_3333'
    for model_name in model_names:
        for patient_index in range(8, 9):
            model_name = f'{model_name}'
            for trained_mode in trained_modes:
                for eval_mode in eval_modes:
                    corrcoef, new_model, X_reshaped, small_window, output = prepare_for_gradients(patient_index, model_name, trained_mode, eval_mode)

                    print('Full window size')

                    batch_X = X_reshaped[:1]
                    plot_correlation(batch_X, new_model, corrcoef, output_file=output)

                    print('Smaller window size')

                    batch_X = X_reshaped[:1, :, :small_window]
                    plot_correlation(batch_X, new_model, corrcoef, output, None, setname='Smaller window')

                    print('Last window only')
                    batch_X = X_reshaped[:1]
                    plot_correlation(batch_X, new_model, corrcoef, output, 'last_window', 'Last window')

                    print('Random window')
                    plot_correlation(batch_X, new_model, corrcoef, output, 'random_window', 'Random window')

                    print('Absolute full window')
                    outs = plot_correlation(batch_X, new_model, corrcoef, output, 'absolute_full_window', 'Absolute Full Window')

                    amp_grads_per_crop = get_amp_grads_per_crops(outs, X_reshaped)
                    plot_gradients(batch_X, np.mean(amp_grads_per_crop, axis=(0, 1)), corrcoef=corrcoef,
                                   title_prefix='Train', wsize='Full windows crops', output_file=output)
                    plot_individual_gradients(batch_X, amp_grads_per_crop, 'Train', corrcoef, output)
                    manually_manipulate_signal(X_reshaped, output, new_model, white_noise=True, maxpool_model=False)
                    manually_manipulate_signal(X_reshaped, output, new_model, white_noise=False, maxpool_model=False)







