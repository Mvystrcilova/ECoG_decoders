import numpy as np
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var, var_to_np
from matplotlib import pyplot as plt  # equiv. to: import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm.autonotebook import tqdm
from cycler import cycler
from interpretation import get_corr_coef, reshape_Xs, calculate_phase_and_amps, plot_correlation, \
    plot_gradients
from models.Model import load_model, create_new_model
import sys
import matplotlib
import seaborn
from global_config import home, input_time_length
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from data.pre_processing import Data


matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14

seaborn.set_style('darkgrid')


def plot_individual_gradients(batch_X, amp_grads_per_crop, setname, coef):
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
    plt.show()
    plt.figure(figsize=(18, 5))
    plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), np.mean(amp_grads_per_crop, axis=(1)).T, lw=0.25,
             color=seaborn.color_palette()[0])
    plt.axhline(y=0, color='black')

    plt.xlim(20, 30)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.title("{:s} Amplitude Gradients (Corr {:.2f}%)".format(setname, corrcoef * 100))
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


def manually_manipulate_signal(X_reshaped):
    for freq in (250 / 3, 60, 40, 250 / (3 ** 2), 250 / (3 ** 3)):
        with torch.no_grad():
            batch_X = X_reshaped[:1]
            outs = new_model(np_to_var(batch_X))
            sine = np.sin(np.linspace(0, freq * np.pi * 2 * batch_X.shape[2] / 250,
                                      batch_X.shape[2]))
            changed_X = batch_X + sine[None, None, :, None] * 0.5
            changed_outs = new_model(np_to_var(changed_X, dtype=np.float32).double())

        plt.figure(figsize=(16, 4))
        plt.plot(batch_X[0, 0, :250, 0], )
        plt.plot(changed_X[0, 0, :250, 0])
        plt.legend(("Original X", "Changed X"))
        plt.xlabel("Timestep")
        plt.title(f"Frequency {freq:.2f} Hz")
        plt.show()

        plt.figure(figsize=(16, 4))
        plt.plot(var_to_np(outs.squeeze()))
        plt.plot(var_to_np(changed_outs.squeeze()))
        plt.legend(("Original out", "Changed out"))
        plt.xlabel("Timestep")
        plt.show()

        plt.figure(figsize=(16, 4))
        plt.plot(var_to_np(changed_outs.squeeze()) - var_to_np(outs.squeeze()))
        plt.plot(sine[-len(outs.squeeze()):] * 0.4)
        plt.legend(("Out diff", "Added sine last part"))
        plt.xlabel("Timestep")
        plt.show()

        plt.figure(figsize=(16, 4))
        plt.plot(np.fft.rfftfreq(len(changed_outs.squeeze()), 1 / 250.0),
                 np.abs(np.fft.rfft(var_to_np(changed_outs.squeeze()) - var_to_np(outs.squeeze()))))
        plt.plot(np.fft.rfftfreq(len(changed_outs.squeeze()), 1 / 250.0),
                 np.abs(np.fft.rfft(sine[-len(outs.squeeze()):] * 0.4)))
        plt.legend(("Out diff", "Added sine last part"))
        plt.xlabel("Frequency [Hz]")
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


if __name__ == '__main__':
    model_file = '/models/saved_models/best_model_1'
    model = load_model(model_file)
    data = Data(home + '/previous_work/ALL_11_FR1_day1_absVel.mat', -1)
    n_preds_per_input = get_output_shape(model, data.in_channels, 1200)[1]
    data.cut_input(input_time_length, n_preds_per_input, False)
    train_set, test_set = data.train_set, data.test_set

    wSizes = [2 * n_preds_per_input, 682]
    corrcoef = get_corr_coef(train_set, model)

    X_reshaped = np.asarray(train_set.X)
    print(X_reshaped.shape)
    X_reshaped = reshape_Xs(wSizes[0], X_reshaped, 'Train', corrcoef)

    new_model = create_new_model(model, 'conv_classifier')
    with torch.no_grad():
        test_out = new_model(np_to_var(X_reshaped[:2]))
    new_model.eval()
    n_filters = test_out.shape[1]

    print('Full window size')

    batch_X = X_reshaped[:1]
    plot_correlation(batch_X, new_model, corrcoef)

    print('Smaller window size')

    batch_X = X_reshaped[:1, :, :682]
    plot_correlation(batch_X, new_model, corrcoef, None, setname='Smaller window')

    print('Last window only')
    batch_X = X_reshaped[:1]
    plot_correlation(batch_X, new_model, corrcoef, 'last_window', 'Last window')

    print('Random window')
    plot_correlation(batch_X, new_model, corrcoef, 'random_window', 'Random window')

    print('Absolute full window')
    outs = plot_correlation(batch_X, new_model, corrcoef, 'absolute_full_window', 'Absolute Full Window')

    amp_grads_per_crop = get_amp_grads_per_crops(outs, X_reshaped)
    plot_gradients(batch_X, np.mean(amp_grads_per_crop, axis=(0, 1)), corrcoef=corrcoef,
                   title_prefix='Train', wsize='Full windows crops')
    plot_individual_gradients(batch_X, amp_grads_per_crop, 'Train', corrcoef)
    manually_manipulate_signal(X_reshaped)






