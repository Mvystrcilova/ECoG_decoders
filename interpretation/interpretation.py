import matplotlib.pyplot as plt
import numpy as np
from braindecode.util import np_to_var, var_to_np
import torch
from global_config import home, output_dir


def get_corr_coef(dataset, model):
    with torch.no_grad():
        outs = model(np_to_var(dataset.X).double())

    all_y = np.array(dataset.y)
    preds = var_to_np(outs)

    preds_flat = np.concatenate(preds)
    y_flat = np.concatenate(all_y[:, -preds.shape[1]:])

    corrcoef = np.corrcoef(y_flat, preds_flat)[0, 1]
    return corrcoef


def reshape_Xs(wSize, X_reshaped):
    nWindows = int(X_reshaped.shape[2] / wSize)
    # will move windows into batch axis
    # so first ensure they are exactly divisible, by cutting out rest
    X_reshaped = X_reshaped[:, :, :wSize * int(X_reshaped.shape[2] / wSize)]
    shape_tmp = X_reshaped.shape
    # now make into windows x window size
    X_reshaped = X_reshaped.reshape(shape_tmp[0], shape_tmp[1], nWindows, wSize)
    # to #batches x #windows x #channels x #window size
    X_reshaped = X_reshaped.transpose(0, 2, 1, 3)
    # now cat batches and windows into one axis
    X_reshaped = X_reshaped.reshape(shape_tmp[0] * nWindows, shape_tmp[1], wSize, 1)
    return X_reshaped


def calculate_phase_and_amps(batch_X):
    ffted = np.fft.rfft(batch_X, axis=2)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    amps_th = np_to_var(amps, requires_grad=True, dtype=np.float32)
    phases_th = np_to_var(phases, requires_grad=True, dtype=np.float32)

    fft_coefs = amps_th.unsqueeze(-1) * torch.stack((torch.cos(phases_th), torch.sin(phases_th)), dim=-1)
    fft_coefs = fft_coefs.squeeze(3)

    iffted = torch.irfft(fft_coefs, signal_ndim=1, signal_sizes=(batch_X.shape[2],)).unsqueeze(-1)
    return iffted, amps_th, phases_th


def get_outs_shape(outs_shape, outs):
    if outs_shape == 'last_window':
        return outs[:, :, -1:, ]
    if outs_shape == 'random_window':
        return outs[:, :, 178:179, :]
    if outs_shape == 'absolute_full_window':
        return torch.mean(torch.abs(outs[:, :, :, ]))
    else:
        return torch.mean(outs[:, :, outs_shape, ])


def plot_correlation(batch_X, new_model, coef, outs_shape=None, setname=''):
    iffted, amps_th, phases_th = calculate_phase_and_amps(batch_X)
    outs = new_model(iffted.double())

    assert outs.shape[1] == 1
    if outs_shape is not None:
        mean_out = get_outs_shape(outs_shape, outs)
    else:
        mean_out = torch.mean(outs)
    mean_out.backward(torch.tensor(2), retain_graph=True)
    amp_grads = var_to_np(amps_th.grad).squeeze(-1)
    plot_gradients(batch_X, np.mean(amp_grads, axis=(0, 1)), corrcoef=coef, wsize=setname)
    plot_gradients(batch_X, np.mean(np.abs(amp_grads), axis=(0, 1)), coef, 'Absolute ', wsize=setname)
    return outs


def plot_gradients(batch_X, y, corrcoef, title_prefix='', setname='', wsize=''):
    fig = plt.figure(figsize=(12, 4))
    plt.plot(np.fft.rfftfreq(batch_X.shape[2], 1 / 250.0), y)
    plt.axhline(y=0, color='black')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gradient')
    plt.title("{:s} {:s}Amplitude Gradients (Corr {:.2f}%, {})".format(title_prefix, setname, corrcoef * 100, wsize))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/graphs/train/{title_prefix}_Amplitude_Gradients_corr{setname}%_{wsize}.png')
    plt.show()

    plt.close(fig)
