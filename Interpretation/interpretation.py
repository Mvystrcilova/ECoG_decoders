from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from braindecode.util import np_to_var, var_to_np
import torch
from global_config import cuda


def get_corr_coef(dataset, model):
    len = int(dataset.X.shape[0]/2)
    print(len)
    with torch.no_grad():
        if cuda:
            outs1 = model.double()(np_to_var(dataset.X[:len]).cuda())
            outs2 = model.double()(np_to_var(dataset.X[len:]).cuda())
            # outs = model.float()(np_to_var(dataset.X).float().cuda())
        else:
            # outs = model(np_to_var(dataset.X).double())
            outs = model.double()(np_to_var(dataset.X))


    all_y = np.array(dataset.y.cpu())
    if cuda:
        preds1 = var_to_np(outs1)
        preds2 = var_to_np(outs2)
        preds = np.concatenate([preds1, preds2])
        # preds = var_to_np(outs)
        print(preds.shape)
    else:
        preds = var_to_np(outs)

    preds_flat = np.concatenate(preds)
    y_flat = np.concatenate(all_y[:, -preds.shape[1]:])
    # corrcoef = np.corrcoef(np.abs(y_flat), np.abs(preds_flat))[0, 1]
    corrcoef = np.corrcoef(y_flat, preds_flat)[0, 1]
    return corrcoef


def reshape_Xs(wSize, X_reshaped):
    """
    :param wSize: size of the input window
    :param X_reshaped: batch of cropped intputs which is reshaped
    :return: reshaped version of X_reshaped
    """
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
    """
    Hooks the amplitude and phase of the signal in batch_X
    :param batch_X: one training batch
    :return: returns the original signal with hooked amplitudes and phases
    """
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


def get_outs(batch_X, new_model, outs_shape=None, grad_type='amps'):
    """gets output of the network when signal with amplitudes and frequencies is given on input"""
    iffted, amps_th, phases_th = calculate_phase_and_amps(batch_X)
    if cuda:
        outs = new_model(iffted.double().cuda())
    else:
        outs = new_model(iffted.double())

    assert outs.shape[1] == 1
    if outs_shape is not None:
        mean_out = get_outs_shape(outs_shape, outs)
    else:
        mean_out = torch.mean(outs)
    mean_out.backward(retain_graph=True)
    amp_grads = var_to_np(amps_th.grad).squeeze(-1)
    phase_grads = var_to_np(phases_th.grad).squeeze(-1)
    if grad_type == 'amps':
        return amp_grads, outs
    else:
        return phase_grads, outs
