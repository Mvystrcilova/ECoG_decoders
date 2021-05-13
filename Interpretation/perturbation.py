from braindecode.models.util import get_output_shape
from torch import nn
import torch
import numpy as np
from braindecode.util import get_balanced_batches, wrap_reshape_apply_fn, corr, np_to_var
from skorch.history import History

from Training.CorrelationMonitor1D import CorrelationMonitor1D
from interpretation import reshape_Xs
from models.Model import load_model
from data.pre_processing import Data
from global_config import home, input_time_length, n_perturbations, output_dir, srate
from matplotlib import pyplot as plt

""" Perturbation analysis by Kay Hartmann"""


class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_list):
        """
        Returns intermediate activations of a network during forward pass

        to_select: list of module names for which activation should be returned
        modules_list: Modules of the network in the form [[name1, mod1],[name2,mod2]...)

        Important: modules_list has to include all modules of the network, not only those of interest
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/8
        """
        super(SelectiveSequential, self).__init__()
        for key, module in modules_list:
            self.add_module(key, module)
            self._modules[key].load_state_dict(module.state_dict())
        self._to_select = to_select

    def forward(self, x):
        # Call modules individually and append activation to output if module is in to_select
        o = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                o.append(x)
        return o


def phase_perturbation(amps, phases, rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Shifts spectral phases randomly for input and frequencies, but same for all channels

    amps: Spectral amplitude (not used)
    phases: Spectral phases
    rng: Random Seed

    Output:
        amps_pert: Input amps (not modified)
        phases_pert: Shifted phases
        pert_vals: Absolute phase shifts
    """
    noise_shape = list(phases.shape)
    noise_shape[1] = 1  # Do not sample noise for channels individually

    # Sample phase perturbation noise
    phase_noise = rng.uniform(-np.pi, np.pi, noise_shape).astype(np.float32)
    phase_noise = phase_noise.repeat(phases.shape[1], axis=1)
    # Apply noise to inputs
    phases_pert = phases + phase_noise
    phases_pert[phases_pert < -np.pi] += 2 * np.pi
    phases_pert[phases_pert > np.pi] -= 2 * np.pi

    return amps, phases_pert, np.abs(phase_noise)


def phase_perturbation_chnls(amps, phases, rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Shifts spectral phases randomly for input and frequencies, but same for all channels

    amps: Spectral amplitude (not used)
    phases: Spectral phases
    rng: Random Seed

    Output:
        amps_pert: Input amps (not modified)
        phases_pert: Shifted phases
        pert_vals: Absolute phase shifts
    """
    noise_shape = list(phases.shape)
    #        noise_shape[1] = 1 # Do not sample noise for channels individually

    # Sample phase perturbation noise
    phase_noise = rng.uniform(-np.pi, np.pi, noise_shape).astype(np.float32)
    #        phase_noise = phase_noise.repeat(phases.shape[1],axis=1)
    # Apply noise to inputs
    phases_pert = phases + phase_noise
    phases_pert[phases_pert < -np.pi] += 2 * np.pi
    phases_pert[phases_pert > np.pi] -= 2 * np.pi

    return amps, phases_pert, np.abs(phase_noise)


def amp_perturbation_additive(amps, phases, rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Adds additive noise to amplitudes

    amps: Spectral amplitude
    phases: Spectral phases (not used)
    rng: Random Seed

    Output:
        amps_pert: Scaled amplitudes
        phases_pert: Input phases (not modified)
        pert_vals: Amplitude noise
    """
    amp_noise = rng.normal(0, 1, amps.shape).astype(np.float32)
    amps_pert = amps + amp_noise
    amps_pert[amps_pert < 0] = 0
    return amps_pert, phases, amp_noise


def amp_perturbation_multiplicative(amps, phases, rng=np.random.RandomState()):
    """
    Takes amps and phases of BxCxF with B input, C channels, F frequencies
    Adds multiplicative noise to amplitudes

    amps: Spectral amplitude
    phases: Spectral phases (not used)
    rng: Random Seed

    Output:
        amps_pert: Scaled amplitudes
        phases_pert: Input phases (not modified)
        pert_vals: Amplitude scaling factor
    """
    amp_noise = rng.normal(1, 0.02, amps.shape).astype(np.float32)
    amps_pert = amps * amp_noise
    amps_pert[amps_pert < 0] = 0
    return amps_pert, phases, amp_noise


def correlate_feature_maps(x, y):
    """
    Takes two activation matrices of the form Bx[F]xT where B is batch size, F number of filters (optional) and T time points
    Returns correlations of the corresponding activations over T

    Input: Bx[F]xT (x,y)
    Returns: Bx[F]
    """
    shape_x = x.shape
    shape_y = y.shape
    assert np.array_equal(shape_x, shape_y)
    assert len(shape_x) < 4
    x = x.reshape((-1, shape_x[-1]))
    y = y.reshape((-1, shape_y[-1]))
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    y = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
    tmp_corr = x * y
    corr_ = tmp_corr.sum(axis=1)
    # corr_ = np.zeros((x.shape[0]))
    # for i in range(x.shape[0]):
    #    # Correlation of standardized variables
    #    corr_[i] = np.correlate((x[i]-x[i].mean())/x[i].std(),(y[i]-y[i].mean())/y[i].std())

    return corr_.reshape(*shape_x[:-1])


def mean_diff_feature_maps(x, y):
    """
    Takes two activation matrices of the form BxFxT where B is batch size, F number of filters and T time points
    Returns mean difference between feature map activations

    Input: BxFxT (x,y)
    Returns: BxF
    """
    return np.mean(x - y, axis=2)


def perturbation_correlation(pert_fn, diff_fn, pred_fn, n_layers, inputs, n_iterations,
                             batch_size=30,
                             seed=((2017, 7, 10))):
    """
    Calculates phase perturbation correlation for layers in network

    pred_fn: Function that returns a list of activations.
             Each entry in the list corresponds to the output of 1 layer in a network
    n_layers: Number of layers pred_fn returns activations for.
    inputs: Original inputs that are used for perturbation [B,X,T,1]
            Phase perturbations are sampled for each input individually, but applied to all X of that input
    n_iterations: Number of iterations of correlation computation. The higher the better
    batch_size: Number of inputs that are used for one forward pass. (Concatenated for all inputs)
    """
    rng = np.random.RandomState(seed)

    # Get batch indeces
    batch_inds = get_balanced_batches(
        n_trials=len(inputs), rng=rng, shuffle=False, batch_size=batch_size)

    # Calculate layer activations and reshape
    orig_preds = [pred_fn(inputs[inds])
                  for inds in batch_inds]
    orig_preds_layers = [np.concatenate([orig_preds[o][l] for o in range(len(orig_preds))])
                         for l in range(n_layers)]

    # Compute FFT of inputs
    fft_input = np.fft.rfft(inputs, n=inputs.shape[2], axis=2)
    amps = np.abs(fft_input)
    phases = np.angle(fft_input)

    pert_corrs = [0] * n_layers
    for i in range(n_iterations):
        # print('Iteration%d'%i)

        amps_pert, phases_pert, pert_vals = pert_fn(amps, phases, rng=rng)

        # Compute perturbed inputs
        fft_pert = amps_pert * np.exp(1j * phases_pert)
        inputs_pert = np.fft.irfft(fft_pert, n=inputs.shape[2], axis=2).astype(np.float32)

        # Calculate layer activations for perturbed inputs
        new_preds = [pred_fn(inputs_pert[inds])
                     for inds in batch_inds]
        new_preds_layers = [np.concatenate([new_preds[o][l] for o in range(len(new_preds))])
                            for l in range(n_layers)]

        for l in range(n_layers):
            # Calculate correlations of original and perturbed feature map activations
            preds_diff = diff_fn(orig_preds_layers[l][:, :, :, 0], new_preds_layers[l][:, :, :, 0])

            # Calculate feature map correlations with absolute phase perturbations
            pert_corrs_tmp = wrap_reshape_apply_fn(corr,
                                                   pert_vals[:, :, :, 0], preds_diff,
                                                   axis_a=(0), axis_b=(0))
            pert_corrs[l] += pert_corrs_tmp

    pert_corrs = [pert_corrs[l] / n_iterations for l in range(n_layers)]  # mean over iterations
    return pert_corrs


def plot_history(history, to_plot):
    f, ax = plt.subplots(2, figsize=(15, 15))
    train_loss = history[:, 'train_loss']
    valid_loss = history[:, 'valid_loss']
    x = np.arange(0, len(train_loss))

    ax[0].plot(x, train_loss, label='train_loss')
    ax[0].plot(x, valid_loss, label='valid_loss')

    train_corr = history[:, 'train_correlation']
    valid_corr = history[:, 'validation_correlation']

    ax[1].plot(x, train_corr, label='train_correlation')
    ax[1].plot(x, valid_corr, label='valid_correlation')
    plt.show()


def plot_predicted_trajectory(preds_per_trial, targets_per_trial):
    plt.figure(figsize=(32, 12))
    t = np.arange(preds_per_trial.shape[0]) / srate
    plt.plot(t, preds_per_trial)
    plt.plot(t, targets_per_trial)
    plt.legend(('Predicted', 'Actual'), fontsize=14)
    plt.title('Fold = {:d}, CC = {:f}'.format(0, cc_folds[0]))
    plt.xlabel('time [s]')


if __name__ == '__main__':
    model_file = '/models/saved_models/best_model_1'
    model = load_model(model_file)

    data_file = 'ALL_11_FR1_day1_absVel'
    data = Data(home + f'/previous_work/{data_file}.mat', -1)
    n_preds_per_input = get_output_shape(model, data.in_channels, 1200)[1]
    data.cut_input(input_time_length, n_preds_per_input, False)
    train_set, test_set = data.train_set, data.test_set

    select_modules = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']  # Specify intermediate outputs
    modules = list(model.named_children())  # Extract modules from model
    model_pert = SelectiveSequential(select_modules, modules)  # Wrap modules
    model_pert.eval()
    model_pert.double()
    model.eval()

    pred_fn = lambda x: [layer_out.data.numpy() for
                         layer_out in model_pert.forward(torch.autograd.Variable(torch.from_numpy(x)).double())]
    perm_X = np.expand_dims(train_set.X, 3)

    wSize = 2 * n_preds_per_input
    new_train_set = np.asarray(train_set.X)
    new_train_set = reshape_Xs(wSize, new_train_set)

    print("visualization: perturbation computation ...")
    phase_pert_corrs = perturbation_correlation(phase_perturbation_chnls, correlate_feature_maps, pred_fn, 5,
                                                new_train_set, n_perturbations, batch_size=200)
    phase_pert_mdiff = perturbation_correlation(phase_perturbation_chnls, mean_diff_feature_maps, pred_fn, 5,
                                                new_train_set, n_perturbations, batch_size=200)
    amp_pert_mdiff = perturbation_correlation(amp_perturbation_additive, mean_diff_feature_maps, pred_fn, 5,
                                              new_train_set, n_perturbations, batch_size=200)

    freqs = np.fft.rfftfreq(new_train_set.shape[2], d=1 / 250.)
    history = History()
    history = history.from_file(home + '/logs/model_1_lr_0.001/histories/history_{last_epoch[epoch]}.json')
    plot_history(history, None)

    correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length, setname='idk')

    all_preds = []
    all_targets = []
    dataset = test_set
    for X, y in zip(train_set.X, train_set.y):
        preds = model(np_to_var(X).double())
        all_preds.append(preds)
        all_targets.append(y)

    preds_2d = [p[:, None] for p in all_preds]
    preds_per_trial = correlation_monitor.compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
    ys_2d = [y[:, None] for y in all_targets]
    targets_per_trial = correlation_monitor.compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
    assert preds_per_trial.shape == targets_per_trial.shape

    pred_vals = []
    resp_vals = []

    cc_folds = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
    pred_vals.append(preds_per_trial)
    resp_vals.append(targets_per_trial)

    plot_predicted_trajectory(preds_per_trial, targets_per_trial)




