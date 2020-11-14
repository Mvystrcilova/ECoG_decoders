#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:12:45 2020
visualization script (amplitude and phase perturbations) for car game paradigm
works only for 1 test set (last xval fold)
source data: v9_realTime (see exportData_4DNN_v9_realTime.m)
@author: jiri
"""

# %% settings: import libraries
import numpy as np
from matplotlib import pyplot as plt  # equiv. to: import matplotlib.pyplot as plt
import scipy.io as sio
import logging
import braindecode

log = logging.getLogger()
log.setLevel('DEBUG')
import sys

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    print("sys args = ", sys.argv)

    if len(sys.argv) == 1:
        n_job = '4'
    else:
        n_job = sys.argv[1]
        print("your input was: " + sys.argv[1])


    def file_for_number(x):
        return {

            '1': 'ALL_11_FR1_day1_xpos',
            '2': 'ALL_11_FR1_day1_xvel',
            '3': 'ALL_11_FR1_day1_absPos',
            '4': 'ALL_11_FR1_day1_absVel',

            '5': 'ALL_11_FR2_day2_xpos',
            '6': 'ALL_11_FR2_day2_xvel',
            '7': 'ALL_11_FR2_day2_absPos',
            '8': 'ALL_11_FR2_day2_absVel',

            '9': 'ALL_11_FR3_day2_xpos',
            '10': 'ALL_11_FR3_day2_xvel',
            '11': 'ALL_11_FR3_day2_absPos',
            '12': 'ALL_11_FR3_day2_absVel',

            '13': 'ALL_13_FR1_day2_xpos',
            '14': 'ALL_13_FR1_day2_xvel',
            '15': 'ALL_13_FR1_day2_absPos',
            '16': 'ALL_13_FR1_day2_absVel',

            '17': 'ALL_14_PR1_day1_xpos',
            '18': 'ALL_14_PR1_day1_xvel',
            '19': 'ALL_14_PR1_day1_absPos',
            '20': 'ALL_14_PR1_day1_absVel',

            '21': 'ALL_15_PR1_day1_xpos',
            '22': 'ALL_15_PR1_day1_xvel',
            '23': 'ALL_15_PR1_day1_absPos',
            '24': 'ALL_15_PR1_day1_absVel',

            '25': 'ALL_15_PR4_day2_xpos',
            '26': 'ALL_15_PR4_day2_xvel',
            '27': 'ALL_15_PR4_day2_absPos',
            '28': 'ALL_15_PR4_day2_absVel',

            '29': 'ALL_16_PR7_day1_xpos',
            '30': 'ALL_16_PR7_day1_xvel',
            '31': 'ALL_16_PR7_day1_absPos',
            '32': 'ALL_16_PR7_day1_absVel',

            '33': 'ALL_17_PR14_day1_xpos',
            '34': 'ALL_17_PR14_day1_xvel',
            '35': 'ALL_17_PR14_day1_absPos',
            '36': 'ALL_17_PR14_day1_absVel',

            '37': 'ALL_17_PR16_day1_xpos',
            '38': 'ALL_17_PR16_day1_xvel',
            '39': 'ALL_17_PR16_day1_absPos',
            '40': 'ALL_17_PR16_day1_absVel',

            '41': 'ALL_18_PR3_day1_xpos',
            '42': 'ALL_18_PR3_day1_xvel',
            '43': 'ALL_18_PR3_day1_absPos',
            '44': 'ALL_18_PR3_day1_absVel',

            '45': 'ALL_18_PR6_day1_xpos',
            '46': 'ALL_18_PR6_day1_xvel',
            '47': 'ALL_18_PR6_day1_absPos',
            '48': 'ALL_18_PR6_day1_absVel',

        }.get(x, 'ALL_11_FR1_day1_xpos')  # latter is default if x not found


    fileName = file_for_number(n_job)
    print("file = " + fileName)

    # %% local (CPU) or remote (GPU cluster) computing
    remoteComputing = False
    if remoteComputing:
        dir_sourceData = '.'
        dir_outputData = './outputs'
        import torch

        log.info("CUDA is avalaible? {:d}".format(torch.cuda.is_available()))
        cuda = True  # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
        maxTrainEpochs = 100
        N_perturbations = 500
    else:
        dir_sourceData = '.'
        dir_outputData = './outputs'
        cuda = False
        maxTrainEpochs = 10
        N_perturbations = 10

    from braindecode.torch_ext.util import np_to_var

    # %% Load data: matlab cell array
    import h5py

    log.info("Loading data...")
    with h5py.File(dir_sourceData + '/' + fileName + '.mat', 'r') as h5file:
        sessions = [h5file[obj_ref] for obj_ref in h5file['D'][0]]
        Xs = [session['ieeg'][:] for session in sessions]
        ys = [session['traj'][0] for session in sessions]
        srates = [session['srate'][0, 0] for session in sessions]

    # %% create datasets
    from braindecode.datautil.signal_target import SignalAndTarget

    # Outer added axis is the trial axis (size one always...)
    datasets = [SignalAndTarget([X.astype(np.float32)], [y.astype(np.float32)])
                for X, y in zip(Xs, ys)]

    from braindecode.datautil.splitters import concatenate_sets

    # only for allocation
    assert len(datasets) >= 4
    train_set = concatenate_sets(datasets[:-1])
    valid_set = datasets[-2]  # dummy variable, could be set to None
    test_set = datasets[-1]

    # %% create model
    from braindecode.models.deep4 import Deep4Net
    from torch import nn
    from braindecode.torch_ext.util import set_random_seeds
    from braindecode.models.util import to_dense_prediction_model
    from braindecode.torch_ext.modules import Expression

    set_random_seeds(seed=20170629, cuda=cuda)

    # This will determine how many crops are processed in parallel
    input_time_length = 1200
    n_classes = 1
    in_chans = train_set.X[0].shape[0]
    model = Deep4Net(in_chans=in_chans, n_classes=1,
                     input_time_length=input_time_length,
                     final_conv_length=2, stride_before_pool=True).create_network()

    # remove softmax
    new_model = nn.Sequential()
    for name, module in model.named_children():
        if name == 'softmax':
            break
        new_model.add_module(name, module)


    # lets remove empty final dimension
    def squeeze_out(x):
        assert x.size()[1] == 1 and x.size()[3] == 1
        return x[:, 0, :, 0]


    new_model.add_module('squeeze', Expression(squeeze_out))
    model = new_model

    to_dense_prediction_model(model)

    if cuda:
        model.cuda()
    from copy import deepcopy

    start_param_values = deepcopy(new_model.state_dict())

    # %% setup optimizer -> new for each x-val fold
    from torch import optim

    # %% # determine output size
    from braindecode.torch_ext.util import np_to_var

    test_input = np_to_var(
        np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
    if cuda:
        test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[1]
    log.info("predictor length = {:d} samples".format(n_preds_per_input))
    log.info("predictor length = {:f} s".format(n_preds_per_input / srates[0]))
    # crop size is: input_time_length - n_preds_per_input + 1
    # print("crop size = {:d} samples".format(input_time_length - n_preds_per_input + 1))

    # %% Iterator
    from braindecode.torch_ext.losses import log_categorical_crossentropy
    from braindecode.experiments.experiment import Experiment
    from braindecode.datautil.iterators import CropsFromTrialsIterator
    from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, \
        CroppedTrialMisclassMonitor, MisclassMonitor
    from braindecode.experiments.stopcriteria import MaxEpochs
    import torch.nn.functional as F
    import torch as th
    from braindecode.torch_ext.modules import Expression

    # Iterator is used to iterate over datasets both for data and evaluation
    iterator = CropsFromTrialsIterator(batch_size=32,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    # %% monitor for correlation
    from braindecode.experiments.monitors import compute_preds_per_trial_from_crops


    class CorrelationMonitor1d(object):
        """
        Compute correlation between 1d predictions

        Parameters
        ----------
        input_time_length: int
            Temporal length of one input to the model.
        """

        def __init__(self, input_time_length=None):
            self.input_time_length = input_time_length

        def monitor_epoch(self, ):
            return

        def monitor_set(self, setname, all_preds, all_losses,
                        all_batch_sizes, all_targets, dataset):
            """Assuming one hot encoding for now"""
            assert self.input_time_length is not None, "Need to know input time length..."
            # this will be timeseries of predictions
            # for each trial
            # braindecode functions expect classes x time predictions
            # so add fake class dimension and remove it again
            preds_2d = [p[:, None] for p in all_preds]
            preds_per_trial = compute_preds_per_trial_from_crops(preds_2d,
                                                                 self.input_time_length,
                                                                 dataset.X)
            preds_per_trial = [p[0] for p in preds_per_trial]
            pred_timeseries = np.concatenate(preds_per_trial, axis=0)
            ys_2d = [y[:, None] for y in all_targets]
            targets_per_trial = compute_preds_per_trial_from_crops(ys_2d,
                                                                   self.input_time_length,
                                                                   dataset.X)
            targets_per_trial = [t[0] for t in targets_per_trial]
            target_timeseries = np.concatenate(targets_per_trial, axis=0)

            corr = np.corrcoef(target_timeseries, pred_timeseries)[0, 1]
            key = setname + '_corr'

            return {key: float(corr)}


    # %% visualization (Kay): Phase and Amplitude perturbation
    import torch
    import numpy as np
    from braindecode.util import wrap_reshape_apply_fn, corr
    from braindecode.datautil.iterators import get_balanced_batches


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


    # %% Loss function takes predictions as they come out of the network and the targets and returns a loss
    loss_function = F.mse_loss

    # Could be used to apply some constraint on the models, then should be object with apply method that accepts a module
    model_constraint = None

    # %% Monitors log the data progress
    monitors = [LossMonitor(),
                CorrelationMonitor1d(input_time_length),
                RuntimeMonitor(), ]

    # %% Stop criterion determines when the first stop happens
    stop_criterion = MaxEpochs(maxTrainEpochs)

    # %% x-validation loop
    if remoteComputing:
        N = len(datasets)
        inds = np.arange(N)
    else:
        N = 6
        inds = np.arange(len(datasets))[-N:]

    cc_folds = np.zeros(N)
    pred_vals = []
    resp_vals = []

    # %% dataset indices
    n = 0
    i_test_set = inds[-1]
    i_valid_set = inds[-2]
    i_train_set = inds[:-1]  # merges valid set with train set, previously: i_train_set = inds_new[:-2]
    log.info("test set = %s" % i_test_set)
    log.info("valid set = %s" % i_valid_set)
    log.info("train set = %s" % i_train_set)

    # %% datasets
    # train_set = concatenate_sets(
    #     np.array(datasets)[i_train_set])
    # also:
    train_set = concatenate_sets([datasets[i] for i in i_train_set])
    valid_set = datasets[i_valid_set]
    test_set = datasets[i_test_set]
    log.info("Train set has {:d} folds".format(len(train_set.X)))

    # %% re-initialize model
    model.load_state_dict(deepcopy(start_param_values))
    optimizer = optim.Adam(model.parameters())

    # %% DNN setup & run
    # exp = Experiment(model, train_set, valid_set, test_set, iterator,
    #                 loss_function, optimizer, model_constraint,
    #                 monitors, stop_criterion,
    #                 remember_best_column='train_loss',
    #                 run_after_early_stop=False, batch_modifier=None, cuda=cuda, do_early_stop=False)
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='train_loss',
                     run_after_early_stop=False, batch_modifier=None, cuda=cuda)
    exp.run()

    # %% visualization (Kay): Wrap Model into SelectiveSequential and set up pred_fn
    assert (len(list(model.children())) == len(list(model.named_children())))  # All modules gotta have names!

    modules = list(model.named_children())  # Extract modules from model

    ## KAY added conv_classifier
    select_modules = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']  # Specify intermediate outputs

    model_pert = SelectiveSequential(select_modules, modules)  # Wrap modules
    # Prediction function that is used in phase_perturbation_correlation
    model_pert.eval()
    pred_fn = lambda x: [layer_out.data.numpy() for
                         layer_out in model_pert.forward(torch.autograd.Variable(torch.from_numpy(x)).float())]

    # Gotta change pred_fn a bit for cuda case
    if cuda:
        model_pert.cuda()
        pred_fn = lambda x: [layer_out.data.cpu().numpy() for
                             layer_out in
                             model_pert.forward(torch.autograd.Variable(torch.from_numpy(x)).float().cuda())]

    perm_X = np.expand_dims(train_set.X, 3)  # Input gotta have dimension BxCxTx1

    # %% reshape the array to 2*n_preds_per_input samples (2-s) long window
    wSize = 2 * n_preds_per_input  # smallest possible=685 (empirically found)
    train_set_new = np.asarray(train_set.X)
    nWindows = int(train_set_new.shape[2] / wSize)
    train_set_new = train_set_new[:, :, :wSize * int(train_set_new.shape[2] / wSize)]
    shape_tmp = train_set_new.shape

    ## EITHER wSize first or second after
    train_set_new = train_set_new.reshape(shape_tmp[0], shape_tmp[1], nWindows, wSize)
    train_set_new = train_set_new.transpose(0, 2, 1, 3)
    train_set_new = train_set_new.reshape(shape_tmp[0] * nWindows, shape_tmp[1], wSize, 1)

    # %% visualization (Kay): Run phase and amplitude perturbations
    log.info("visualization: perturbation computation ...")
    phase_pert_corrs = perturbation_correlation(phase_perturbation_chnls, correlate_feature_maps, pred_fn, 5,
                                                train_set_new, N_perturbations, batch_size=200)
    phase_pert_mdiff = perturbation_correlation(phase_perturbation_chnls, mean_diff_feature_maps, pred_fn, 5,
                                                train_set_new, N_perturbations, batch_size=200)
    amp_pert_mdiff = perturbation_correlation(amp_perturbation_additive, mean_diff_feature_maps, pred_fn, 5,
                                              train_set_new, N_perturbations, batch_size=200)

    # %% save perturbation over layers
    freqs = np.fft.rfftfreq(train_set_new.shape[2], d=1 / 250.)
    for l in range(len(phase_pert_corrs)):
        layer_cc = phase_pert_corrs[l]
        sio.savemat(
            dir_outputData + '/' + fileName + '_phiPrtCC' + '_layer{:d}'.format(l) + '_fold{:d}'.format(n) + '.mat',
            {'layer_cc': layer_cc})
        layer_cc = phase_pert_mdiff[l]
        sio.savemat(
            dir_outputData + '/' + fileName + '_phiPrtMD' + '_layer{:d}'.format(l) + '_fold{:d}'.format(n) + '.mat',
            {'layer_cc': layer_cc})
        layer_cc = amp_pert_mdiff[l]
        sio.savemat(
            dir_outputData + '/' + fileName + '_ampPrtMD' + '_layer{:d}'.format(l) + '_fold{:d}'.format(n) + '.mat',
            {'layer_cc': layer_cc})

    # %% plot learning curves
    if not remoteComputing:
        f, axarr = plt.subplots(2, figsize=(15, 15))
        exp.epochs_df.loc[:, ['train_loss', 'valid_loss', 'test_loss']].plot(ax=axarr[0], title='loss function')
        exp.epochs_df.loc[:, ['train_corr', 'valid_corr', 'test_corr']].plot(ax=axarr[1], title='correlation')
        plt.savefig(dir_outputData + '/' + fileName + '_fig_lc_fold{:d}.png'.format(n), bbox_inches='tight')

    # %% evaluation
    all_preds = []
    all_targets = []
    dataset = test_set
    for batch in exp.iterator.get_batches(dataset, shuffle=False):
        preds, loss = exp.eval_on_batch(batch[0], batch[1])
        all_preds.append(preds)
        all_targets.append(batch[1])

    preds_2d = [p[:, None] for p in all_preds]
    preds_per_trial = compute_preds_per_trial_from_crops(preds_2d, input_time_length, dataset.X)[0][0]
    ys_2d = [y[:, None] for y in all_targets]
    targets_per_trial = compute_preds_per_trial_from_crops(ys_2d, input_time_length, dataset.X)[0][0]
    assert preds_per_trial.shape == targets_per_trial.shape

    # %% save values: CC, pred, resp
    exp.epochs_df.to_csv(dir_outputData + '/' + fileName + '_epochs_fold{:d}.csv'.format(n), sep=',', header=False)
    cc_folds[n] = np.corrcoef(preds_per_trial, targets_per_trial)[0, 1]
    pred_vals.append(preds_per_trial)
    resp_vals.append(targets_per_trial)

    # %% plot predicted trajectory
    if not remoteComputing:
        plt.figure(figsize=(32, 12))
        t = np.arange(preds_per_trial.shape[0]) / srates[n]
        plt.plot(t, preds_per_trial)
        plt.plot(t, targets_per_trial)
        plt.legend(('Predicted', 'Actual'), fontsize=14)
        plt.title('Fold = {:d}, CC = {:f}'.format(n, cc_folds[n]))
        plt.xlabel('time [s]')
        plt.savefig(dir_outputData + '/' + fileName + '_fig_predResp_fold{:d}.png'.format(n), bbox_inches='tight')
    log.info("x-validation loop = " + str(n) + " done!")
    log.info("-----------------------------------------")

    # %% save CCs
    # np.save(dir_outputData + '/' + fileName + '_cc_folds', cc_folds)
    sio.savemat(dir_outputData + '/' + fileName + '_cc_folds' + '.mat', {'cc_folds': cc_folds})
    # np.save(dir_outputData + '/' + fileName + '_pred_vals', pred_vals)
    sio.savemat(dir_outputData + '/' + fileName + '_pred_vals' + '.mat', {'pred_vals': pred_vals})
    # np.save(dir_outputData + '/' + fileName + '_resp_vals', resp_vals)
    sio.savemat(dir_outputData + '/' + fileName + '_resp_vals' + '.mat', {'resp_vals': resp_vals})
    sio.savemat(dir_outputData + '/' + fileName + '_freqs' + '.mat', {'freqs': freqs})
    log.info("job: " + fileName + " done!")
