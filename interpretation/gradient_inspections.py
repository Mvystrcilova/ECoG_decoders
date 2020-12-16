import matplotlib
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var, var_to_np
from interpretation import plot_gradients, reshape_Xs, calculate_phase_and_amps, get_corr_coef
from data.pre_processing import Data
from models.Model import load_model
from global_config import home, input_time_length
import logging, torch
log = logging.getLogger()
log.setLevel('DEBUG')
import sys
import numpy as np

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14
import seaborn
import torch
seaborn.set_style('darkgrid')


def load_data(data_file, num_of_folds=5):
    print("file = {:s}".format(data_file))
    data = Data(home + data_file, num_of_folds=num_of_folds)
    data.cut_input(input_time_length, n_preds_per_input[1], False)
    return data


if __name__ == '__main__':
    select_modules = ['conv_spat', 'conv_2', 'conv_3', 'conv_4', 'conv_classifier']
    model = load_model('/models/saved_models/best_model_1')
    n_preds_per_input = get_output_shape(model, 85, 1200)
    data = load_data('/previous_work/ALL_11_FR1_day1_absVel.mat', -1)

    for setname, dataset in (("Test", data.test_set), ("Train", data.train_set)):
        corrcoef = get_corr_coef(dataset, model)
        print(setname, 'correlation_coeff:', corrcoef)

        # Get corretly sized windows for gradients
        input_time_length = 1200

        wSize = 2 * n_preds_per_input[1]  # smallest possible=685 (empirically found)
        wSize = 439
        X_reshaped = np.asarray(dataset.X)
        X_reshaped = reshape_Xs(wSize, X_reshaped)
        for module_name in select_modules:
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
                test_out = new_model(np_to_var(X_reshaped[:2]))

            n_filters = test_out.shape[1]

            # filters x windows x channels x freqs
            all_amp_grads = np.ones(
                (n_filters,) + X_reshaped.shape[:2] + (len(np.fft.rfftfreq(X_reshaped.shape[2], d=1 / 250.0)),),
                dtype=np.float32) * np.nan
            all_phases_grads = np.ones(
                (n_filters,) + X_reshaped.shape[:2] + (len(np.fft.rfftfreq(X_reshaped.shape[2], d=1 / 250.0)),),
                dtype=np.float32) * np.nan

            i_start = 0
            for batch_X in np.array_split(X_reshaped, 5):
                iffted, amps_th, phases_th = calculate_phase_and_amps(batch_X)

                outs = new_model(iffted.double())
                assert outs.shape[1] == n_filters
                for i_filter in range(n_filters):
                    mean_out = torch.mean(outs[:, i_filter])
                    mean_out.backward(retain_graph=True)
                    amp_grads = var_to_np(amps_th.grad)
                    all_amp_grads[i_filter, i_start:i_start + len(amp_grads)] = amp_grads.squeeze(-1)
                    phases_grads = var_to_np(phases_th.grad)
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

            if module_name == 'conv_3':
                plot_gradients(X_reshaped, np.mean(meaned_amp_grads, axis=(0, 1)), corrcoef, '', setname=setname,
                               wsize=wSize)
                plot_gradients(X_reshaped, np.mean(np.abs(meaned_amp_grads), axis=(0, 1)), corrcoef=corrcoef,
                               title_prefix='Absolute ', setname=setname, wsize=wSize)
