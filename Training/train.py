from pathlib import Path

import pandas
import torch
import numpy as np

from braindecode import EEGRegressor
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var
from skorch.callbacks import TensorBoard, Checkpoint
from torch.utils.tensorboard import SummaryWriter

from Interpretation.interpretation import get_corr_coef
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from data.pre_processing import Data
from global_config import home, input_time_length, cuda
from models.Model import load_model, change_network_stride, Model


def get_writer(path='/logs/playing_experiment_1'):
    writer = SummaryWriter(home + path)
    # writer.add_graph(model, example_input)
    return writer


def test_input(input_channels, model):
    test_input = np_to_var(np.ones((2, input_channels, input_time_length, 1), dtype=np.float32))
    print(test_input.shape)
    out = model(test_input.double())
    n_preds_per_input = out.cpu().data.numpy().shape[1]
    return n_preds_per_input, test_input


def get_model(input_channels, input_time_length, dilations=None, kernel_sizes=None):
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]

    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=True)
    model.make_regressor()
    if cuda:
        model.model = model.model.cuda()

    conv_dilations = None

    model_name = ''.join([str(x) for x in kernel_sizes])
    if dilations is not None:
        dilations_name = ''.join(str(x) for x in dilations)
        model_name = f'{model_name}_dilations_{dilations_name}'

    if conv_dilations is not None:
        conv_dilations_name = ''.join(str(x) for x in conv_dilations)
        model_name = f'{model_name}_conv_d_{conv_dilations_name}'

    changed_model = change_network_stride(model.model, kernel_sizes, dilations, remove_maxpool=False,
                                          change_conv_layers=conv_dilations is not None, conv_dilations=conv_dilations)

    return model, changed_model, model_name


def train(data, dilation, kernel_size, lr, patient_index, model_string, correlation_monitor, output_dir,
          max_train_epochs=300, split=None, cropped=True):
    model, changed_model, model_name = get_model(data.in_channels, input_time_length, dilations=dilation,
                                                 kernel_sizes=kernel_size)
    if cuda:
        device = 'cuda'
        model.model = changed_model.cuda()

    else:
        model.model = changed_model
        device = 'cpu'
    n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]
    Path(
        home + f'/models/saved_models/{output_dir}/').mkdir(
        parents=True,
        exist_ok=True)

    data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)

    writer = get_writer(f'/logs/{output_dir}/cv_run_{1}')

    print(f'starting cv epoch {-1} out of {data.num_of_folds} for model: {model_string}_k_{model_name}')
    correlation_monitor.step_number = 0
    if split is not None:
        correlation_monitor.split = split

    monitor = 'validation_correlation_best'

    monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                        f_history=home + f'/logs/model_{model_name}/histories/{model_string}_k_{model_name}_p_{patient_index}.json',
                                                                                        )),
                ('tensorboard', TensorBoard(writer, ))]

    regressor = EEGRegressor(cropped=cropped, module=model.model, criterion=model.loss_function,
                             optimizer=model.optimizer,
                             max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,
                             callbacks=monitors, lr=lr, device=device).initialize()

    torch.save(model.model,
               home + f'/models/saved_models/{output_dir}/initial_{model_string}_k_{model_name}_p_{patient_index}')
    regressor.max_correlation = -1000

    regressor.fit(data.train_set.X, data.train_set.y)
    best_model = load_model(
        f'/models/saved_models/{output_dir}/best_model_split_0')
    if cuda:
        best_corr = get_corr_coef(correlation_monitor.validation_set, best_model.cuda(device=device))
    else:
        best_corr = get_corr_coef(correlation_monitor.validation_set, best_model)
    print(patient_index, best_corr)
    return best_corr


def train_nets(model_string, patient_indices, dilation, kernel_size, lr, num_of_folds, trajectory_index, low_pass,
               shift, variable, result_df, max_train_epochs, high_pass=False, cropped=True):
    best_valid_correlations = []
    for patient_index in patient_indices:
        data = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=num_of_folds, low_pass=low_pass,
                    trajectory_index=trajectory_index, shift_data=shift, high_pass=high_pass)

        input_channels = data.in_channels
        model, changed_model, model_name = get_model(input_channels, input_time_length,
                                                     dilations=dilation,
                                                     kernel_sizes=kernel_size)

        output_dir = f'lr_{lr}/{model_string}_k_{model_name}/{model_string}_k_{model_name}_p_{patient_index}'
        correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length,
                                                   output_dir=output_dir)
        if cuda:
            device = 'cuda'
            model.model = changed_model.cuda()

        else:
            model.model = changed_model
            device = 'cpu'

        if data.num_of_folds == -1:
            best_corr = train(data, dilation, kernel_size, lr, patient_index, model_string, correlation_monitor,
                              max_train_epochs=max_train_epochs, output_dir=output_dir, split=None)
            best_valid_correlations.append(best_corr)
            if len(best_valid_correlations) == 12:
                result_df[f'{model_string}_k_{model_name}'] = best_valid_correlations
                best_valid_correlations = []
                result_df.to_csv(f'{home}/outputs/{variable}_avg_best_results.csv', sep=';')

        else:
            fold_corrs = []
            for i in range(data.num_of_folds-1):
                best_corr = train(data, dilation, kernel_size, lr, patient_index, model_string,
                                  correlation_monitor, output_dir, split=i,
                                  max_train_epochs=max_train_epochs)
                fold_corrs.append(best_corr)
            best_valid_correlations.append(fold_corrs)
            patient_df = pandas.DataFrame()
            patient_df[f'p_{patient_index}'] = best_valid_correlations
            df = pandas.concat([patient_df, result_df], ignore_index=True, axis=1)

            df.to_csv(
                f'{home}/outputs/performance/lr_{lr}/{model_string}_k_{model_name}/{variable}_performance.csv',
                sep=';')
