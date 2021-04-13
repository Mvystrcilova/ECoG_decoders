from pathlib import Path
import  matplotlib.pyplot as plt
import pandas
import torch
import numpy as np
import pickle
from braindecode import EEGRegressor
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var
from skorch.callbacks import TensorBoard, Checkpoint
from torch.utils.tensorboard import SummaryWriter

from Interpretation.interpretation import get_corr_coef
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from data.OnePredictionData import OnePredictionData
from data.pre_processing import Data, get_num_of_channels
from global_config import home, input_time_length, cuda, get_model_name_from_kernel_and_dilation
from models.Model import load_model, change_network_stride, Model, add_padding


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


def get_model(input_channels, input_time_length, dilations=None, kernel_sizes=None, padding=False):
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]
    print('SBP False!!!')
    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=False)
    model.make_regressor()
    if cuda:
        model.model = model.model.cuda()

    conv_dilations = None

    model_name = ''.join([str(x) for x in kernel_sizes])
    if dilations is not None:
        dilations_name = ''.join(str(x) for x in dilations)
        model_name = f'{model_name}_dilations_{dilations_name}'
    model_name = get_model_name_from_kernel_and_dilation(kernel_sizes, dilations)
    # if conv_dilations is not None:
    #     conv_dilations_name = ''.join(str(x) for x in conv_dilations)
    #     model_name = f'{model_name}_conv_d_{conv_dilations_name}'

    changed_model = change_network_stride(model.model, kernel_sizes, dilations, remove_maxpool=False,
                                          change_conv_layers=conv_dilations is not None, conv_dilations=conv_dilations)
    #     changed_model = add_padding(changed_model, input_channels)
    print('Model not changing!')
    # changed_model = model.model
    return model, changed_model, model_name


def train(data, dilation, kernel_size, lr, patient_index, model_string, correlation_monitor, output_dir,
          max_train_epochs=300, split=None, cropped=True, padding=False):
    model, changed_model, model_name = get_model(data.in_channels, input_time_length, dilations=dilation,
                                                 kernel_sizes=kernel_size, padding=padding)
    if cuda:
        device = 'cuda'
        model.model = changed_model.cuda()

    else:
        model.model = changed_model
        device = 'cpu'
    if not padding:
        n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]
    else:
        n_preds_per_input = 1
    Path(
        home + f'/models/saved_models/{output_dir}/').mkdir(
        parents=True,
        exist_ok=True)

    data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)

    writer = get_writer(f'/logs/{output_dir}/cv_run_{1}')

    print(f'starting cv epoch {split} out of {data.num_of_folds} for model: {model_string}_{model_name}')
    correlation_monitor.step_number = 0
    if split is not None:
        correlation_monitor.split = split

    monitor = 'validation_correlation_best'

    monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                        f_history=home + f'/logs/model_{model_name}/histories/{model_string}_k_{model_name}_p_{patient_index}.json',
                                                                                        )),
                ('tensorboard', TensorBoard(writer, ))]
    # cropped=False
    print('cropped:', cropped)
    regressor = EEGRegressor(cropped=cropped, module=model.model, criterion=model.loss_function,
                             optimizer=model.optimizer,
                             max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,

                             callbacks=monitors, lr=lr, device=device, batch_size=32).initialize()

    torch.save(model.model,
               home + f'/models/saved_models/{output_dir}/initial_{model_string}_{model_name}_p_{patient_index}')
    regressor.max_correlation = -1000
    if padding:
        regressor.fit(data.train_set[0], data.train_set[1])
    # X = data.train_set.X[0]
    # ffted = np.fft.rfft(X[0, :, 0], n=X.shape[1])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('|amplitude|')
    # plt.plot(np.fft.rfftfreq(X.shape[1], 1/250.0), np.abs(ffted))
    # plt.show()
    regressor.fit(np.stack(data.train_set.X), np.stack(data.train_set.y))

    # best_model = load_model(
    #     f'/models/saved_models/{output_dir}/best_model_split_0')
    torch.save(model.model,
               home + f'/models/saved_models/{output_dir}/last_model_{split}')
    if cuda:
        best_corr = get_corr_coef(correlation_monitor.validation_set, model.model.cuda(device=device))
    else:
        best_corr = get_corr_coef(correlation_monitor.validation_set, model.model)
    print(patient_index, best_corr)
    return best_corr


def train_nets(model_string, patient_indices, dilation, kernel_size, lr, num_of_folds, trajectory_index, low_pass,
               shift, variable, result_df, max_train_epochs, high_pass=False, high_pass_valid=False,
               padding=False, cropped=True, low_pass_train=False, shift_by=None, saved_model_dir=f'lr_0.001',
               whiten=False, indices=None):
    best_valid_correlations = []
    # valid_indices = {}
    # train_indices = {}
    curr_patient_indices = None
    for patient_index in patient_indices:
        if indices is not None:
            curr_patient_indices = indices[f'P_{patient_index}']
        input_channels = get_num_of_channels(home + f'/previous_work/P{patient_index}_data.mat')
        model, changed_model, model_name = get_model(input_channels, input_time_length,
                                                     dilations=dilation,
                                                     kernel_sizes=kernel_size, padding=padding)
        small_window = 522
        if padding:
            data = OnePredictionData(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=num_of_folds,
                                     low_pass=low_pass, input_time_length=input_time_length,
                                     trajectory_index=trajectory_index, high_pass=high_pass,
                                     valid_high_pass=high_pass_valid)
        else:
            n_preds_per_input = get_output_shape(changed_model, input_channels, input_time_length)[1]
            small_window = input_time_length - n_preds_per_input + 1
            if shift_by is None:
                shift_index = int(small_window / 2)
            else:
                shift_index = int((small_window/2) - shift_by)
            data = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=num_of_folds,
                        low_pass=low_pass,
                        trajectory_index=trajectory_index, shift_data=shift, high_pass=high_pass,
                        shift_by=int(shift_index),
                        valid_high_pass=high_pass_valid, low_pass_training=low_pass_train, pre_whiten=whiten,
                        indices=curr_patient_indices)
        # valid_indices[f'P{patient_index}'] = data.valid_indices
        # train_indices[f'P{patient_index}'] = data.train_indices
        output_dir = f'{saved_model_dir}/{model_string}_{model_name}/{model_string}_{model_name}_p_{patient_index}'
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
                              max_train_epochs=max_train_epochs, output_dir=output_dir, split=None, cropped=cropped,
                              padding=padding)
            print('shift by:', shift_by)
            best_valid_correlations.append(best_corr)
            if len(best_valid_correlations) == 12:
                Path(f'{home}/outputs/{saved_model_dir}/{model_string}_{model_name}/{model_string}_{model_name}').mkdir(
                    parents=True,
                    exist_ok=True)
                result_df[f'{model_string}_{model_name}'] = best_valid_correlations
                best_valid_correlations = []
                result_df.to_csv(f'{home}/outputs/{saved_model_dir}/{model_string}_{model_name}/{model_string}_{model_name}/results.csv', sep=';')

        else:
            fold_corrs = []
            for i in range(data.num_of_folds):
                best_corr = train(data, dilation, kernel_size, lr, patient_index, model_string,
                                  correlation_monitor, output_dir, split=i,
                                  max_train_epochs=max_train_epochs, cropped=cropped)
                fold_corrs.append(best_corr)
            best_valid_correlations.append(fold_corrs)
            print('whole_patient:', patient_index, fold_corrs)
            patient_df = pandas.DataFrame()
            patient_df[f'P_{patient_index}'] = fold_corrs
            result_df = pandas.concat([result_df, patient_df], axis=1)
            result_df.to_csv(f'{home}/outputs/performances_{data.num_of_folds}/{model_string}_{model_name}/performances.csv', sep=';')
