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

from Interpretation.interpretation import get_corr_coef
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from data.OnePredictionData import OnePredictionData
from data.pre_processing import Data, get_num_of_channels
from global_config import home, input_time_length, cuda, get_model_name_from_kernel_and_dilation
from models.Model import load_model, change_network_kernel_and_dilation, Model, add_padding



def test_input(input_channels, model):
    test_input = np_to_var(np.ones((2, input_channels, input_time_length, 1), dtype=np.float32))
    print(test_input.shape)
    out = model(test_input.double())
    n_preds_per_input = out.cpu().data.numpy().shape[1]
    return n_preds_per_input, test_input


def get_model(input_channels, input_time_length, dilations=None, kernel_sizes=None, padding=False):
    """
    initializes a new Deep4Net and changes the kernel sizes and dilations of the network based on the input parameters
    :param input_channels: 1 axis input shape
    :param input_time_length: 0 axis input shape
    :param dilations: dilations of the max-pool layers of the network
    :param kernel_sizes: kernel sizes of the max-pool layers of the network
    :param padding: if padding is to be added

    :return: a Model object, the changed Deep4Net based on the kernel sizes and dilation parameters and the name
    of the model based on the kernel sizes and dilatiosn
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]
    print('SBP False!!!')
    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=False)
    model.make_regressor()
    if cuda:
        model.model = model.model.cuda()

    model_name = get_model_name_from_kernel_and_dilation(kernel_sizes, dilations)

    changed_model = change_network_kernel_and_dilation(model.model, kernel_sizes, dilations, remove_maxpool=False)
    # print(changed_model)

    return model, changed_model, model_name


def train(data, dilation, kernel_size, lr, patient_index, model_string, correlation_monitor, output_dir,
          max_train_epochs=300, split=None, cropped=True, padding=False):
    """
    Creates and fits a model with the specified parameters onto the specified data
    :param data: dataset on which the model is to be trained
    :param dilation: dilation parameters of the model max-pool layers
    :param kernel_size: kernel sizes of the model's max-pool layers
    :param lr: learning rate
    :param patient_index: index of the patient on whose data the model is trained
    :param model_string: string specifying the setting of the data
    :param correlation_monitor: correlation monitor object calculating the correlations while fitting
    :param output_dir: where the trained model should be saved
    :param max_train_epochs: number of epochs for which to train the model
    :param split: the fold from cross-validation for which we are currently trainig the model
    :param cropped: if the decoding is cropped, alwasy True in thesis experiments
    :param padding: if padding should be added, always False in thesis experiments
    :return:
    """
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
    # cutting the input into batches compatible with model
    # if data.num_of_folds != -1, then also pre-whitening or filtering takes place
    # as part of the cut_input method
    data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)


    print(f'starting cv epoch {split} out of {data.num_of_folds} for model: {model_string}_{model_name}')
    correlation_monitor.step_number = 0
    if split is not None:
        correlation_monitor.split = split

    monitor = 'validation_correlation_best'

    monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                        f_history=home + f'/logs/model_{model_name}/histories/{model_string}_k_{model_name}_p_{patient_index}.json',
                                                                                        )),
                ]
    # cropped=False
    print('cropped:', cropped)

    # object EEGRegressor from the braindecode library suited for fitting models for regression tasks
    regressor = EEGRegressor(cropped=cropped, module=model.model, criterion=model.loss_function,
                             optimizer=model.optimizer,
                             max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,

                             callbacks=monitors, lr=lr, device=device, batch_size=32).initialize()

    torch.save(model.model,
               home + f'/models/saved_models/{output_dir}/initial_{model_string}_{model_name}_p_{patient_index}')
    regressor.max_correlation = -1000

    if padding:
        regressor.fit(data.train_set[0], data.train_set[1])

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
               whiten=False, indices=None, dummy_dataset=False):
    """
    Performs num_of_folds cross-validation on each of the patients
    :param model_string: specifies the setting in which the model was trained
    :param patient_indices: specifies the indices for patients for which a model should be trained
    :param dilation: dilation parameter of the max-pool layers in the network
    :param kernel_size: the kernel sizes of the max-pool layers in the network
    :param lr: learning rate
    :param num_of_folds: number of cross-validation folds. If -1, then only one 80-20 split is performed.
    :param trajectory_index: 0 for velocity, 1 for absolute velocity
    :param low_pass: specifies if validation data should be low-passed
    :param shift: specifies if predicted time-point should be shifted
    :param variable: 'vel' for velocity, 'absVel' for absolute velocity
    :param result_df: pandas.DataFrame where the results for the different patients are to be saved
    :param max_train_epochs: number of epochs for which to train the network
    :param high_pass: specifies if the train set and validation set should be high-passed
    :param high_pass_valid: specifies if the validation set should be high-passed
    :param padding: specifies if padding should be added to the network. Always False in this thesis.
    :param cropped: specifies if the input should be cropped. Always True in this thesis.
    :param low_pass_train: specifies if the training set should be low-passed
    :param shift_by: specifies by how much to shift the predicted time-point with across the receptive field
    :param saved_model_dir: specifies where the models should be saved
    :param whiten: specifies if the dataset should be whitened
    :param indices: specifies the indices for the different folds

    :return: None, only saves the learning statistics
    """

    best_valid_correlations = []
    # valid_indices = {}
    # train_indices = {}
    curr_patient_indices = None
    if dummy_dataset:
        patient_indices = [1]
    for patient_index in patient_indices:
        if indices is not None:
            curr_patient_indices = indices[f'P_{patient_index}']
        if dummy_dataset:
            data_file = f'{home}/data/dummy_dataset.mat'
        else:
            data_file = f'{home}/previous_work/P{patient_index}_data.mat'
        print('data_file', data_file)
        input_channels = get_num_of_channels(data_file, dummy_dataset=dummy_dataset)
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
                print('shift_index:', shift_index)
            print('dummy dataset', dummy_dataset)
            data = Data(data_file, num_of_folds=num_of_folds,
                        low_pass=low_pass,
                        trajectory_index=trajectory_index, shift_data=shift, high_pass=high_pass,
                        shift_by=int(shift_index),
                        valid_high_pass=high_pass_valid, low_pass_training=low_pass_train, pre_whiten=whiten,
                        indices=curr_patient_indices, dummy_dataset=dummy_dataset)
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
            # only one 80-20 train-valiation split
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
                # data.num_of_folds cross-validation
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
