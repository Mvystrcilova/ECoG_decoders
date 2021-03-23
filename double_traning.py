import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from braindecode import EEGRegressor
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var
from skorch.callbacks import TensorBoard, Checkpoint

from Interpretation.interpretation import get_corr_coef
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from Training.train import get_model, get_writer
from data.pre_processing import get_num_of_channels, Data
from global_config import home, input_time_length, cuda, random_seed
from models.DoubleModel import DoubleModel
from models.Model import load_model

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
activations = {}

parser = argparse.ArgumentParser()
parser.add_argument('--variable', default=1, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    dilation = [None]
    kernel_size = [3, 3, 3, 3]
    trajectory_index = args.variable
    low_pass = False
    shift_by = None
    lr = 0.001
    shift = False
    max_train_epochs = 300
    if trajectory_index == 0:
        model_string = f'sbp0_dm_vel'
        variable = 'vel'
    else:
        model_string = 'sbp0_dm_absVel'
        variable = 'absVel'
    all_xs = []
    all_ys = []
    all_zs = []
    corr_coefs_full = []
    corr_coefs_hp = []
    for patient_index in range(1, 13):
        input_channels = get_num_of_channels(home + f'/previous_work/P{patient_index}_data.mat')
        # model, changed_model_full, model_name = get_model(input_channels, input_time_length,
        #                                                   dilations=dilation[0],
        #                                                   kernel_sizes=kernel_size, padding=False)
        # changed_model_full = load_model(f'/models/saved_models/lr_0.001/sbp1_sm_{variable}_k_3333/sbp1_sm_{variable}_k_3333_p_{patient_index}/best_model_split_0')
        changed_model_full = load_model(f'/models/saved_models/lr_0.001/sbp0_m_{variable}_k3_d3/sbp0_m_{variable}_k3_d3_p_{patient_index}/best_model_split_0')
        # _, changed_model_hp, _ = get_model(input_channels, input_time_length,
        #                                    dilations=dilation[0],
        #                                    kernel_sizes=kernel_size, padding=False)
        # changed_model_hp = load_model(f'/models/saved_models/lr_0.001/sbp1_hps_{variable}_k_3333/sbp1_hps_{variable}_k_3333_p_{patient_index}/best_model_split_0')
        changed_model_hp = load_model(f'/models/saved_models/lr_0.001/sbp0_hp_m_{variable}_k3_d3/sbp0_hp_m_{variable}_k3_d3_p_{patient_index}/best_model_split_0')

        model_name = 'k3_d3'
        n_preds_per_input = get_output_shape(changed_model_full, input_channels, input_time_length)[1]
        small_window = input_time_length - n_preds_per_input + 1
        if shift_by is None:
            shift_index = int(small_window / 2)
        else:
            shift_index = int((small_window / 2) - shift_by)

        data_full = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=-1,
                         low_pass=low_pass,
                         trajectory_index=trajectory_index, shift_data=shift, high_pass=False,
                         shift_by=int(shift_index),
                         valid_high_pass=False, low_pass_training=False, double_training=True)
        data_hp = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=-1,
                       low_pass=low_pass,
                       trajectory_index=trajectory_index, shift_data=shift, high_pass=True,
                       shift_by=int(shift_index),
                       valid_high_pass=False, low_pass_training=False)
        output_dir = f'{model_string}_{model_name}/{model_string}_{model_name}_p_{patient_index}'
        correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length,
                                                   output_dir=output_dir)

        if cuda:
            device = 'cuda'
            changed_model_full = changed_model_full.cuda()
            changed_model_hp = changed_model_hp.cuda()

        else:
            device = 'cpu'

        model = DoubleModel(changed_model_full, changed_model_hp)

        data_hp.cut_input(input_time_length, n_preds_per_input, False)
        data_full.cut_input(input_time_length, n_preds_per_input, False)
        # data_hp.train_set.X = np.zeros(data_hp.train_set.X.shape)
        writer = get_writer(f'/logs/{output_dir}/cv_run_{1}')
        correlation_monitor.step = 0
        monitor = 'validation_correlation_best'

        monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                            f_history=home + f'/logs/model_{model_name}/histories/{model_string}_k_{model_name}_p_{patient_index}.json',
                                                                                            )),
                    ('tensorboard', TensorBoard(writer, ))]

        regressor = EEGRegressor(cropped=True, module=model, criterion=model.loss_function,
                                 optimizer=model.optimizer,
                                 max_epochs=max_train_epochs, verbose=1,
                                 train_split=data_full.cv_split,
                                 callbacks=monitors, lr=lr, device=device, batch_size=32).initialize()
        args, kwargs = regressor.get_params_for_optimizer(
            'optimizer', regressor.module_.named_parameters())
        print('output dir:', output_dir)
        Path(home + f'/models/double_models/{output_dir}/').mkdir(exist_ok=True, parents=True)
        # torch.save(model,
        #            home + f'/models/double_models/{output_dir}/initial_{model_string}_{model_name}_p_{patient_index}')
        regressor.max_correlation = -1000
        index = 0
        while index < data_full.train_set.X.shape[0]:
            if index == 0:
                full_train_set = np.stack([data_full.train_set.X[index:index+32], data_hp.train_set.X[index:index+32]])
                full_train_set = np.moveaxis(full_train_set, 0, 3)
                full_train_set = full_train_set.reshape(
                    [full_train_set.shape[0], full_train_set.shape[1], full_train_set.shape[2], 2])
            else:
                new_stack = np.stack([data_full.train_set.X[index:index+32], data_hp.train_set.X[index:index + 32]])
                new_stack = np.moveaxis(new_stack, 0, 3)
                new_stack = new_stack.reshape([new_stack.shape[0], new_stack.shape[1], new_stack.shape[2], 2])
                full_train_set = np.concatenate([full_train_set, new_stack])


            index += 32

        full_train_set = np.stack([data_full.train_set.X, data_hp.train_set.X])
        full_train_set = full_train_set.reshape([full_train_set.shape[1], full_train_set.shape[2], full_train_set.shape[3], 2])
        # for epoch in range(max_train_epochs):
        #     for X_full, X_hp, y_full, y_hp in zip(data_full.train_set.X, data_hp.train_set.X, data_full.train_set.y, data_hp.train_set.y):
        #         regressor.train_step((X_full, X_hp), y_full)
        #     valid_corr =
        #     train_corr =
        # optimizer = torch.optim.Adam(lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
        # params = regressor.set_params({'optimizer': optimizer})
        regressor.fit(full_train_set, data_full.train_set.y)
        best_model = load_model(f'/models/double_models/{output_dir}/best_model_split_0')
        if cuda:
            best_corr = get_corr_coef(correlation_monitor.validation_set, best_model.cuda(device=device))
        else:
            best_corr = get_corr_coef(correlation_monitor.validation_set, best_model)
        print(patient_index, best_corr)
