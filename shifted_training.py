import argparse
from pathlib import Path
import os
import pandas
from braindecode.regressor import EEGRegressor

from Interpretation.interpretation import get_corr_coef
from data.pre_processing import Data
from global_config import home, random_seed, cuda
import numpy as np
from models.Model import Model, change_network_stride, load_model
import torch
from braindecode.util import set_random_seeds
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from skorch.callbacks import Checkpoint, TensorBoard
from sklearn.model_selection import cross_val_predict
from torch.utils.tensorboard.writer import SummaryWriter
from braindecode.util import np_to_var
from braindecode.models.util import get_output_shape
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from torchsummary import summary
import random

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
activations = {}

parser = argparse.ArgumentParser()
parser.add_argument("--kernel_size", default=[3, 3, 3, 3], type=int, nargs=4, help="Render some episodes.")
parser.add_argument("--dilations", default=[3, 9, 27, 81], type=int, nargs=4, help="Random seed.")
parser.add_argument("--starting_patient_index", default=1, type=int, help="Learning rate.")
parser.add_argument('--variable', default=0, type=int)


def get_writer(path='/logs/playing_experiment_1'):
    writer = SummaryWriter(home + path)
    # writer.add_graph(model, example_input)
    return writer


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


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


if __name__ == '__main__':
    args = parser.parse_args()
    input_time_length = 1200
    max_train_epochs = 500
    batch_size = 16
    print(cuda, home)
    set_random_seeds(seed=random_seed, cuda=cuda)
    cropped = True
    remove_maxpool = False
    trajectory_index = args.variable
    kernel_index = 2
    learning_rate = 0.001

    if trajectory_index == 0:
        model_string = f'sm_vel'
        variable = 'vel'
    else:
        model_string = 'sm_absVel'
        variable = 'absVel'
    if remove_maxpool:
        model_string = 'no_maxpool_model'


    model_name = ''

    best_valid_correlations = []

    dilations = [[1, 1, 1, 1], [2, 4, 8, 16]]
    # dilations = [None]
    # dilations = [[1, 1, 1, 1], [2, 4, 8, 16]]
    # kernel_sizes = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]

    for dilation in dilations:
        best_valid_correlations = []
        print(dilation)
        print(args.kernel_size)
        model_name = ''.join([str(x) for x in args.kernel_size])
        if dilation is not None:
            dilations_name = ''.join(str(x) for x in dilation)
            model_name = f'{model_name}_dilations_{dilations_name}'

        starting_patient_index = args.starting_patient_index
        if os.path.exists(f'{home}/outputs/shifted_performance/lr_{learning_rate}/{model_string}_k_{model_name}/{variable}_performance.csv'):
            df = pandas.read_csv(f'{home}/outputs/shifted_performance/lr_{learning_rate}/{model_string}_k_{model_name}/{variable}_performance.csv',
                                 sep=';', index_col=0)
            df = df.T.drop_duplicates().T
            starting_patient_index = df.shape[1] + 1
            print('exists')
        else:
            df = pandas.DataFrame()
        print(starting_patient_index)
        for patient_index in range(starting_patient_index, 13):
            data = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=-1, low_pass=False,
                        trajectory_index=trajectory_index, shift_data=True)

            input_channels = data.in_channels
            model, changed_model, model_name = get_model(input_channels, input_time_length,
                                                         dilations=dilation,
                                                         kernel_sizes=args.kernel_size)

            n_preds_per_input = get_output_shape(changed_model, model.input_channels, model.input_time_length)[1]
            data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)

            correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length,
                                                       output_dir=f'lr_{learning_rate}/{model_string}_k_{model_name}_p_{patient_index}')

            if data.num_of_folds == -1:
                model, changed_model, model_name = get_model(input_channels, input_time_length, dilations=dilation,
                                                             kernel_sizes=args.kernel_size)
                if cuda:
                    device = 'cuda'
                    model.model = changed_model.cuda()

                else:
                    model.model = changed_model
                    device = 'cpu'
                n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]
                Path(home + f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}_p_{patient_index}/').mkdir(parents=True,
                                                                                                            exist_ok=True)

                # data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)

                writer = get_writer(f'/logs/lr_{learning_rate}/{model_string}_k_{model_name}_p_{patient_index}/cv_run_{1}')
                # n_preds_per_input, test_input = test_input(input_channels, model.model)
                # writer.add_graph(model.model, test_input)
                print(f'starting cv epoch {-1} out of {data.num_of_folds} for model: {model_string}_k_{model_name}')
                correlation_monitor.step_number = 0

                monitor = 'validation_correlation_best'

                monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                                    f_history=home + f'/logs/model_{model_name}/histories/{model_string}_k_{model_name}_p_{patient_index}.json',
                                                                                                    )),
                            ('tensorboard', TensorBoard(writer, ))]

                regressor = EEGRegressor(cropped=cropped, module=model.model, criterion=model.loss_function,
                                         optimizer=model.optimizer,
                                         max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,
                                         callbacks=monitors, lr=learning_rate, device=device).initialize()

                torch.save(model.model,
                           home + f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}_p_{patient_index}/initial_{model_string}_k_{model_name}_p_{patient_index}')
                regressor.max_correlation = -1000

                # for name, module in regressor.module.named_children():
                #     module.register_forward_hook(get_activation(name))

                regressor.fit(data.train_set.X, data.train_set.y)
                best_model = load_model(
                    f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}_p_{patient_index}/best_model_split_0')
                best_corr = get_corr_coef(data.test_set, best_model.cuda(device=device))
                best_valid_correlations.append(best_corr)
                print(patient_index, best_corr)
                if len(best_valid_correlations) == 12:
                    f = open(f'{home}/outputs/{variable}_avg_best_results.txt', 'a')
                    f.write(f'{model_string}_k_{model_name} average best correlation:{sum(best_valid_correlations)/len(best_valid_correlations)}')
                    f.write('\n')
                    f.write(f'{model_string}_k_{model_name} best:{best_valid_correlations}')
                    f.write('\n')
                    f.close()


            # regressor.predict(data.train_set.X[:1])
            # for name, value in activation.items():
            #     print(name)
            #     if ('pool' in name) or ('conv' in name) or ('nonlin' in name):
            #         value = value.numpy()[0, 1, :, 0]
            #         values = [str(x) for x in value]
            #         print(' '.join(values[:100]))
            # print('done')

            else:
                fold_corrs = []
                for i in range(data.num_of_folds):
                    model, changed_model, model_name = get_model(input_channels, input_time_length,
                                                                 dilations=dilation,
                                                                 kernel_sizes=args.kernel_size)

                    writer = get_writer(
                        f'/logs/lr_{learning_rate}/{model_string}_k_{model_name}/{model_string}_folds_{data.num_of_folds}_k_{model_name}_p_{patient_index}/cv_run_{i + 1}')

                    Path(home + f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}/{model_string}_folds_{data.num_of_folds}_k_{model_name}_p_{patient_index}').mkdir(
                        parents=True,
                        exist_ok=True)

                    print(f'starting cv epoch {i} out of {data.num_of_folds}')
                    correlation_monitor.step_number = 0
                    correlation_monitor.split = i
                    correlation_monitor.output_dir = f'lr_{learning_rate}/{model_string}_k_{model_name}/{model_string}_folds_{data.num_of_folds}_k_{model_name}_p_{patient_index}'

                    if cuda:
                        device = 'cuda'
                        model.model = changed_model.cuda()

                    else:
                        model.model = changed_model
                        device = 'cpu'

                    monitors = [('correlation monitor', correlation_monitor),
                                ('checkpoint', Checkpoint(monitor='validation_correlation_best', )),
                                ('tensorboard', TensorBoard(writer, ))]

                    regressor = EEGRegressor(module=model.model, criterion=model.loss_function, optimizer=model.optimizer,
                                             max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,
                                             callbacks=monitors, lr=learning_rate, device=device).initialize()
                    torch.save(model.model,
                               home + f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}/{model_string}_folds_{data.num_of_folds}_k_{model_name}_p_{patient_index}/initial_{model_string}_split_{i}')
                    regressor.max_correlation = -1000

                    regressor.fit(data.train_set.X, data.train_set.y)

                    best_model = load_model(
                        f'/models/saved_models/lr_{learning_rate}/{model_string}_k_{model_name}/{model_string}_folds_{data.num_of_folds}_k_{model_name}_p_{patient_index}/best_model_split_{i}')
                    best_corr = get_corr_coef(correlation_monitor.validation_set, best_model.cuda(device=device))
                    fold_corrs.append(best_corr)
                    print(patient_index, best_corr)

                best_valid_correlations.append(fold_corrs)
                Path(
                    home + f'/outputs/shifted_performance/lr_{learning_rate}/{model_string}_k_{model_name}').mkdir(
                    parents=True,
                    exist_ok=True)
                patient_df = pandas.DataFrame()
                patient_df[f'p_{patient_index}'] = fold_corrs
                df = pandas.concat([patient_df, df], ignore_index=True, axis=1)

                df.to_csv(f'{home}/outputs/shifted_performance/lr_{learning_rate}/{model_string}_k_{model_name}/{variable}_performance.csv', sep=';')