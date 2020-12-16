from braindecode.regressor import EEGRegressor
from data.pre_processing import Data
from global_config import home, random_seed
import numpy as np
from models.Model import Model, change_network_stride
import torch
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from skorch.callbacks import Checkpoint, TensorBoard
from sklearn.model_selection import cross_val_predict
from torch.utils.tensorboard.writer import SummaryWriter
from braindecode.util import np_to_var
from braindecode.models.util import get_output_shape
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from torchsummary import summary
import random

random.seed(random_seed)


def get_writer(path='/logs/playing_experiment_1'):
    writer = SummaryWriter(home + path)
    # writer.add_graph(model, example_input)
    return writer


def test_input(input_channels, model):
    test_input = np_to_var(np.ones((2, input_channels, input_time_length, 1), dtype=np.float32))
    print(test_input.shape)
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[1]
    return n_preds_per_input, test_input


if __name__ == '__main__':
    input_time_length = 1200
    max_train_epochs = 200
    batch_size = 16

    data = Data(home + '/previous_work/ALL_11_FR1_day1_absVel.mat', num_of_folds=-1)
    input_channels = data.in_channels

    # writer.add_graph(model.model, example_input)
    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=True)
    model.make_regressor()
    summary(model.model, input_size=(input_channels, input_time_length, 1))
    changed_model = change_network_stride(model.model)
    model.model = changed_model
    summary(model.model, input_size=(input_channels, input_time_length, 1))
    n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length, output_dir='model_strides_4422')


    data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)
    if data.num_of_folds == -1:
        writer = get_writer(f'/logs/model_strides_4422/cv_run_{1}')
        print(f'starting cv epoch {-1} out of {data.num_of_folds}')
        correlation_monitor.step_number = 0

        monitor = 'validation_correlation_best'

        monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint(monitor=monitor,
                                                                                            f_params=home + "/logs/model_4422/params/params_{last_epoch[epoch]}.pt",
                                                                                            f_history=home + '/logs/model_4422/histories/history_{last_epoch[epoch]}.json',
                                                                                            f_optimizer=home + '/logs/model_4422/optimizers/optimizer_{last_epoch[epoch]}.pt',
                                                                                            )),
                    ('tensorboard', TensorBoard(writer, ))]
        regressor = EEGRegressor(module=model.model, criterion=model.loss_function, optimizer=model.optimizer,
                                 max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,
                                 callbacks=monitors, lr=0.001).initialize()
        torch.save(model.model, home + '/models/saved_models/initial_model_strides_4422')
        regressor.max_correlation = -1000

        regressor.fit(data.train_set.X, data.train_set.y)
        # regressor.fit(data.train_set.X, data.train_set.y)


    else:
        for i in range(2, data.num_of_folds):
            writer = get_writer(f'/logs/model_2_lr_0.001/cv_run_{i + 1}')
            print(f'starting cv epoch {i} out of {data.num_of_folds}')
            correlation_monitor.step_number = 0
            correlation_monitor.split = i

            monitors = [('correlation monitor', correlation_monitor),
                        ('checkpoint', Checkpoint(monitor='validation_correlation_best',)),
                        ('tensorboard', TensorBoard(writer, ))]
            n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]

            regressor = EEGRegressor(module=model.model, criterion=model.loss_function, optimizer=model.optimizer,
                                     max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split,
                                     callbacks=monitors, lr=0.001).initialize()
            torch.save(model.model, home + f'/models/saved_models/models_2/initial_model_split_{i}')
            regressor.max_correlation = -1000

            regressor.fit(data.train_set.X, data.train_set.y)
            X = torch.Tensor(data.test_set.X)
            predictions = torch.Tensor(regressor.predict(X))
            # print(predictions)
            y = torch.Tensor(data.test_set.y)
            loss = torch.autograd.Variable(regressor.criterion()(predictions, y), True)
            # test data correlation
            print('test data correlation:')
            test_correlation = correlation_monitor.on_epoch_end(regressor, dataset_train=data.test_set,
                                                                dataset_valid=data.test_set, test='test')
            print('test data loss')
            print(loss)

    # regressor.fit(data.train_set.X, data.train_set.y)
