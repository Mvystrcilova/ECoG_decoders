from braindecode.regressor import EEGRegressor
from data.pre_processing import Data
from global_config import home
import numpy as np
from models.Model import Model
import torch
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from skorch.callbacks import Checkpoint, TensorBoard
from sklearn.model_selection import cross_val_predict
from torch.utils.tensorboard.writer import SummaryWriter
from braindecode.util import np_to_var
from braindecode.models.util import get_output_shape
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from torchsummary import summary


def get_writer():
    writer = SummaryWriter(home + '/logs/setup_training')
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
    max_train_epochs = 1
    batch_size = 16

    data = Data(home + '/previous_work/ALL_11_FR1_day1_absVel.mat', num_of_folds=6)
    input_channels = data.in_channels

    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=True)
    model.make_regressor()

    n_preds_input, example_input = test_input(input_channels, model.model)
    writer = get_writer()
    n_preds_per_input = get_output_shape(model.model, model.input_channels, model.input_time_length)[1]
    # writer.add_graph(model.model, example_input)
    correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length, setname='')
    monitors = [('correlation monitor', correlation_monitor), ('checkpoint', Checkpoint()),
                ('tensorboard', TensorBoard(writer, ))]

    regressor = EEGRegressor(module=model.model, criterion=model.loss_function, optimizer=model.optimizer,
                             max_epochs=max_train_epochs, verbose=1, train_split=data.cv_split, callbacks=monitors).initialize()

    iterator = CropsFromTrialsIterator(batch_size=batch_size,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)
    correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length, setname='idk')
    optimizer = regressor.optimizer(model.model.parameters())
    summary(model.model, input_size=(input_channels, input_time_length, 1))
    data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)
    for i in range(data.num_of_folds):
        print(f'starting cv epoch {i} out of {data.num_of_folds}')
        regressor.fit(data.train_set.X, data.train_set.y)
        X = torch.Tensor(data.test_set.X)
        predictions = torch.Tensor(regressor.predict(X))
        # print(predictions)
        y = torch.Tensor(data.test_set.y)
        loss = torch.autograd.Variable(regressor.criterion()(predictions, y), True)
        print(loss)

    regressor.fit(data.train_set.X, data.train_set.y)

    torch.save(model.model.state_dict(), home +'/models/saved_models/playing_experiment_1')

