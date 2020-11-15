from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.datautil.iterators import get_balanced_batches, CropsFromTrialsIterator
from braindecode.torch_ext.util import np_to_var

from data.pre_processing import Data
from global_config import home
import numpy as np
from models.Model import Model
from torch.utils.tensorboard import SummaryWriter, FileWriter
from Training.CorrelationMonitor1D import CorrelationMonitor1D


def get_writer():
    writer = FileWriter(home + '/logs/playing_experiment_1')
    return writer


def test_input(input_channels, model):
    test_input = np_to_var(
        np.ones((2, input_channels, input_time_length, 1), dtype=np.float32))
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[1]
    return n_preds_per_input, test_input


if __name__ == '__main__':
    input_time_length = 1200
    max_train_epochs = 100
    batch_size = 32

    data = Data(home + '/previous_work/ALL_11_FR1_day1_absVel.mat', num_of_folds=6)
    input_channels = data.in_channels

    model = Model(input_channels=input_channels, n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=True)
    model.make_regressor()
    n_preds_input, example_input = test_input(input_channels, model.model)
    writer = get_writer()
    # writer.add_graph(model.model, example_input)

    monitors = [LossMonitor(),
                CorrelationMonitor1D(input_time_length=input_time_length),
                RuntimeMonitor()]

    iterator = CropsFromTrialsIterator(batch_size=batch_size, input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_input)

    stop_criterion = MaxEpochs(max_train_epochs)

    exp = Experiment(model.model, train_set=data.train_set, valid_set=data.valid_set, test_set=data.test_set,
                      loss_function=model.loss_function, optimizer=model.optimizer, loggers=[writer],
                     model_constraint=None, monitors=monitors, stop_criterion=stop_criterion,
                     remember_best_column='train_loss', run_after_early_stop=False, batch_modifier=None,
                     iterator=iterator, cuda=False)
    exp.run()

