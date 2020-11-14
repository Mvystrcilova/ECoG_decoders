from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor
from braindecode.experiments.stopcriteria import MaxEpochs

from data.pre_processing import Data
from global_config import home
import numpy as np
from models.Model import Model
from torch.utils.tensorboard import SummaryWriter

from training.CorrelationMonitor1D import CorrelationMonitor1d


def get_writer():
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(home + '/logs/playing_experiment_1')
    return writer


if __name__ == '__main__':
    input_time_length = 1200
    max_train_epochs = 100
    data = Data(home + 'previous_work/ALL_11_FR1_day1_absVel.mat', num_of_folds=6)

    model = Model(input_channels=data.train_set.X[0].shape[0], n_classes=1, input_time_length=input_time_length,
                  final_conv_length=2, stride_before_pool=True)
    model.make_regressor()

    writer = get_writer()
    writer.add_graph(model.model)

    monitors = [LossMonitor(),
                CorrelationMonitor1d(input_time_length=input_time_length),
                RuntimeMonitor()]

    stop_criterion = MaxEpochs(max_train_epochs)
    exp = Experiment(model, train_set=data.train_set, valid_set=data.valid_set, test_set=data.test_set,
                      loss_function=model.loss_function, optimizer=model.optimizer, loggers=[writer],
                     model_constraint=None, monitors=monitors, )

