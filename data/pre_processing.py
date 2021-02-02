import mat73
import numpy as np
from skorch.dataset import CVSplit, Dataset
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from scipy import signal
import matplotlib.pyplot as plt


def read_mat_file(mat_file):
    data = mat73.loadmat(mat_file)
    return data


class MyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class Data:
    def __init__(self, mat_file, num_of_folds, low_pass, trajectory_index):
        self.data = read_mat_file(mat_file)
        self.low_pass = low_pass
        self.band_passed_dataset = None
        self.num_of_folds = num_of_folds
        self.datasets = self.create_datasets(True, trajectory_index=trajectory_index)
        self.train_set, self.valid_set, self.test_set = self.split_data()
        self.low_pass_train, self.low_pass_valid, self.low_pass_test = self.split_data(self.band_passed_dataset)
        self.in_channels = self.train_set.X[0].shape[1]
        self.n_classes = len(self.train_set.y[0].shape)
        self.fold_number = 0

        # if trajectory index is 0 velocity, index 1 absolute velocity
        self.traj_ind = trajectory_index

    def create_datasets(self, low_pass=False, trajectory_index=0):
        sessions = self.data.D
        Xs = [session[0].ieeg[:] for session in sessions]
        ys = [session[0].traj[:, trajectory_index] for session in sessions]
        # ys = [session[0].traj for session in sessions]
        print(len(Xs), len(ys))
        self.num_of_folds = len(Xs)
        dataset = Dataset(Xs, ys)
        if low_pass:
            filter = signal.butter(3, 40, output='sos', fs=250)
            print(Xs[0][:, 0].shape)
            prev_signal = abs(np.fft.rfft(Xs[0][:, 0], n=250))
            bandpassed_x = [signal.sosfilt(filter, x, axis=0) for x in Xs]
            later_signal = abs(np.fft.rfft(bandpassed_x[0][:, 0], n=250))
            # plt.xscale('log')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('|amplitude|')
            plt.plot(prev_signal)
            # plt.show()

            # plt.xscale('log')
            plt.xlabel('frequency [Hz]')
            plt.ylabel('|amplitude|')
            plt.plot(later_signal)
            # plt.show()

            self.band_passed_dataset = Dataset(bandpassed_x, ys)
            plt.plot(np.arange(0, len(Xs[0][:, 0][:1000])), Xs[0][:, 0][:1000], label='original')
            plt.plot(np.arange(0, len(bandpassed_x[0][:, 0][:1000])), bandpassed_x[0][:, 0][:1000], label='low-passed')
            # plt.show()

        return dataset

    def split_data(self, dataset=None):
        length = len(self.datasets.X)
        index = int((length/100)*80)
        if dataset is None:
            train_set = MyDataset(self.datasets.X[:index], self.datasets.y[:index])
            test_set = MyDataset(self.datasets.X[index:], self.datasets.y[index:])
        else:
            train_set = MyDataset(dataset.X[:index], dataset.y[:index])
            test_set = MyDataset(dataset.X[index:], dataset.y[index:])
        return train_set, None, test_set

    def cut_input(self, input_time_length, n_preds_per_input, shuffle):
        iterator = CropsFromTrialsIterator(batch_size=32,
                                           input_time_length=input_time_length,
                                           n_preds_per_input=n_preds_per_input)
        self.train_set = concatenate_batches(self.train_set, iterator, False)
        self.test_set = concatenate_batches(self.test_set, iterator, False)
        self.low_pass_train = concatenate_batches(self.low_pass_train, iterator, False)
        self.low_pass_test = concatenate_batches(self.low_pass_test, iterator, False)

    def cv_split(self, X, y):
        length = len(X.X)
        if self.num_of_folds == -1:
            index = int((length/100)*10)
            index = -1
            if not self.low_pass:
                if index > -1:
                    train_set = Dataset(X.X[:-index], y[:-index])
                    valid_set = Dataset(X.X[-index:], y[-index:])
                else:
                    train_set = Dataset(X.X[:], y[:])
                    valid_set = self.test_set
                return train_set, valid_set
            else:
                if index > -1:
                    train_set = Dataset(X.X[:-index], y[:-index])
                    valid_set = Dataset(self.low_pass_train.X[-index:], self.low_pass_train.y[-index:])
                else:
                    train_set = Dataset(X.X[:], y[:])
                    valid_set = self.low_pass_test
                return train_set, valid_set

        fold_length = length / self.num_of_folds

        if self.fold_number == 0:
            train_set = Dataset(X.X[:int(fold_length * (self.num_of_folds - 1))],
                                y[:int(fold_length * (self.num_of_folds - 1))])
            validation_set = Dataset(X.X[int(fold_length * (self.num_of_folds - 1)):],
                                     y[int(fold_length * (self.num_of_folds - 1)):])

        elif self.fold_number == self.num_of_folds:
            train_set = Dataset(X.X[int(fold_length):], y[int(fold_length):])
            validation_set = Dataset(X.X[0:int(fold_length)], y[0:int(fold_length)])

        else:
            train_set = Dataset(np.concatenate([X.X[int(fold_length) * (self.fold_number - 1):int(fold_length) * self.fold_number],
                                X.X[int(fold_length) * (self.fold_number + 1):]]),
                                np.concatenate([y[int(fold_length) * (self.fold_number - 1):int(fold_length) * self.fold_number],
                                y[int(fold_length) * (self.fold_number + 1):]]
                                ))
            validation_set = Dataset(X.X[int(fold_length) * self.fold_number:int(fold_length) * (self.fold_number + 1)],
                                     y[int(fold_length) * self.fold_number:int(fold_length) * (self.fold_number + 1)])
        self.fold_number += 1
        return train_set, validation_set


def concatenate_batches(set, iterator, shuffle):
    complete_input = []
    complete_targets = []
    for batch in iterator.get_batches(set, shuffle=shuffle):
        for entry in batch[0]:
            complete_input.append(entry)
        for entry in batch[1]:
            complete_targets.append(entry)
    complete_input = np.array(complete_input)
    complete_targets = np.array(complete_targets)
    print(complete_input.shape)
    print(complete_targets.shape)
    return Dataset(complete_input, complete_targets)


if __name__ == '__main__':
    data = Data('../previous_work/ALL_11_FR1_day1_absVel.mat', -1, low_pass=False, trajectory_index=0)
