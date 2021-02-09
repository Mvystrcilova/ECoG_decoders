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
    def __init__(self, mat_file, num_of_folds, low_pass, trajectory_index, shift_data=False, high_pass=False):
        self.data = read_mat_file(mat_file)
        self.low_pass, self.high_pass = low_pass, high_pass
        self.shift_data = shift_data
        self.band_passed_dataset = None
        self.num_of_folds = num_of_folds
        self.datasets = self.create_datasets(trajectory_index=trajectory_index)
        self.train_set, self.valid_set, self.test_set = self.split_data()
        if self.low_pass:
            self.low_pass_train, self.low_pass_valid, self.low_pass_test = self.split_data(self.band_passed_dataset)
        if self.high_pass:
            self.train_set, self.valid_set, self.test_set = self.split_data(self.band_passed_dataset)
        self.in_channels = self.train_set.X[0].shape[1]
        self.n_classes = len(self.train_set.y[0].shape)
        self.fold_number = 0

        # if trajectory index is 0 velocity, index 1 absolute velocity
        self.traj_ind = trajectory_index

    def create_datasets(self, trajectory_index=0):
        sessions = self.data.D
        if self.shift_data:
            Xs = [session[0].ieeg[600:] for session in sessions]
            ys = [session[0].traj[:-600, trajectory_index] for session in sessions]
        else:
            Xs = [session[0].ieeg[:] for session in sessions]
            ys = [session[0].traj[:, trajectory_index] for session in sessions]
        # ys = [session[0].traj for session in sessions]
        print(len(Xs), len(ys))
        if self.num_of_folds != -1:
            self.num_of_folds = len(Xs)
        dataset = Dataset(Xs, ys)
        if self.low_pass:
            self.band_pass_data(Xs, ys, 3, 40, 'low')
        elif self.high_pass:
            self.band_pass_data(Xs, ys, 3, 60, 'hp')
        return dataset

    def band_pass_data(self, Xs, ys, order=3, cut_off_frequency=40, btype='low'):
        filter = signal.butter(order, cut_off_frequency, btype=btype, output='sos', fs=250)
        print(Xs[0][:, 0].shape)
        prev_signal = abs(np.fft.rfft(Xs[0][:, 0], n=250))
        bandpassed_x = [signal.sosfilt(filter, x, axis=0) for x in Xs]
        later_signal = abs(np.fft.rfft(bandpassed_x[0][:, 0], n=250))
        # plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('|amplitude|')
        plt.plot(prev_signal)
        plt.show()

        # plt.xscale('log')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('|amplitude|')
        plt.plot(later_signal)
        plt.show()

        self.band_passed_dataset = Dataset(bandpassed_x, ys)
        plt.plot(np.arange(0, len(Xs[0][:, 0][:1000])), Xs[0][:, 0][:1000], label='original')
        plt.plot(np.arange(0, len(bandpassed_x[0][:, 0][:1000])), bandpassed_x[0][:, 0][:1000], label='low-passed')
        plt.show()

    def split_data(self, dataset=None):
        length = len(self.datasets.X)
        index = int((length/100)*80)
        if self.num_of_folds != -1:
            train_set = MyDataset(self.datasets.X[:], self.datasets.y[:])
            return train_set, None, None

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
        if self.num_of_folds == -1:
            self.test_set = concatenate_batches(self.test_set, iterator, False)
            if self.low_pass:
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

    def get_validation_set(self):
        length = len(self.train_set.X)
        fold_length = length / self.num_of_folds
        if self.fold_number == 0:
            validation_set = Dataset(self.train_set.X[int(fold_length * (self.num_of_folds - 1)):],
                                     self.train_set.y[int(fold_length * (self.num_of_folds - 1)):])

        elif self.fold_number == self.num_of_folds:
            validation_set = Dataset(self.train_set.X[0:int(fold_length)],
                                     self.train_set.y[0:int(fold_length)])

        else:
            validation_set = Dataset(self.train_set.X[int(fold_length) * self.fold_number:int(fold_length) * (self.fold_number + 1)],
                                     self.train_set.y[int(fold_length) * self.fold_number:int(fold_length) * (self.fold_number + 1)])

        return validation_set


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
    data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=0)
    shifted_data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=0,
                        shift_data=True, high_pass=True)
    print(data)
    data.cut_input(1200, 519, False)
    shifted_data.cut_input(1200, 519, False)
    print(data)



