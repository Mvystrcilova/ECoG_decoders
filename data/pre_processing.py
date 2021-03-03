import mat73
import numpy as np
from skorch.dataset import CVSplit, Dataset
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from scipy import signal
import matplotlib.pyplot as plt


def read_mat_file(mat_file):
    data = mat73.loadmat(mat_file)
    return data


def get_num_of_channels(mat_file):
    data = read_mat_file(mat_file)
    session = data.D[0]
    return session[0].ieeg.shape[1]


def band_pass_data(Xs, ys, order=3, cut_off_frequency=40, btype='low'):
    filter = signal.butter(order, cut_off_frequency, btype=btype, output='sos', fs=250)
    print(Xs[0][:, 0].shape)
    prev_signal = abs(np.fft.rfft(Xs[0][:, 0], n=250))
    bandpassed_x = [signal.sosfilt(filter, x, axis=0) for x in Xs]
    later_signal = abs(np.fft.rfft(bandpassed_x[0][:, 0], n=250))
    # plt.xscale('log')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('|amplitude|')
    # plt.plot(prev_signal)
    # plt.show()

    # plt.xscale('log')
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('|amplitude|')
    # plt.plot(later_signal)
    # plt.show()

    return bandpassed_x, ys
    # plt.plot(np.arange(0, len(Xs[0][:, 0][:1000])), Xs[0][:, 0][:1000], label='original')
    # plt.plot(np.arange(0, len(bandpassed_x[0][:, 0][:1000])), bandpassed_x[0][:, 0][:1000], label='low-passed')
    # plt.show()


class MyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class Data:
    def __init__(self, mat_file, num_of_folds, low_pass, trajectory_index, shift_data=False, high_pass=False,
                 valid_high_pass=False, shift_by=0, low_pass_training=False):
        self.data = read_mat_file(mat_file)
        self.motor_channels, self.non_motor_channels = None, None
        self.low_pass, self.high_pass, self.valid_high_pass, self.low_pass_training = low_pass, high_pass, valid_high_pass, low_pass_training
        self.shift_data = shift_data
        self.shift_by = shift_by
        self.band_passed_dataset = None
        self.num_of_folds = num_of_folds
        self.datasets = self.create_datasets(trajectory_index=trajectory_index)
        self.train_set, self.valid_set, self.test_set = self.split_data()
        if self.low_pass or self.low_pass_training:
            self.low_pass_train, self.low_pass_valid, self.low_pass_test = self.split_data(self.band_passed_dataset)
        if self.low_pass:
            self.test_set = self.low_pass_test

        if self.low_pass_training:
            self.train_set = self.low_pass_train

        if self.high_pass:
            self.train_set, self.valid_set, self.test_set = self.split_data(self.band_passed_dataset)
        if self.valid_high_pass:
            self.high_pass_train, self.high_pass_valid, self.high_pass_test = self.split_data(self.high_passed_dataset)
            self.test_set = self.high_pass_test
        if self.low_pass_training:
            self.train_set, self.low_pass_valid, self.low_pass_test = self.split_data(self.band_passed_dataset)
        self.in_channels = self.train_set.X[0].shape[1]
        self.n_classes = len(self.train_set.y[0].shape)
        self.fold_number = 0

        # if trajectory index is 0 velocity, index 1 absolute velocity
        self.traj_ind = trajectory_index

    def create_datasets(self, trajectory_index=0):
        sessions = self.data.D
        if self.shift_data:
            Xs = [session[0].ieeg[self.shift_by:] for session in sessions]
            ys = [session[0].traj[:-self.shift_by, trajectory_index] for session in sessions]
        else:
            Xs = [session[0].ieeg[:] for session in sessions]
            ys = [session[0].traj[:, trajectory_index] for session in sessions]

        print(len(Xs), len(ys))

        self.motor_channels = self.data.H.selCh_D_MTR - 1
        self.non_motor_channels = self.data.H.selCh_D_CTR - 1

        if self.num_of_folds != -1:
            self.num_of_folds = len(Xs)
        dataset = Dataset(Xs, ys)
        if self.low_pass:
            X, y = band_pass_data(Xs, ys, 3, 40, 'low')
            self.band_passed_dataset = Dataset(X, y)
        elif self.high_pass:
            X, y = band_pass_data(Xs, ys, 3, 60, 'hp')
            self.band_passed_dataset = Dataset(X, y)
        if self.valid_high_pass:
            X, y = band_pass_data(Xs, ys, 3, 60, 'hp')
            self.high_passed_dataset = Dataset(X, y)
        return dataset

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
            # index = int((length/100)*10)
            index = -1
            if index > -1:
                train_set = Dataset(X.X[:-index], y[:-index])
                valid_set = Dataset(X.X[-index:], y[-index:])
            else:
                train_set = Dataset(X.X[:], y[:])
                valid_set = self.test_set
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

    def get_certain_channels(self, dataset, motor=True):
        if motor:
            self.motor_channels = self.motor_channels.astype(int)
            new_set = np.copy(dataset.X)
            mask = np.ones(self.in_channels, np.bool)
            mask[self.motor_channels] = 0
            new_set[:, mask, :, :] = 0
            return Dataset(new_set, dataset.y)

        else:
            self.non_motor_channels = self.non_motor_channels.astype(int)
            new_set = np.copy(dataset.X)
            mask = np.ones(self.in_channels, np.bool)
            mask[self.non_motor_channels] = 0
            new_set[:, mask, :, :] = 0
            return Dataset(new_set, dataset.y)


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
    num_of_channels = get_num_of_channels('../previous_work/P1_data.mat')
    data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=0)
    # shifted_data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=0,
    #                     shift_data=True, high_pass=False)
    data.cut_input(1200, 519, False)
    prev_trainset = data.train_set
    motor_train_set = data.get_certain_channels(prev_trainset, True)
    non_motor_train_set = data.get_certain_channels(prev_trainset, False)
    print(motor_train_set.X.shape)
    print(non_motor_train_set.X.shape)

    # ax[2].plot(np.arange(0, 1200), shifted_data.datasets.X[0][:1200, 0], label= 'shifted Xs')
    # ax[2].plot(np.arange(0, 1200), shifted_data.datasets.y[0][:1200], label='shifted ys')
    # ax[2].set_title('Shifted')
    # plt.legend()
    #
    # data.cut_input(1200, 519, False)
    # shifted_data.cut_input(1200, 519, False)
    # ax[1].plot(np.arange(0, 1200), data.train_set.X[0, 0, :, 0], label='cut initial Xs')
    # ax[1].plot(np.arange(1200-519, 1200), data.train_set.y[0, :], label='cut initial ys')
    # plt.legend()
    #
    # ax[3].plot(np.arange(0, 1200), shifted_data.train_set.X[0, 0, :, 0], label='cut shifted Xs')
    # ax[3].plot(np.arange(1200 - 519, 1200), shifted_data.train_set.y[0, :], label='cut shifted ys')
    # plt.legend()
    # plt.show()
    print(data)



