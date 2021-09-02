import pickle
import random

import mat73
import numpy as np
import scipy
from random import gauss

from braindecode.util import np_to_var
from skorch.dataset import CVSplit, Dataset
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator
from scipy import signal
import matplotlib.pyplot as plt
import torch

from global_config import home


def read_mat_file(mat_file):
    data = mat73.loadmat(mat_file)
    return data


def get_num_of_channels(mat_file, dummy_dataset=False):
    if dummy_dataset:
        return 85
    data = read_mat_file(mat_file)
    session = data['D'][0]
    return session[0]['ieeg'].shape[1]


def band_pass_data(Xs, ys, order=3, cut_off_frequency=40, btype='low'):
    print('passing order:', order)
    filter = signal.butter(order, cut_off_frequency, btype=btype, output='sos', fs=250)
    print(Xs[0][:, 0].shape)
    prev_signal = abs(np.fft.rfft(Xs[0][:, 0], n=len(Xs[0])))
    bandpassed_x = [signal.sosfilt(filter, x, axis=0) for x in Xs]
    later_signal = abs(np.fft.rfft(bandpassed_x[0][:, 0], n=len(Xs[0])))

    # uncomment if interested what the filtering did with the signal
    """
    plt.xscale('log')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('|amplitude|')
    plt.plot(prev_signal)
    plt.show()

    plt.xscale('log')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('|amplitude|')
    plt.plot(later_signal)
    plt.show()

    plt.plot(np.arange(0, len(Xs[0][:, 0][:])), Xs[0][:, 0][:], label='original')
    plt.plot(np.arange(0, len(bandpassed_x[0][:, 0][:])), bandpassed_x[0][:, 0][:], label='low-passed')
    plt.show()"""
    return bandpassed_x, ys


def get_amplitude_strength(Xs):
    amplitudes = []
    for x in Xs:
        for i in range(x.shape[1]):
            prev_signal = abs(np.fft.rfft(x[:, i], n=250))
            amplitudes.append(prev_signal)
    print(len(prev_signal))
    print(prev_signal[0].shape)


class MyDataset:
    """ A class substituting the skorch.Dataset which can be used when using objects from the Braindecode library."""

    def __init__(self, X, y):
        self.X = X
        self.y = y


def whiten_data(dataset, valid_set=False, channel_normalizations=None, iqrs=None, means=None):
    """
    This function performs whitening of the signal in the
    :param dataset: the signals dataset which is to be whitened
    :param valid_set: indicates if the provided dataset is the training set or the validation set
    :param channel_normalizations: absolute values of the frequencies from the training set for normalizing the
    validation set
    :param iqrs: iqr values of the training set used to iqr normalize the validation set
    :param means: mean values of the training set used to normalize the validation set
    :return: returns the whitened data and in case valid_set is set to False also the training set values
    for channel_normalizations, iqrs and means to be used on the validation set
    """

    if valid_set:
        # For the validation set, the normalization parameters of the training set need to be specified
        assert channel_normalizations is not None
        assert means is not None
        assert iqrs is not None
    print('Whitening!')
    trial_ffteds = []
    trial_means = []
    trial_iqrs = []
    normalized_data = []
    for j, trial in enumerate(dataset.X):
        normalized_trial = []
        if not valid_set:
            channel_normalizations = []
            mean = []
            iqrs = []
        # for each channel
        for i in range(trial.shape[1]):
            channel = trial[:, i]
            ffted = np.fft.rfft(channel, axis=0, n=len(channel))
            # print(f'initial signal {i}', channel[:10])
            if not valid_set:
                channel_normalizations.append(np.abs(ffted))
                normalized_ffted = np.divide(ffted, np.abs(ffted))
                text = 'Train set'
            else:
                normalized_ffted = np.divide(ffted, channel_normalizations[i])
                text = 'Valid set'
            normalized_out = np.real(np.fft.ifft(normalized_ffted, n=len(channel)))
            # if interested to see the effects of whitening on the signal, uncomment the
            # the block below
            """
            second_ffted = np.fft.rfft(normalized_out, axis=0, n=len(channel))
            chopped_second_ffted = np.fft.rfft(normalized_out[:1200], axis=0, n=1200)
            second_normalized = np.fft.ifft(second_ffted, n=len(channel))
            plt.plot(np.fft.rfftfreq(len(channel), 1 / 250.0), np.abs(second_ffted))
            plt.show()
            plt.plot(np.fft.rfftfreq(1200, 1 / 250.0), np.abs(chopped_second_ffted))
            plt.show()
            print(f'normalized amplitudes signal {i}', normalized_out[:10])
            """

            if not valid_set:
                mean = np.mean(normalized_out)
                iqr = scipy.stats.iqr(normalized_out)
                iqrs.append(iqr)
                means.append(mean)
                normalized_out = (normalized_out - mean) / iqr
            else:
                normalized_out = (normalized_out - means[i]) / iqrs[i]
            # print(f'normalized amplitudes iqr signal {i}', normalized_out[:10])
            # if interested to see the effects of whitening on the signal, uncomment the
            # the block below
            """
            if (i == 0) and (j == 0):
                fig, ax = plt.subplots(2, 2, figsize=(13, 10))
                ax[(0, 0)].plot(np.fft.rfftfreq(len(channel), 1 / 250.0), np.abs(ffted))
                ax[(0, 0)].set_title('Original spectrum')
                ax[(0, 0)].set_ylabel('Amplitude')
                ax[(0, 0)].set_xlabel('Frequency (Hz)')
                ax[(0, 1)].plot([x for x in range(0, len(channel))][:2500], channel[:2500])
                ax[(0, 1)].set_title('Original signal')
                ax[(0, 1)].set_ylabel('Signal value')
                ax[(0, 1)].set_xlabel('Time (samples)')
                ax[(1, 0)].plot(np.fft.rfftfreq(len(channel), 1 / 250.0), np.abs(normalized_ffted))
                ax[(1, 0)].set_title('Normalized spectrum')
                ax[(1, 0)].set_ylabel('Amplitude')
                ax[(1, 0)].set_xlabel('Frequency (Hz)')
                ax[(1, 1)].plot([x for x in range(0, len(channel))][:2500], normalized_out[:2500])
                ax[(1, 1)].set_title('Normalized spectrum signal')
                ax[(1, 1)].set_ylabel('Signal value')
                ax[(1, 1)].set_xlabel('Time (samples)')
                ax[(0, 0)].annotate(text, xy=(1., 1), xytext=(12, 13),
                                        xycoords='axes fraction', textcoords='offset points',
                                        size='15', ha='center', va='baseline')
                plt.subplots_adjust(hspace=0.5)
                # plt.subplots_adjust(vspace=0.3)
                plt.savefig(f'{home}/outputs/pre_whitening_{i}{j}.png')
                plt.show()
                """
            normalized_trial.append(normalized_out)
        normalized_trial = np.stack(normalized_trial, axis=1)
        trial_ffteds.append(channel_normalizations)
        trial_means.append(mean)
        trial_iqrs.append(iqrs)
        normalized_data.append(normalized_trial)
    channel_normalizations = np.mean(np.stack(trial_ffteds), axis=0)
    mean = np.mean(np.stack(trial_means), axis=0)
    iqrs = np.mean(np.stack(trial_iqrs), axis=0)
    assert len(dataset.X) == len(normalized_data)
    assert len(dataset.X[0]) == len(normalized_data[0])
    return normalized_data, channel_normalizations, iqrs, mean


class Data:
    """
    This class handles the dataset, its splitting, shifting, filtering and whitening

    """

    def __init__(self, mat_file, num_of_folds, low_pass, trajectory_index, indices=None, shift_data=False,
                 high_pass=False,
                 valid_high_pass=False, shift_by=0, low_pass_training=False, double_training=False, train_indices=None,
                 valid_indices=None, pre_whiten=False, random_valid=True, absVel_from_vel=False, dummy_dataset=False):
        """

        :param mat_file: a Matlab which holds the data
        :param num_of_folds: number of folds into which the data is to be split
        :param low_pass: if True, the validation set is low-passed
        :param trajectory_index: 0 to get velocity data, 1 to get absolute velocity data
        :param indices: indices of the folds
        :param shift_data: if True, the predicted time-point is shifted with respect to the signals based
        on the value specified in shift_by.
        :param high_pass: If True, the training set and the validation set are high-passed
        :param valid_high_pass: If True, only the validation set is high-passed
        :param shift_by: specifies by how much the predicted time-point should be shifted with respect to the
         receptive field of the network. If none and shift_data is True, the signal is shifted to the centre
         of the receptive field
        :param low_pass_training: If set to True, the training set is low-passed
        :param double_training: Not finished, meant to combine data from the high-passed and full dataset
        :param train_indices: indices to be used for the train set
        :param valid_indices: indices to be used for the validation set
        :param pre_whiten: if True, the datasets are whitened
        :param random_valid:
        :param absVel_from_vel: says if we are predicting absolute velocity values as taking absolute values of
        velocity
        """
        self.valid_hp_predictions = None
        self.hp_predictions = None
        self.random_valid = random_valid
        self.absVel_from_vel = absVel_from_vel
        self.pre_whiten = pre_whiten
        self.dummy_dataset = dummy_dataset
        if not self.dummy_dataset:
            self.data = read_mat_file(mat_file)
        else:
            self.data = None
        self.indices = indices
        self.double_training = double_training
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.motor_channels, self.non_motor_channels = None, None
        self.low_pass, self.high_pass, self.valid_high_pass, self.low_pass_training = low_pass, high_pass, valid_high_pass, low_pass_training
        self.high_pass_train, self.high_pass_valid, self.high_pass_test = None, None, None
        self.low_pass_train, self.low_pass_test = None, None
        self.shift_data = shift_data
        self.shift_by = shift_by
        self.num_of_folds = num_of_folds
        if num_of_folds != -1:
            assert indices is not None

        # checking if the trajectory index has allowed values
        assert (trajectory_index == 0) or (trajectory_index == 1)

        self.datasets = self.create_datasets(trajectory_index=trajectory_index)
        self.train_set, self.valid_set, self.test_set = self.split_data()
        if num_of_folds == -1:
            if self.low_pass:
                self.test_set = self.low_pass_test

            if self.low_pass_training:
                self.train_set = self.low_pass_train

            if self.high_pass:
                self.train_set, self.valid_set, self.test_set = self.high_pass_train, self.high_pass_valid, self.high_pass_test

            if self.valid_high_pass:
                self.test_set = self.high_pass_test

        self.in_channels = self.train_set.X[0].shape[1]
        self.n_classes = len(self.train_set.y[0].shape)
        self.fold_number = 0

        # if trajectory index is 0 velocity, index 1 absolute velocity
        self.traj_ind = trajectory_index

    def create_datasets(self, trajectory_index=0):
        """
        Creates an Dataset object from data which has been read from the Matlab file
        Also saves information about which electrodes belong to motor channels and which
        to non-motor channels
        :param trajectory_index: Selects velocity with 0 and absolute velocity with 1
        :return:
        """
        if not self.dummy_dataset:
            sessions = self.data['D']
            if self.shift_data:
                Xs = [session[0]['ieeg'][self.shift_by:] for session in sessions]
                ys = [session[0]['traj'][:-self.shift_by, trajectory_index] for session in sessions]
            else:
                Xs = [session[0]['ieeg'][:] for session in sessions]
                ys = [session[0]['traj'][:, trajectory_index] for session in sessions]
            # if self.absVel_from_vel:
            #     ys = np.abs(ys)
            print(len(Xs), len(ys))

            self.motor_channels = self.data['H']['selCh_D_MTR'] - 1
            self.non_motor_channels = self.data['H']['selCh_D_CTR'] - 1

            if self.num_of_folds == 0:
                self.num_of_folds = len(Xs)
        else:
            print('creating dummy dataset')
            Xs = []
            ys = []
            for i in range(32):
                ieeg = np.zeros([7500, 85])
                for row in range(85):
                    series = [gauss(0.0, 1.0) for i in range(7500)]
                    ieeg[:, row] = series
                traj = np.zeros([7500])
                series = [gauss(0.0, 1.0) for i in range(7500)]
                traj[:] = series
                Xs.append(ieeg)
                ys.append(traj)
            self.motor_channels = None
            self.non_motor_channels = None

        dataset = Dataset(Xs, ys)

        return dataset

    def split_data(self, dataset=None):
        """
        Splits dataset into the training and validation set while also performing filtering, whitening
        :param dataset: if dataset is None, the self.datasets is split, else the given dataset is split into
        training and validation indices. In this case, only whitening is done, not filtering
        :return: returns the training and validation set
        """
        length = len(self.datasets.X)
        if self.num_of_folds == -1:
            index = int((length / 100) * 80)

            if (self.train_indices is None) and self.random_valid:
                valid_indices = random.sample([x for x in range(length)], length - index)
                self.valid_indices = valid_indices
                train_indices = [x for x in range(length) if x not in valid_indices]
                self.train_indices = train_indices

            elif not self.random_valid:
                self.train_indices = [x for x in range(0, index)]
                self.valid_indices = [x for x in range(index, length)]

            if dataset is None:
                train_set = MyDataset([self.datasets.X[i] for i in self.train_indices],
                                      [self.datasets.y[i] for i in self.train_indices])
                test_set = MyDataset([self.datasets.X[i] for i in self.valid_indices],
                                     [self.datasets.y[i] for i in self.valid_indices])
                if self.random_valid:
                    print('Random sets')
            else:
                train_set = MyDataset([dataset.X[i] for i in self.train_indices],
                                      [dataset.y[i] for i in self.train_indices])
                test_set = MyDataset([dataset.X[i] for i in self.valid_indices],
                                     [dataset.y[i] for i in self.valid_indices])
            print('validation_indices:', self.valid_indices)
            print('train_indices:', self.train_indices)

            if self.pre_whiten and (dataset is None):
                train_set.X, channel_norms, iqr, median = whiten_data(train_set)
                test_set.X, _, _, _ = whiten_data(test_set, True, channel_normalizations=channel_norms, iqrs=iqr)
            if dataset is None:
                train_Xs, train_ys = train_set.X, train_set.y
                test_Xs, test_ys = test_set.X, test_set.y
                if self.low_pass or self.low_pass_training:
                    X, y = band_pass_data(train_Xs, train_ys, 15, 40, 'low')
                    self.low_pass_train = Dataset(X, y)
                    X, y = band_pass_data(test_Xs, test_ys, 15, 40, 'low')
                    self.low_pass_test = Dataset(X, y)

                elif self.high_pass or self.valid_high_pass:
                    X, y = band_pass_data(train_Xs, train_ys, 15, 60, 'hp')
                    self.high_pass_train = Dataset(X, y)
                    X, y = band_pass_data(test_Xs, test_ys, 15, 60, 'hp')
                    self.high_pass_test = Dataset(X, y)

            return train_set, None, test_set

        else:
            train_set = MyDataset(self.datasets.X[:], self.datasets.y[:])
            return train_set, None, None

    def cut_input(self, input_time_length, n_preds_per_input, shuffle):
        """self.train_set and self.test_set into crops to be compatible with the
        EEGRegressor in the Braindecode library"""
        iterator = CropsFromTrialsIterator(batch_size=32,
                                           input_time_length=input_time_length,
                                           n_preds_per_input=n_preds_per_input)
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input
        if self.num_of_folds == -1:
            self.test_set = concatenate_batches(self.test_set, iterator, False)
            self.train_set = concatenate_batches(self.train_set, iterator, False)
        else:
            pass
            # random_indices = [x for x in range(len(self.test_set))]
            # random.shuffle(random_indices)
            # ys = self.test_set.y
            # self.test_set.y = np.stack([ys[index] for index in random_indices])
            # print('shuffled valid set!')
            # if self.low_pass:
            #     self.low_pass_train = concatenate_batches(self.low_pass_train, iterator, False)
            #     self.low_pass_test = concatenate_batches(self.low_pass_test, iterator, False)

    def cv_split(self, X, y):
        """
        In case of only one fold validation, splits the data into the training and validation set.
        In case of n_fold cross-validation also splits the data into the training and validation set.
        However, in the case of n-fold cross-validation because we are combining filtered and non-filtered
        datasets as well as changing the training set when pre-whitening, the filtering and whitening has
        to be performed with each fold which is implemented in this method
        :param X: input signals
        :param y: gold labels
        :return: returns the training and validation set for this cross-validation fold
        """
        assert self.n_preds_per_input is not None
        "Needs to run cut_input first to assign n_preds per input and input_time_length"
        if isinstance(X, np.ndarray):
            length = len(X)
        else:
            X = X.X
            length = len(X)
        if self.num_of_folds == -1:
            # index = int((length/100)*10)
            index = -1
            if index > -1:
                train_set = Dataset(X[:-index], y[:-index])
                valid_set = Dataset(X[-index:], y[-index:])
            else:
                train_set = Dataset(X[:], y[:])
                valid_set = self.test_set
            if self.double_training:
                second_X, _ = band_pass_data(valid_set.X, valid_set.y, order=3, cut_off_frequency=60, btype='hp')
                second_test_set = np.stack(second_X)
                # second_test_set = np.zeros(second_test_set.shape)
                i = 0
                while i < valid_set.X.shape[0]:
                    if i == 0:
                        full_train_set = np.stack(
                            [valid_set.X[i:i + 32], second_test_set[i:i + 32]])
                        full_train_set = np.moveaxis(full_train_set, 0, 3)
                        full_train_set = full_train_set.reshape(
                            [full_train_set.shape[0], full_train_set.shape[1], full_train_set.shape[2], 2])
                    else:
                        new_stack = np.stack(
                            [valid_set.X[i:i + 32], second_test_set[i:i + 32]])
                        new_stack = np.moveaxis(new_stack, 0, 3)
                        new_stack = new_stack.reshape([new_stack.shape[0], new_stack.shape[1], new_stack.shape[2], 2])
                        full_train_set = np.concatenate([full_train_set, new_stack])
                    i += 32
                valid_set = Dataset(full_train_set, self.test_set.y)
            return train_set, valid_set

        # this code executes self.num_of_folds is larger than -1
        self.valid_indices = self.indices[self.fold_number]
        self.train_indices = []
        for i in range(self.num_of_folds):
            if i != self.fold_number:
                self.train_indices += list(self.indices[i])
        print('train indices:', self.train_indices)
        print('valid indices:', self.valid_indices)

        train_set = MyDataset([self.train_set.X[i] for i in self.train_indices],
                              [self.train_set.y[i] for i in self.train_indices])

        validation_set = MyDataset([self.train_set.X[i] for i in self.valid_indices],
                                   [self.train_set.y[i] for i in self.valid_indices])
        # print('Attention! shuffled set')
        # random_indices = get_random_permutation_with_no_fixed_point(self.valid_indices)
        # validation_set = MyDataset([self.train_set.X[i] for i in self.valid_indices],
        #                            [self.train_set.y[self.valid_indices[index]] for index in random_indices])

        if self.pre_whiten:
            train_set.X, channel_norms, iqr, median = whiten_data(train_set)
            validation_set.X, _, _, _ = whiten_data(validation_set, True, channel_normalizations=channel_norms,
                                                    iqrs=iqr)

        train_Xs, train_ys = train_set.X, train_set.y
        validation_Xs, validation_ys = validation_set.X, validation_set.y

        # filters the training set and validation set for the current cv fold
        if self.low_pass or self.low_pass_training:
            X, y = band_pass_data(train_Xs, train_ys, 15, 40, 'low')
            self.low_pass_train = Dataset(X, y)
            X, y = band_pass_data(validation_Xs, validation_ys, 15, 40, 'low')
            self.low_pass_test = Dataset(X, y)
        if self.low_pass:
            validation_set = self.low_pass_test
        if self.low_pass_training:
            train_set = self.low_pass_train

        if self.high_pass or self.valid_high_pass:
            X, y = band_pass_data(train_Xs, train_ys, 15, 60, 'hp')
            self.high_pass_train = Dataset(X, y)
            X, y = band_pass_data(validation_Xs, validation_ys, 15, 60, 'hp')
            self.high_pass_test = Dataset(X, y)
        if self.high_pass:
            train_set = self.high_pass_train
            validation_set = self.high_pass_test
        if self.valid_high_pass:
            validation_set = self.high_pass_test

        # cuts the input similarly to Data.cut_input()
        iterator = CropsFromTrialsIterator(batch_size=32,
                                           input_time_length=self.input_time_length,
                                           n_preds_per_input=self.n_preds_per_input)

        validation_set = concatenate_batches(validation_set, iterator, False)
        train_set = concatenate_batches(train_set, iterator, False)
        if self.hp_predictions is not None:
            train_set.y = self.hp_predictions
            validation_set.y = self.valid_hp_predictions
        self.fold_number += 1
        return train_set, validation_set

    def get_certain_channels(self, dataset, motor=True):
        """
        Returns the indices of either motor or non-motor channels
        :param dataset: one patient data for which we want the channels
        :param motor: if motor True, motor channel indices are returned, else
        non-motor channel indices are returned
        :return: Dataset object of only data belonging to either motor or non-motor channels
        """
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


def get_random_permutation_with_no_fixed_point(indices):
    while True:
        v = np.arange(len(indices))
        for j in np.arange(len(indices) - 1, -1, -1):
            p = np.random.randint(0, j + 1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                print('shuffled indices:', v)
                return v


if __name__ == '__main__':
    """
    Toy examples to check correctness and behaviour of the parameters specified for the 
    Data class
    """
    num_of_channels = get_num_of_channels('../previous_work/P1_data.mat')
    with open(f'{home}/data/train_dict_5', 'rb') as file:
        indices = pickle.load(file)

    # no_shift_data = Data('../previous_work/P1_data.mat', 5, low_pass=False, trajectory_index=0,
    #                      low_pass_training=False,
    #                      valid_high_pass=False, shift_by=int(628 / 2), shift_data=False,
    #                      pre_whiten=True, high_pass=True, indices=indices['P_1'])
    #
    data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=0, low_pass_training=False,
                valid_high_pass=False, shift_data=False, dummy_dataset=False)

    # shifted_data = Data('../previous_work/P1_data.mat', -1, low_pass=False, trajectory_index=1,
    #                     shift_data=False, high_pass=False)
    data.cut_input(1200, 519, False)
    # shifted_data.cut_input(1200, 519, False)
    # prev_trainset = data.train_set
    # motor_train_set = data.get_certain_channels(prev_trainset, True)
    # non_motor_train_set = data.get_certain_channels(prev_trainset, False)
    # print(motor_train_set.X.shape)
    # print(non_motor_train_set.X.shape)
    fig, ax = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    ax[0].plot(np.arange(0, 5000), data.datasets.X[0][:5000, 0], label='iEEG signal')
    # ax[0].plot(np.arange(0, 5000), data.datasets.y[0][:5000], label='Velocity value')
    ax[1].plot(np.arange(0, 5000), data.datasets.X[10][:5000, 0], label='iEEG signal')
    ax[1].plot(np.arange(0, 5000), data.datasets.y[0][:5000], label='Velocity value')

    ax[1].set_xlabel('Time in samples')
    ax[0].set_ylabel('Electrode 1')
    ax[1].set_ylabel('Electrode 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('myplot.png', dpi=300)
    plt.show()

    # ax[0].plot(np.arange(0, 1200), data.datasets.y[0][:1200])
    # ax[0].set_title('Velocity')
    # ax[0].plot(np.arange(0, 1200), shifted_data.datasets.y[0][:1200] - np.abs(data.datasets.y[0][:1200]))
    # ax[0].set_xlabel('Time in samples')
    # ax[0].set_ylabel('Variable value')

    # plt.legend()
    # ax[1].plot(np.arange(0, 1200), shifted_data.datasets.X[0][:1200, 0], label= 'shifted Xs')
    # ax[1].plot(np.arange(0, 1200), shifted_data.datasets.y[0][:1200])
    # ax[1].set_title('Absolute velocity')
    # ax[1].plot(np.arange(0, 1200), shifted_data.datasets.y[0][:1200] - np.abs(data.datasets.y[0][:1200]),
    #            label='difference between \nabsolute velocity and\nabsolute value of velocity')
    # ax[1].set_xlabel('Time in samples')

    # plt.legend()
    # plt.show()
    # ax[1].set_title('Shifted middle - 100')
    # plt.legend()
    # ax[2].plot(np.arange(0, 1200), no_shift_data.datasets.X[0][:1200, 0], label='shifted Xs')
    # ax[2].plot(np.arange(0, 1200), no_shift_data.datasets.y[0][:1200], label='shifted ys')
    # ax[2].set_title('Initial')
    #
    # data.cut_input(1200, 519, False)
    # ax[1].plot(np.arange(0, 1200), data.train_set.X[0, 0, :, 0], label='cut initial Xs')
    # ax[1].plot(np.arange(1200-519, 1200), data.train_set.y[0, :], label='cut initial ys')
    # plt.legend()
    #
    # ax[3].plot(np.arange(0, 1200), shifted_data.train_set.X[0, 0, :, 0], label='cut shifted Xs')
    # ax[3].plot(np.arange(1200 - 519, 1200), shifted_data.train_set.y[0, :], label='cut shifted ys')
    # plt.legend()
    # plt.show()
    # print(data)
