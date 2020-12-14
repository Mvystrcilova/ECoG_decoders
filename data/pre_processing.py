import mat73
import numpy as np
from skorch.dataset import CVSplit, Dataset
from Training.CropsFromTrialsIterator import CropsFromTrialsIterator


def read_mat_file(mat_file):
    data = mat73.loadmat(mat_file)
    return data


class MyDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


class Data:
    def __init__(self, mat_file, num_of_folds):
        self.data = read_mat_file(mat_file)
        self.datasets = self.create_datasets()
        self.num_of_folds = num_of_folds
        self.train_set, self.valid_set, self.test_set = self.split_data()
        self.in_channels = self.train_set.X[0].shape[1]
        self.n_classes = len(self.train_set.y[0].shape)
        self.fold_number = 0

    def create_datasets(self):
        sessions = self.data.D
        Xs = [session[0].ieeg[:] for session in sessions]
        ys = [session[0].traj for session in sessions]
        print(len(Xs), len(ys))
        dataset = Dataset(Xs, ys)
        return dataset

    def split_data(self, ):
        train_set = MyDataset(self.datasets.X[:-3], self.datasets.y[:-3])
        test_set = MyDataset(self.datasets.X[-3:], self.datasets.y[-3:])
        return train_set, None, test_set

    def cut_input(self, input_time_length, n_preds_per_input, shuffle):
        iterator = CropsFromTrialsIterator(batch_size=32,
                                           input_time_length=input_time_length,
                                           n_preds_per_input=n_preds_per_input)
        self.train_set = concatenate_batches(self.train_set, iterator, False)
        self.test_set = concatenate_batches(self.test_set, iterator, False)

    def cv_split(self, X, y):
        length = len(X.X)
        if self.num_of_folds == -1:
            train_set = Dataset(X.X[:], y[:])
            return train_set, self.test_set

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
    data = read_mat_file('../previous_work/ALL_11_FR1_day1_absVel.mat')
