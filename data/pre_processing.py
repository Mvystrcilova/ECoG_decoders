import mat73
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.splitters import concatenate_sets
import numpy as np


def read_mat_file(mat_file):
    data = mat73.loadmat(mat_file)
    return data


class Data:
    def __init__(self, mat_file, num_of_folds):
        self.data = read_mat_file(mat_file)
        self.datasets = self.create_datasets()
        self.train_set, self.valid_set, self.test_set = self.split_data(num_of_folds)
        self.in_channels = self.train_set.X[0].shape[0]
        self.n_classes = self.train_set.y[0].shape[2]

    def create_datasets(self):
        return [SignalAndTarget([l[0].ieeg.astype(np.float32)], [l[0].traj.astype()]) for l in self.data.D]

    def split_data(self, num_of_folds):
        indices = np.arange(len(self.datasets))[-num_of_folds:]
        test_inds = indices[-1]
        valid_inds = indices[-2]
        train_inds = indices[:-1]

        train_set = concatenate_sets([self.data[i] for i in train_inds])
        valid_set = self.data[valid_inds] # dummy variable, could be set to None
        test_set = self.data[test_inds]
        return train_set, valid_set, test_set


if __name__ == '__main__':
    data = read_mat_file('../previous_work/ALL_11_FR1_day1_absVel.mat')