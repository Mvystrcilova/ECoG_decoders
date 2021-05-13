from skorch.dataset import Dataset
import numpy as np
from data.pre_processing import read_mat_file, band_pass_data


def get_crops(X_trial, y_trial, input_time_length):
    X_crops = []
    y_crops = []
    i = 0
    while i + input_time_length < len(y_trial):
        X_crops.append(X_trial[i:i+input_time_length, :])
        y_crops.append(y_trial[i+input_time_length])
        i += 1
    return X_crops, y_crops


def cut_Xy(Xs, ys, input_time_length):
    X_data = []
    y_data = []
    for X, y in zip(Xs, ys):
        X_crops, y_crops = get_crops(X, y, input_time_length)
        X_data = X_data + X_crops
        y_data = y_data + y_crops
    return X_data, y_data


class OnePredictionData:
    """ Not finished class for building a network with an uniform receptive field"""
    def __init__(self, mat_file, input_time_length, num_of_folds, low_pass, high_pass, valid_high_pass, trajectory_index):
        self.Xs, self.ys = None, None
        self.lp_Xs, self.lp_ys, self.hp_ys, self.hp_Xs = None, None, None, None
        self.input_time_length = input_time_length
        self.data = read_mat_file(mat_file)
        self.low_pass, self.high_pass, self.valid_high_pass = low_pass, high_pass, valid_high_pass
        self.num_of_folds = num_of_folds
        self.trajectory_index = trajectory_index
        self.create_dataset()
        self.train_set = None
        self.valid_set = None

        self.in_channels = self.Xs[0].shape[1]

    def create_dataset(self):
        sessions = self.data.D
        Xs = [session[0].ieeg[:] for session in sessions]
        ys = [session[0].traj[:, self.trajectory_index] for session in sessions]

        if self.low_pass:
            self.lp_Xs, self.lp_ys = band_pass_data(Xs, ys, 3, 40, 'low')
        elif self.high_pass:
            self.lp_Xs, self.lp_ys = band_pass_data(Xs, ys, 3, 60, 'hp')
        if self.valid_high_pass:
            self.hp_Xs, self.hp_ys = band_pass_data(Xs, ys, 3, 60, 'hp')

        self.Xs = Xs
        self.ys = ys

    def cut_input(self, input_time_length, n_preds_per_input, shuffle=False):
        self.Xs, self.ys = cut_Xy(self.Xs, self.ys, self.input_time_length)
        if self.low_pass:
            self.lp_Xs, self.lp_ys = cut_Xy(self.lp_Xs, self.lp_ys, self.input_time_length)
        if self.high_pass or self.valid_high_pass:
            self.hp_Xs, self.hp_ys = cut_Xy(self.hp_Xs, self.hp_ys, self.input_time_length)
        self.train_set = np.asarray(self.Xs).reshape([-1, self.Xs[0].shape[1], self.Xs[0].shape[0], 1]), np.asarray(self.ys).reshape([-1, 1])

    def cv_split(self, X, y):
        length = len(X.X)
        index = int((length/100)*20)
        # print(X.X.shape)
        # print(y.shape)
        self.train_set = Dataset(X.X[:-index], y[:-index])
        self.valid_set = Dataset(X.X[-index:], y[-index:])

        return self.train_set, self.valid_set






