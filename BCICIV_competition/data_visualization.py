import matplotlib
import scipy.io
import pandas
import numpy as np
from scipy.signal import wiener
from sklearn import linear_model


def get_mutlivariate_data(data, vector_length):
    shape = data.shape
    time_steps, channels = data.shape
    new_train_data = np.zeros([time_steps, channels, vector_length])
    for i in range(vector_length - 1, time_steps):
            vector = data[i-(vector_length-1):i+1, :]
            vector = vector.transpose()
            # print(new_train_data[i])
            new_train_data[i] = vector
    # new_train_data  = new_train_data.reshape([time_steps, channels*vector_length])
    return new_train_data


def read_mat_file(mat_file):
    data = scipy.io.loadmat(mat_file)
    train_data = data['train_data']
    labels = data['train_dg']
    print(train_data.shape)
    return train_data, labels


def multivariate_data():
    train_data, labels = read_mat_file('data/BCICIV_4_mat/sub1_comp.mat')
    new_train_data = get_mutlivariate_data(train_data, 5)
    wiener(new_train_data, labels[:, 0])


def fit_linear_regression(train, labels):
    model = linear_model.LinearRegression()
    regr = model.fit(train, labels)
    print(regr)


def running_average_calculations(data):
    pass

# multivariate_data()