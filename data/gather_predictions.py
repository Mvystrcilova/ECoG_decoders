import torch
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var

from data.pre_processing import read_mat_file, Data
from global_config import input_time_length, cuda
from models.Model import load_model
import numpy as np


def get_network_predictions(model, hp_data, input_channels):
    model.requires_grad_(False)
    n_preds_per_input = get_output_shape(model, input_channels, input_time_length)[1]
    hp_data.cut_input(input_time_length=input_time_length, n_preds_per_input=n_preds_per_input, shuffle=False)
    # hp_network_predictions = []
    print('hp data fold number:', {hp_data.fold_number})

    dataset_train, dataset_valid = hp_data.cv_split(np.stack(hp_data.train_set.X), np.stack(hp_data.train_set.y))

    if cuda:
        preds_2 = model(np_to_var(dataset_train.X[int(len(dataset_train.X)/2):]).cuda())
        preds_1 = model(np_to_var(dataset_train.X[:int(len(dataset_train.X)/2)]).cuda())
        valid_preds = model(np_to_var(dataset_valid.X).cuda())
        preds = torch.cat((preds_1, preds_2), dim=0)
    else:
        preds_2 = model(np_to_var(dataset_train.X[int(len(dataset_train.X) / 2):]))
        preds_1 = model(np_to_var(dataset_train.X[:int(len(dataset_train.X) / 2)]))
        preds = torch.cat((preds_1, preds_2), dim=0)
        valid_preds = model(np_to_var(dataset_valid.X))
    print(preds.shape)
    # hp_network_predictions.append(preds)

    return preds, valid_preds

