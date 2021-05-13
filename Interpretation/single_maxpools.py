import os
from pathlib import Path

import torch
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var
from torch import nn

from Interpretation.interpretation import reshape_Xs
from Interpretation.manual_manipulation import manually_manipulate_signal
from data.pre_processing import Data
from global_config import input_time_length, home
import numpy as np

"""Script which implements passing the signal through single max-pool layers."""


class MaxPoolModel(torch.nn.Module):
    def __init__(self, num_of_layers, kernel_sizes, strides, dilations):
        super().__init__()
        self.num_of_layers = num_of_layers
        self.maxpools = []
        for i in range(num_of_layers):
            self.maxpools.append(torch.nn.MaxPool2d(kernel_sizes[i], stride=strides[i], dilation=dilations[i]))

    def forward(self, x):
        for i in range(self.num_of_layers):
            x = self.maxpools[i](x)
        return x


def get_maxpool_layers(num_of_layers, kernel_sizes, strides, dilations):
    model = MaxPoolModel(num_of_layers, kernel_sizes, strides, dilations)
    data = Data(home + '/previous_work/ALL_11_FR1_day1_absVel.mat', -1)
    return model, data


if __name__ == '__main__':
    num_of_layers = 1
    kernel_size = '3'
    strides = '1'
    dilations = '3'
    w_size = 1038

    graph_output = os.path.join(home + f'/outputs/max_pool_graphs/model_k_{kernel_size}_s{strides}_d{dilations}')
    model_output = os.path.join(home + f'/models/maxpool_models/model_k_{kernel_size}_s{strides}_d{dilations}')
    model, data = get_maxpool_layers(1, [(3, 1)], [(1, 1)]*num_of_layers, [(3, 1)])
    torch.save(model, model_output)
    test_input = np_to_var(
        np.ones((2, data.in_channels, input_time_length, 1), dtype=np.float32))
    test_out = model(test_input)
    n_preds_per_input = test_out.shape[2]
    data.cut_input(input_time_length, n_preds_per_input, False)
    x_reshaped = reshape_Xs(w_size, np.asarray(data.train_set.X))

    Path(graph_output).mkdir(exist_ok=True, parents=True)
    manually_manipulate_signal(np.asarray(data.train_set.X), graph_output, model, maxpool_model=True, white_noise=True)

