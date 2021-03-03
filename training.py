import argparse
from pathlib import Path
import os
import pandas

from Training.train import train_nets
from global_config import home, random_seed, cuda
import torch
from braindecode.util import set_random_seeds

import random

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
activations = {}

parser = argparse.ArgumentParser()
parser.add_argument("--kernel_size", default=[3, 3, 3, 3], type=int, nargs=4, help="Render some episodes.")
parser.add_argument("--dilations", default=[3, 9, 27, 81], type=int, nargs=4, help="Random seed.")
parser.add_argument("--starting_patient_index", default=1, type=int, help="Learning rate.")
parser.add_argument('--variable', default=0, type=int)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


if __name__ == '__main__':
    args = parser.parse_args()
    input_time_length = 1200
    max_train_epochs = 300
    batch_size = 16
    print(cuda, home)
    set_random_seeds(seed=random_seed, cuda=cuda)
    cropped = True
    num_of_folds = -1
    trajectory_index = args.variable
    learning_rate = 0.001
    low_pass = False
    shift = False
    high_pass = False
    high_pass_valid = False
    add_padding = False

    if trajectory_index == 0:
        model_string = f'p_vel'
        variable = 'vel'
    else:
        model_string = 'p_absVel'
        variable = 'absVel'

    model_name = ''

    best_valid_correlations = []
    dilations = [None, [1, 1, 1, 1], [2, 4, 8, 16]]

    if args.kernel_size == [1, 1, 1, 1]:
        dilations = [None]
    for dilation in dilations:
        best_valid_correlations = []
        print(dilation)
        kernel_size = args.kernel_size
        print(kernel_size)
        model_name = ''.join([str(x) for x in kernel_size])

        if dilation is not None:
            dilations_name = ''.join(str(x) for x in dilation)
            model_name = f'{model_name}_dilations_{dilations_name}'

        starting_patient_index = args.starting_patient_index
        if num_of_folds != -1:
            if os.path.exists(
                    f'{home}/outputs/performance/lr_{learning_rate}/{model_string}_k_{model_name}/{variable}_performance.csv'):
                df = pandas.read_csv(
                    f'{home}/outputs/performance/lr_{learning_rate}/{model_string}_k_{model_name}/{variable}_performance.csv',
                    sep=';', index_col=0)
                df = df.T.drop_duplicates().T
                starting_patient_index = df.shape[1] + 1
                print('exists')
            else:
                df = pandas.DataFrame()
        else:
            df = pandas.DataFrame()
        print(starting_patient_index)

        train_nets(model_string, [x for x in range(starting_patient_index, 13)], dilation, kernel_size,
                   lr=learning_rate,
                   num_of_folds=num_of_folds, trajectory_index=trajectory_index, low_pass=low_pass, shift=shift,
                   variable=variable, result_df=df, max_train_epochs=max_train_epochs, high_pass=high_pass,
                   high_pass_valid=high_pass_valid, padding=add_padding, cropped=cropped)
