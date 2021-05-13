import argparse
import os
from pathlib import Path

import pandas
import pickle

from Training.train import train_nets
from global_config import home, random_seed, cuda, get_model_name_from_kernel_and_dilation, vel_string, absVel_string
import torch
from braindecode.util import set_random_seeds
from torch.utils.tensorboard.writer import SummaryWriter
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


""" 
This script allows to specify the configurations of the networks and datasets that 
are being trained. Even though it is called shifted_training.py it can be used to train any 
network on any dataset based on the parameters. The parameters are explained in README.md
"""

if __name__ == '__main__':
    args = parser.parse_args()
    input_time_length = 1200
    max_train_epochs = 100
    batch_size = 16
    print(cuda, home)
    set_random_seeds(seed=random_seed, cuda=cuda)
    cropped = True
    low_pass = False
    trajectory_index = args.variable
    num_of_folds = 5
    indices = None
    if num_of_folds != -1:
        with open(f'{home}/data/train_dict_{num_of_folds}', 'rb') as file:
            indices = pickle.load(file)
    shift = True
    learning_rate = 0.001
    high_pass = False
    high_pass_valid = False
    low_pass_train = False

    print('low_pass:', low_pass)
    print('shift:', shift)
    print('high-pass:', high_pass)
    print('high-pass valid:', high_pass_valid)
    print('low-pass train:', low_pass_train)

    whiten = False
    saved_model_dir = f'lr_{learning_rate}_{num_of_folds}'
    print('whiten:', whiten)
    if whiten:
        saved_model_dir = f'pre_whitened_{num_of_folds}'

    if trajectory_index == 0:
        model_string = f'abs_m_{vel_string}'
        variable = vel_string
    else:
        model_string = f'abs_m_{absVel_string}'
        variable = absVel_string

    model_name = ''

    best_valid_correlations = []

    dilations = [None, [1, 1, 1, 1], [2, 4, 8, 16]]
    # dilations = [None]
    if args.kernel_size == [1, 1, 1, 1]:
        dilations = [None]

    for dilation in dilations:
        best_valid_correlations = []
        print(dilation)
        kernel_size = args.kernel_size
        print(kernel_size)
        # model_name = ''.join([str(x) for x in kernel_size])
        # if dilation is not None:
        #     dilations_name = ''.join(str(x) for x in dilation)
        model_name = get_model_name_from_kernel_and_dilation(kernel_size, dilation)

        starting_patient_index = args.starting_patient_index

        if os.path.exists(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/performances.csv'):
            df = pandas.read_csv(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/performances.csv', sep=';',
                                 index_col=0)
            starting_patient_index = int(df.columns[-1].split('_')[1]) + 1
        else:
            Path(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/').mkdir(exist_ok=True, parents=True)
            df = pandas.DataFrame()
        print(starting_patient_index)

        train_nets(model_string, [x for x in range(starting_patient_index, 13)], dilation, kernel_size, lr=learning_rate,
                   num_of_folds=num_of_folds, trajectory_index=trajectory_index, low_pass=low_pass, shift=shift,
                   variable=variable, result_df=df, max_train_epochs=max_train_epochs, high_pass_valid=high_pass_valid,
                   low_pass_train=low_pass_train, whiten=whiten, saved_model_dir=saved_model_dir, high_pass=high_pass,
                   indices=indices)