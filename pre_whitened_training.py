import argparse
from pathlib import Path
import os
import pandas

from Training.train import train_nets
from global_config import home, random_seed, cuda, get_model_name_from_kernel_and_dilation
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


if __name__ == '__main__':
    args = parser.parse_args()
    input_time_length = 1200
    max_train_epochs = 100
    batch_size = 16
    print(cuda, home)
    set_random_seeds(seed=random_seed, cuda=cuda)
    cropped = True
    num_of_folds = -1
    trajectory_index = args.variable
    learning_rate = 0.001
    low_pass = False
    shift = False
    high_pass = True
    high_pass_valid = False
    add_padding = False
    low_pass_training = False
    whiten = True
    saved_model_dir = f'lr_{learning_rate}'
    if whiten:
        saved_model_dir = 'pre_whitened'

    if trajectory_index == 0:
        model_string = f'hp_m_vel'
        variable = 'vel'
    else:
        model_string = 'hp_m_absVel'
        variable = 'absVel'

    best_valid_correlations = []
    dilations = [None, [1, 1, 1, 1], [2, 4, 8, 16]]
    # dilations = [None]
    if args.kernel_size == [1, 1, 1, 1]:
        dilations = [None]
    # shifts7 = [-250, -225, -200, -175]
    # shifts6 = [-150, -125, -100, -75]
    # shifts2 = [-50, -25, 25, 0]
    # shifts3 = [50, 75, 100, 125, 150]
    # shifts4 = [175, 200, 225, 250]
    # shifts8 = [150]
    # shifts = shifts6 + shifts4 + shifts3 + shifts7 + shifts2
    model_name = ''
    for dilation in dilations:
        best_valid_correlations = []
        print(dilation)
        kernel_size = args.kernel_size
        print(kernel_size)
        model_name = get_model_name_from_kernel_and_dilation(kernel_size, dilation)

        starting_patient_index = args.starting_patient_index

        df = pandas.DataFrame()

        print(starting_patient_index)
        # for s in shifts2:

        train_nets(model_string, [x for x in range(starting_patient_index, 13)], dilation, kernel_size,
                   lr=learning_rate,
                   num_of_folds=num_of_folds, trajectory_index=trajectory_index, low_pass=low_pass, shift=shift,
                   variable=variable, result_df=df, max_train_epochs=max_train_epochs, high_pass=high_pass,
                   high_pass_valid=high_pass_valid, padding=add_padding, cropped=cropped,
                   low_pass_train=low_pass_training, shift_by=None, saved_model_dir=saved_model_dir, whiten=whiten)
