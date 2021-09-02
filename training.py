import argparse
import pickle
from pathlib import Path
import os
import pandas

from braindecode import EEGRegressor
from Training.train import train_nets
from global_config import home, random_seed, cuda, get_model_name_from_kernel_and_dilation, vel_string, absVel_string
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
parser.add_argument("--dilations", default=[1, 3, 9, 27], type=int, nargs=4, help="Random seed.")
parser.add_argument("--starting_patient_index", default=1, type=int, help="Learning rate.")
parser.add_argument('--variable', default=0, type=int)
parser.add_argument('--dummy_dataset', default=False, type=bool, help='Training the network on a dummy dataset')


if __name__ == '__main__':
    """
    This script allows to specify the configurations of the networks and datasets that 
    are being trained. The parameters are explained in README.md
    """
    args = parser.parse_args()
    input_time_length = 1200
    max_train_epochs = 100
    batch_size = 16
    print(cuda, home)
    set_random_seeds(seed=random_seed, cuda=cuda)
    cropped = True
    num_of_folds = 5
    indices = None
    if num_of_folds != -1:
        with open(f'{home}/data/train_dict_{num_of_folds}', 'rb') as file:
            indices = pickle.load(file)
    trajectory_index = args.variable
    learning_rate = 0.001
    low_pass = False
    shift = False
    high_pass = True
    high_pass_valid = True
    add_padding = False
    low_pass_training = False
    whiten = False
    if num_of_folds != -1:
        saved_model_dir = f'lr_{learning_rate}_{num_of_folds}'
    else:
        saved_model_dir = f'lr_{learning_rate}'
    if whiten:
        saved_model_dir = f'pre_whitened_{num_of_folds}'
    dummy_dataset = args.dummy_dataset
    print('dummy dataset', dummy_dataset)
    if dummy_dataset:
        dummy_string = 'dummy_'
    else:
        dummy_string = ''

    if trajectory_index == 0:
        model_string = f'{dummy_string}hp_for_hp_m_{vel_string}'
        variable = vel_string
    else:
        model_string = f'{dummy_string}hp_for_hp_m_{absVel_string}'
        variable = absVel_string

    best_valid_correlations = []
    dilations = [None, [1, 1, 1, 1], [2, 4, 8, 16]]
    # dilations = [None]
    if args.kernel_size == [1, 1, 1, 1]:
        dilations = [None]

    model_name = ''
    for dilation in dilations:
        best_valid_correlations = []
        print(dilation)
        kernel_size = args.kernel_size
        print(kernel_size)
        model_name = get_model_name_from_kernel_and_dilation(kernel_size, dilation)

        starting_patient_index = args.starting_patient_index

        if os.path.exists(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/performances.csv'):
            df = pandas.read_csv(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/performances.csv', sep=';', index_col=0)
            starting_patient_index = int(df.columns[-1].split('_')[1]) + 1
            print('dataframe found, starting index:', starting_patient_index)
        else:
            Path(f'{home}/outputs/performances_{num_of_folds}/{model_string}_{model_name}/').mkdir(exist_ok=True, parents=True)

            df = pandas.DataFrame()

        print('starting with patient: ', starting_patient_index)
        # for s in shifts2:
        #     print('shifting by:', s)
        #     saved_model_dir = saved_model_dir + f'/shift_{s}/'
        hp_model_name = f'hp_m_{variable}_{get_model_name_from_kernel_and_dilation(kernel_size, dilation)}'
        print(hp_model_name, model_name)
        train_nets(model_string, [x for x in range(starting_patient_index, 13)], dilation, kernel_size,
                   lr=learning_rate,
                   num_of_folds=num_of_folds, trajectory_index=trajectory_index, low_pass=low_pass, shift=shift,
                   variable=variable, result_df=df, max_train_epochs=max_train_epochs, high_pass=high_pass,
                   high_pass_valid=high_pass_valid, padding=add_padding, cropped=cropped,
                   low_pass_train=low_pass_training, shift_by=None, saved_model_dir=saved_model_dir, whiten=whiten,
                   indices=indices, dummy_dataset=dummy_dataset, mimic_hp_predictions=True, hp_model_file=hp_model_name)
