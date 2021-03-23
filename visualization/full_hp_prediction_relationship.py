import argparse
import pickle
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
from braindecode import EEGRegressor
from braindecode.models.util import get_output_shape
from braindecode.util import np_to_var
from skorch.callbacks import TensorBoard, Checkpoint

from Interpretation.interpretation import get_corr_coef
from Training.CorrelationMonitor1D import CorrelationMonitor1D
from Training.train import get_model, get_writer
from data.pre_processing import get_num_of_channels, Data
from global_config import home, input_time_length, cuda, random_seed
from models.DoubleModel import DoubleModel
from models.Model import load_model

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
activations = {}

parser = argparse.ArgumentParser()
parser.add_argument('--variable', default=1, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    dilation = [None]
    kernel_size = [3, 3, 3, 3]
    trajectory_index = args.variable
    low_pass = False
    shift_by = None
    lr = 0.001
    shift = True
    max_train_epochs = 100
    if trajectory_index == 0:
        model_string = f'sbp0_sm_vel_shuffled'
        variable = 'vel'
    else:
        model_string = 'sbp0_sm_absVel_shuffled'
        variable = 'absVel'
    saved_model_dir = f'lr_{lr}'
    all_xs = []
    all_ys = []
    all_zs = []
    corr_coefs_full = []
    corr_coefs_hp = []
    for patient_index in range(1, 13):
        input_channels = get_num_of_channels(home + f'/previous_work/P{patient_index}_data.mat')
        changed_model_full = load_model(f'/models/saved_models/{saved_model_dir}/sbp0_shuffled_sm_{variable}_k3_d3/sbp0_shuffled_sm_{variable}_k3_d3_p_{patient_index}//best_model_split_0')
        changed_model_hp = load_model(f'/models/saved_models/{saved_model_dir}/sbp0_shuffled_hp_sm2_{variable}_k3_d3/sbp0_shuffled_hp_sm2_{variable}_k3_d3_p_{patient_index}//best_model_split_0')
        model_name = 'k3_d3'
        n_preds_per_input = get_output_shape(changed_model_full, input_channels, input_time_length)[1]
        small_window = input_time_length - n_preds_per_input + 1
        if shift_by is None:
            shift_index = int(small_window / 2)
        else:
            shift_index = int((small_window / 2) - shift_by)
        train_file = open(f'{home}/models/indices/train.dict', 'rb')
        valid_file = open(f'{home}/models/indices/valid.dict', 'rb')

        train_indices = pickle.load(train_file)
        valid_indices = pickle.load(valid_file)

        data_full = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=-1,
                         low_pass=low_pass,
                         trajectory_index=trajectory_index, shift_data=shift, high_pass=False,
                         shift_by=int(shift_index),
                         valid_high_pass=False, low_pass_training=False, double_training=True,
                         train_indices=train_indices[f'P{patient_index}'], valid_indices=valid_indices[f'P{patient_index}'])
        data_hp = Data(home + f'/previous_work/P{patient_index}_data.mat', num_of_folds=-1,
                       low_pass=low_pass,
                       trajectory_index=trajectory_index, shift_data=shift, high_pass=True,
                       shift_by=int(shift_index),
                       valid_high_pass=False, low_pass_training=False,
                       train_indices=train_indices[f'P{patient_index}'], valid_indices=valid_indices[f'P{patient_index}'])
        output_dir = f'{model_string}_{model_name}/{model_string}_{model_name}_p_{patient_index}'
        correlation_monitor = CorrelationMonitor1D(input_time_length=input_time_length,
                                                   output_dir=output_dir)

        if cuda:
            device = 'cuda'
            changed_model_full = changed_model_full.cuda()
            changed_model_hp = changed_model_hp.cuda()

        else:
            device = 'cpu'

        model = DoubleModel(changed_model_full, changed_model_hp)

        data_hp.cut_input(input_time_length, n_preds_per_input, False)
        data_full.cut_input(input_time_length, n_preds_per_input, False)
        # data_hp.train_set.X = np.zeros(data_hp.train_set.X.shape)

        xs = []
        ys = []
        zs = []
        corr_coef_full = get_corr_coef(data_full.test_set, changed_model_full)
        corr_coef_hp = get_corr_coef(data_hp.test_set, changed_model_hp)
        print('coef_full:', corr_coef_full, 'coef_hp:', corr_coef_hp)
        corr_coefs_full.append(corr_coef_full)
        corr_coefs_hp.append(corr_coef_hp)
        output = f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid/graph_patient_{patient_index}.png'
        full_out = f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid/graph_patient_{patient_index}_full.png'
        hp_out = f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid/graph_patient_{patient_index}_hp.png'
        Path(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid/').mkdir(parents=True, exist_ok=True)
        Path(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid_data/').mkdir(parents=True, exist_ok=True)
        df = pandas.DataFrame()
        for input_full, input_hp, correct_out in zip(data_full.test_set.X, data_hp.test_set.X, data_full.test_set.y):
            out_full = changed_model_full.double()(np_to_var(input_full.reshape([1, input_full.shape[0], input_full.shape[1], input_full.shape[2]])).double())
            out_hp = changed_model_hp.double()(np_to_var(input_hp.reshape(1, input_full.shape[0], input_full.shape[1], input_full.shape[2])).double())
            xs.append(out_full)
            ys.append(out_hp)
            zs.append(correct_out.reshape([1, correct_out.shape[0]]))
            # out_full = out_full.reshape([out_full.shape[1]])
            # out_hp = out_hp.reshape([out_hp.shape[1]])

            # print(correct_out.shape)
            # fig = plt.figure()
            #
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter3D(out_full, out_hp, correct_out, c=correct_out, cmap='Greens')
            # ax.set_xlabel('Full model predictions')
            # ax.set_ylabel('High-pass model predictions')
            # ax.set_zlabel('True values')
            # plt.show()

        xs = np.concatenate(xs, axis=1)
        ys = np.concatenate(ys, axis=1)
        zs = np.concatenate(zs, axis=1)

        df['X'] = xs.reshape([xs.shape[1]])
        df['Y'] = ys.reshape([ys.shape[1]])
        df['Z'] = zs.reshape([zs.shape[1]])
        df.to_csv(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid_data/df_patient_{patient_index}.csv', sep=';')
        # df = pandas.read_csv(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid_data/df_patient_{patient_index}.csv', sep=';', index_col=[0])
        # xs = df['X'].tolist()
        # ys = df['Y'].tolist()
        # zs = df['Z'].tolist()
        all_xs.append(xs)
        all_ys.append(ys)
        all_zs.append(zs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        points = ax.scatter(xs, ys, zs, c=zs, cmap='coolwarm')
        fig.colorbar(points, label='True prediction')
        ax.set_xlabel(f'Full model predictions\n corr: {corr_coef_full:.2f}', labelpad=10)
        ax.set_ylabel(f'High-pass model predictions\n corr: {corr_coef_hp:.2f}', labelpad=10)
        # ax.set_zlabel('True values')
        plt.title(f'Patient {patient_index} - {variable}')
        plt.tight_layout()
        plt.savefig(output)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        points = ax.scatter(xs, zs, c=zs, cmap='coolwarm')
        # fig.colorbar(points, label='True prediction')
        ax.set_xlabel(f'Full model predictions\n corr: {corr_coef_full:.2f}', labelpad=10)
        ax.set_ylabel(f'True values', labelpad=10)
        # ax.set_zlabel('True values')
        plt.title(f'Patient {patient_index} - {variable}')
        plt.tight_layout()
        plt.savefig(full_out)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        points = ax.scatter(ys, zs, c=zs, cmap='coolwarm')
        # fig.colorbar(points, label='True prediction')
        ax.set_xlabel(f'High-pass model predictions\n corr: {corr_coef_hp:.2f}', labelpad=10)
        ax.set_ylabel(f'True values', labelpad=10)
        # ax.set_zlabel('True values')
        plt.title(f'Patient {patient_index} - {variable}')
        plt.tight_layout()
        plt.savefig(hp_out)
        plt.show()

    all_patient_df = pandas.DataFrame()
    all_xs = np.concatenate(all_xs, axis=1)
    all_ys = np.concatenate(all_ys, axis=1)
    all_zs = np.concatenate(all_zs, axis=1)
    all_patient_df['X'] = all_xs
    all_patient_df['Y'] = all_ys
    all_patient_df['Z'] = all_zs
    all_patient_df.to_csv(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/valid_data/df_all_patients.csv',
        sep=';')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # points = ax.scatter(all_xs, all_ys, all_zs, c=all_zs, cmap='coolwarm')
    # fig.colorbar(points, label='True values')
    # ax.set_xlabel(f'Full model predictions', labelpad=10)
    # ax.set_ylabel(f'High-pass model predictions', labelpad=10)
    # ax.set_zlabel('True values')
    # plt.title(f'All patients')
    # plt.tight_layout()
    # plt.savefig(f'{home}/results/model_to_true_predictions/{model_string}_{model_name}/{variable}/all_patient_graph_3d.png')
    # plt.show()

