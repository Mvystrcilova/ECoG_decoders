import mne
from braindecode.datasets.xy import create_from_X_y
from BCICIV_competition.data_visualization import read_mat_file, get_mutlivariate_data
import numpy as np
import pandas as pd
import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGRegressor
from braindecode.datautil import create_fixed_length_windows
from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.training.losses import CroppedLoss
from braindecode.models import Deep4Net
from braindecode.models import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from braindecode.util import set_random_seeds, create_mne_dummy_raw


def create_compatible_dataset(mat_file):
    train_data, labels = read_mat_file(mat_file)
    validation_data = train_data[7500:10000]
    validation_labels = labels[7500:10000]
    train_data = train_data[:7500]
    labels = labels[:7500]
    # train_data = train_data.transpose()
    new_train_data = get_mutlivariate_data(train_data, 500)
    new_validation_data = get_mutlivariate_data(validation_data, 500)

    # train_data = train_data.reshape([train_data.shape[0], train_data.shape[1], 1])
    compatible_train_dataset = create_from_X_y(new_train_data, labels, drop_last_window=False, sfreq=1000,
                                         window_size_samples=500, window_stride_samples=10)
    compatible_valid_dataset = create_from_X_y(new_validation_data, validation_labels, drop_last_window=False,
                                               sfreq=1000, window_size_samples=500, window_stride_samples=10)

    return compatible_train_dataset, compatible_valid_dataset


model_name = "shallow"  # 'shallow' or 'deep'
n_epochs = 3
seed = 20200220

input_window_samples = 7500
batch_size = 64
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

n_chans = 62
# set to how many targets you want to regress (age -> 1, [x, y, z] -> 3)
n_classes = 5

set_random_seeds(seed=seed, cuda=cuda)

# initialize a model, transform to dense and move to gpu
if model_name == "shallow":
    model = ShallowFBCSPNet(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=40,
        n_filters_spat=40,
        final_conv_length=35,
    )
    optimizer_lr = 0.000625
    optimizer_weight_decay = 0
elif model_name == "deep":
    model = Deep4Net(
        in_chans=n_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        stride_before_pool=True,
        n_filters_2=int(n_chans * 2),
        n_filters_3=int(n_chans * (2 ** 2.0)),
        n_filters_4=int(n_chans * (2 ** 3.0)),
        final_conv_length=1,
    )
    optimizer_lr = 0.01
    optimizer_weight_decay = 0.0005
else:
    raise ValueError(f'{model_name} unknown')

new_model = torch.nn.Sequential()
for name, module_ in model.named_children():
    if "softmax" in name:
        continue
    new_model.add_module(name, module_)
model = new_model

if cuda:
    model.cuda()

to_dense_prediction_model(model)
n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

train_set, valid_set = create_compatible_dataset('./data/BCICIV_4_mat/sub1_comp.mat')

# dataset = create_fixed_length_windows(
#     dataset,
#     start_offset_samples=0,
#     stop_offset_samples=0,
#     window_size_samples=input_window_samples,
#     window_stride_samples=n_preds_per_input,
#     drop_last_window=False,
#     drop_bad_windows=True,
# )

# splits = dataset.split("session")
# train_set = splits["train"]
# valid_set = splits["eval"]

regressor = EEGRegressor(
    model,
    cropped=True,
    criterion=CroppedLoss,
    criterion__loss_function=torch.nn.functional.mse_loss,
    optimizer=torch.optim.AdamW,
    train_split=predefined_split(valid_set),
    optimizer__lr=optimizer_lr,
    optimizer__weight_decay=optimizer_weight_decay,
    iterator_train__shuffle=True,
    batch_size=batch_size,
    callbacks=[
        "neg_root_mean_squared_error",
        # seems n_epochs -1 leads to desired behavior of lr=0 after end of data?
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
)

regressor.fit(train_set, y=None, epochs=n_epochs)

if __name__ == '__main__':
    create_compatible_dataset('./data/BCICIV_4_mat/sub1_comp.mat')
