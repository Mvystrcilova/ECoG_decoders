# Master Thesis code - Documentation

This project `ECoG_decoders` was created as a part of a Master Thesis 
called "Prediction of velocity and speed of movement from human intracranial EEG recordings using deep neural networks" 
by Michaela Vystrčilová.

The project contains code which was used to perform experiments and analyses described in the thesis. 
The dataset used to perform the experiments and analysis is not publicly available. 
Therefore, it is not part of this project.

## Runnable scripts

The project has multiple runnable Python scripts located in its root.
They can be run directly or using bash scripts located 
in the `bash_scripts` folder.
Without the dataset, only scripts with a *dummy* parameter
can be executed.
The bash scripts are designed for the `slurm` environment which is present in the `gpulab` at MFF UK.
The following bash script implements the dummy dataset and can be run in the `slurm` environment 
using the following commands:

```
git clone git@github.com:Mvystrcilova/ECoG_decoders.git # clone the repository

cd ECoG_decoders
sbatch bash_scripts/run_in_ch0_k3_dummy.sh
```

The Python scripts can be also run directly.
Again only scripts with the `dummy_dataset` set to `True` can be executed without the dataset.


```
git clone git@github.com:Mvystrcilova/ECoG_decoders.git # clone the repository

cd ECoG_decoders

python3 -m pip3 install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

python3 training.py --dummy_dataset=True
```

The models trained and validated on the full dataset are available [here](https://drive.google.com/drive/folders/1KFZ8MlRURG-IrWrEkoDlWqECfNxTZfHm?usp=sharing).

### Training scripts

There are four training scripts, namely `training.py`, 
`shifted_training.py`, `high_pass_training.py` and `pre_whitened_training`.
All four of these scripts have the same functionalities. 
Except the `training.py` script which also has the option of using the *dummy_dataset*.
We created three because when running them in the `gpulab`, at any time, the process can be 
canceled and then restarted from the beginning. 
And if we change the parameters in the script between the first start of the process and the restart,
the process is restarted with the changed parameters.


#### Configuration parameters
| Parameter | Description | Possible values |
| --------- | ----------- | --------------- |
| `input_time_length` | expected size of the input window | positive integer up to 7500 which is the length of the segments (trials) in the dataset |
| `max_train_epochs` | the number of epochs for which the network is trained | positive integer, in the thesis always 100 to reproduce Hammer et al. results |
| `batch_size` | size of the batch | positive integer, in the thesis always 16 to reproduce Hammer et al. results |
| `cropped` | whether the input windows should be cropped to perform dense prediction | always `True` for experiments which are part of the thesis |
| `num_of_folds` | number of folds into which the dataset is split to perform *num_of_folds*-cross-validation | `-1`: only one 80-20 ration split; '0': leave-one-out cross-validation on the dataset segments (trials); other positive n: n-fold cross-validation |
| `trajectory index` | taken from command line arguments, specifies which variable should be decoded | `0`: velocity; `1`: absolute velocity |
| `learning rate` | learning rate | in Hammer et al. 0.01, here we set 0.001 for all experiments |
| `low_pass` | specifies if the validation set should be low-passed | `True`: validation set is low-passed; `False`: validation set is not low-passed |
| `shift` | specifies if the predicted time-point should be shifted | `True`: the predicted time-point is shifted, if `shift_by` is `None` it is shifted to the center of the receptive field, otherwise, it is shifted based on `shift_by`;`False`: the predicted time-point is at the end of the receptive field performing causal prediction. |
| `high_pass` | specifies if both the training and validation data should be high-passed | `True`: validation and training data is high-passed; `False`: validation and training data is not high-passed |
| `high_pass_valid` | specifies if the validation data should be high-passed | `True`: validation data is high-passed; `False`: validation data is not high-passed |
| `add_padding` | parameter prepared for building network with uniform receptive field, adds padding to the network | always `False` for experiments of the thesis | 
| `low_pass_training` | specifies if the training set should be low-passed | `True`: training set is low-passed; `False`: Training set is not low-passed | 
| `whiten` | specifies if the dataset should be whitened | `True`: the dataset is whitened; `False`: the dataset is not whitened |
| `model_string` | prefix which is together with the `model_name` is used as a filename under which the model is saved in the `saved_model_dir`; the prefix should describe the setting in which the model is trained  | any string to which the `variable_string` is added; for an overview of the strings used in this thesis and the settings they refer to, see table below |
| `dilations` | specifies the dilations of the max-pool layers in the network; while possible to set in the arguments, the scripts are currently set up so that disregarding the dilation parameter, it loops over `[3, 9, 27, 81]`, `[1, 1, 1, 1]` and `[2, 4, 8, 16]`. | `None`: in this case the dilations are left as in the original Deep4Net; a list of four integers, for the purposes of this thesis, also `[3, 9, 27, 81]`, `[1, 1, 1, 1]` and `[2, 4, 8, 16]` were used. |
| `kernel_size` | specifies the kernel sizes of the max-pool layers in the network; set as a command line argument; | a list of four integers; default is `[3, 3, 3, 3]` as in the original Deep4Net; for the purposes of this thesis also `[1, 1, 1, 1]` and `[2, 2, 2, 2]` |

The bash scripts which can be used to run the above mentioned training scripts are following:

* `training.py` - `run_in_ch0_k1_d1.sh`, `run_in_ch0_k1_d2.sh`, `run_in_ch0_k1_d3.sh`, 
  `run_in_ch0_k2_d1.sh`,  `run_in_ch0_k2_d2.sh`,  `run_in_ch0_k2_d3.sh`,
  `run_in_ch0_k3_d1.sh`,  `run_in_ch0_k3_d2.sh`,  `run_in_ch0_k3_d3.sh`,
  `run_in_ch1_k1_d1.sh`, `run_in_ch1_k1_d2.sh`, `run_in_ch1_k1_d3.sh`, 
  `run_in_ch1_k2_d1.sh`,  `run_in_ch1_k2_d2.sh`,  `run_in_ch1_k2_d3.sh`,
  `run_in_ch1_k3_d1.sh`,  `run_in_ch1_k3_d2.sh`,  `run_in_ch1_k3_d3.sh`
  
* `shifted_training.py` - `run_in_ch0_k1_shifted.sh`, `run_in_ch0_k2_shifted.sh`, `run_in_ch0_k3_shifted.sh`,
`run_in_ch1_k1_shifted.sh`, `run_in_ch1_k2_shifted.sh`, `run_in_ch1_k3_shifted.sh`
  
* `high_pass_training.py` - `run_in_ch0_k1_hp.sh`, `run_in_ch0_k2_hp.sh`, `run_in_ch0_k3_hp.sh`,
`run_in_ch1_k1_hp.sh`, `run_in_ch1_k2_hp.sh`, `run_in_ch1_k3_hp.sh`
  
* `pre_whitened_training.py` - `run_in_ch0_k1_w.sh`, `run_in_ch0_k2_w.sh`, `run_in_ch0_k3_w.sh`,
`run_in_ch1_k1_w.sh`, `run_in_ch1_k2_w.sh`, `run_in_ch1_k3_w.sh`
  

#### Training settings

A table displaying values of `model_string` together with the setting they describe:

| param: `model_string` | param: `shift` | param: `high_pass` |param: `high_pass_valid` | param: `low_pass` | param: `low_pass_training` | setting description |
| -------------- | ------- | ----------- | ----------------- | ---------- | ------------------- | ---------- |
| *m_* | `False` | `False` | `False` | `False` | `False` | original setting (full training and validation, non-shifted) as presented in Hammer et al. |
| *sm_* | `True` | `False` | `False` | `False` | `False` | shifted setting, full training and validation | 
| *hp_m_*| `False` | `True` | `False` | `False` | `False` | non-shifted setting, high-pass training and validation |
| *hp_sm_*| `True` | `False` | `False` | `False` | `False` | shifted setting, high-pass training and validation |
| *hpv_m_* | `False` | `False` | `True` | `False` | `False` | non-shifted setting, full training and high-pass validation |
| *hpv_sm_*| `True` | `False` | `True` | `False` | `False` | shifted setting, full training and high-pass validation
| *lp_m_* | `False` | `False` | `False` | `True` | `False` | non-shifted setting, full training and low-pass validation
| *lp_sm_* | `True` | `False` | `False` | `True` | `False` | shifted setting, full training and low-pass validation
| *lpt_hpv_m* | `False` | `False` | `True` | `False` | `True` | non-shifted setting, low-pass training and high-pass validation
| *lpt_hpv_sm* | `True` | `False` | `True` | `False` | `True` | shifted setting, low-pass training and high-pass validation

A prefix with `pw_` can be added to each of the `model_string`s. If `pw_` is present, the data was whitened, 
if not, the data was not whitened.

#### Training pipeline

1. First we check if the same setting (based on the `model_string` and `model_name`) was not already trained.
If we find results for computation in the same setting, we either continue the computation starting from the first 
   un-computed patient or if the previous computation finished we skip this setting.
2. For each un-computed patient, we perform a `num_of_folds` cross-validation.
 We save the validation results after computations for each patient are finished to avoid re-computing
   when process is canceled in the `gpulab`.
   

### Visualization scripts

The `gradient_inspection.py` script is used to calculate the gradients of the layers of the different architectures.
Similarly to the training scripts, it has multiple parameters most of which specify for which architecture 
and setting the gradients will be calculated.

| Parameter | Description | 
| --------- | ----------- |
| `file` | `model_name` from training scripts, specifies the kernel sizes and dilations of the network; here it is used to put toghether the path from which the inspected model should be loaded | 
| `prefix` | `model_string` from training scripts, specified in which setting the model was trained; here it is used to put toghether the path from which the inspected model should be loaded |

The data on which the gradients are calculated is always processed in the same way as the model was trained.
Therefore, we also set the parameters which are present in training accordingly.

The `gradient_inspection.py` script can be run using multiple bash scripts with different parameters:
`plot_all_layer_pw_grads0_k1.sh`, `plot_all_layer_pw_grads0_k1_hp.sh`,
`plot_all_layer_pw_grads0_k2.sh`, `plot_all_layer_pw_grads0_k2_hp.sh`,
`plot_all_layer_pw_grads0_k3.sh`, `plot_all_layer_pw_grads0_k3_hp.sh`,
`plot_all_layer_pw_grads1_k1.sh`, `plot_all_layer_pw_grads1_k1_hp.sh`,
`plot_all_layer_pw_grads1_k2.sh`, `plot_all_layer_pw_grads1_k2_hp.sh`,
`plot_all_layer_pw_grads1_k3.sh`, `plot_all_layer_pw_grads1_k3_hp.sh`.

#### Gradient visualization

After calculating the gradients we also have a runnable script - `gradient_heatmap.py` - to visualize them. 
The gradient heatmap script can be executed direcly or through `pw_gradient_heatmaps.sh`.

## Project description

Here we describe the functionalities contained in the folders 
of this project and the standalone scripts in the root of the project.


### BCICIV_competition
This folder contains modules which were used to get familiar with the `Braindecode` library.
None of the contents was used in the experiments. 

### data
This folder contains modules used to handle the dataset (i.e. data filtering, whitening and shifting).

The `data_fold_handling.py` module can be used to split the data into folds and save the indices so that
the same folds are created across the settings (filtered, shifted, whitened, ...).

The `OnePredictionData.py` is a module which is being prepared for experiments outside the scope of the thesis.

The `pre_processing.py` is the most crucial module for data handling. It contains the `Data` object 
which is designed to allow data loading, filtering, whitening, shifting, splitting into cross-validation
folds etc.

### Interpretation
This folder contains modules with methods used to calculate and visualize the gradients
and also perform other network analysis tasks.

Most notably the `manual_manipulation.py` contains methods which are used in the `gradient_inspection.py` script
for calculating and visualizing gradients.

The `perturbation.py` module contains code provided by Mgr. Jiří Hammer, Ph.D.
from his perturbation analysis and was not used for the analyses in the thesis.

The `single_maxpools.py` module was used to pass the signal through single max-pool layers.

### models
The `models` folder contains the `DoubleModel.py` module which is being prepared for 
a new experiment which is not a part of the thesis. 
It also contains the `Model.py` module where the `Model` object is defined. 
This object is used to transform the Dee4Net to dense prediction.
The `Model.py` script also contains methods used to change the parameters of the max-pool layers 
in the Deep4Net.

Untrained and trained models are also currently being saved here in a sub-folder called `saved_models`.

### outputs
This folder contains only two scripts which are suitable for reading `slurm` files.
These files are the output of the process running in the `gpulab`. It is possible to extract
the training results from them, however, it is not necessary as they are also saved in a pandas.DataFrame.
These DataFrames with results are also saved in the `outputs` folder.
Specifically in its sub-folder named `performances_{num_of_folds}` depending on the `num_of_folds` parameter.

The gradient files are also saved here into a corresponding folder whose name is specified in the `gradient_inspection.py`
script. 

### results

This folder is only used to store the results of the experiments. Most notably it contains graphs
where the performance box-plots are visualized.

### Training

The `Training` folder contains three modules. 
The `CorrelationMonitor1D.py` which was taken from the 
old version of the `Braindecode library` (0.4.85) and modified to be compatible with the current
version (0.5). It contains the `CorrelationMonitor1D` class which is a callback
used during training to monitor the correlations on the training and validation sets.

The `CropsFromTrialsIterator.py` was also taken from the 
old version of the `Braindecode library` (0.4.85) and modified to be compatible with the current
version (0.5). It contains the `CropsFromTrialsIterator` class which is used to
cut the network's input into crops for cropped decoding.

Lastly it contains the `train.py` module where methods used for training the networks are defined.

### visualization

The visualization folder contains modules with methods used for various visualization tasks which 
were useful to see the results of the experiments. 
The `comparison_plot.py` is a script where the results of the different performances can be visualized.
The settings which are to be compared can be specified by `model_string` in this script called `prefixes`.

The `distance_performance_corr.py` can be used to visualize the dependence of the performance on the
size of the receptive field.

The `fold_result_visualization.py` script similarly to the comparison plot allows you to plot performances 
of the different networks under various settings. 
It is currently suitable for visualizing the difference between absolute velocity decoding and decoding absolute velocity 
when taking absolute value of velocity.

The `full_hp_prediction_relationship.py` was meant to be used to combine the predictions of two networks trained
in a different setting but is not part of the thesis.

The `performance_visualization.py` is suited to plot only single boxplots to
see preliminary results.



