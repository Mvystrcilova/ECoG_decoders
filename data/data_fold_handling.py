from data.pre_processing import Data
import numpy as np
import pickle


def set_valid_indices(num_of_folds):
    """
    This function splits the trials for each patient in the dataset into the specified number of folds.
    :param num_of_folds: number of fold into which the dataset is split
    :return: None, only saves the indices for the folds for each patient
    """
    patient_dict = {}
    for patient in range(1, 13):
        data = Data(f'../previous_work/P{patient}_data.mat', num_of_folds, trajectory_index=0, low_pass=False)
        indices = set_valid_indices_for_patient(data, num_of_folds)
        patient_dict[f'P_{patient}'] = indices
    with open(f'train_dict_{num_of_folds}', 'wb') as handle:
        pickle.dump(patient_dict, handle)
    handle.close()


def set_valid_indices_for_patient(data, num_of_folds):
    """
    Indices of trials of one patient are split into the folds. The indices are randomly permuted and then split
    into the folds to avoid sequentiality of data in the different folds.
    :param data: one patient data which is to be split into the train and validation set
    :param num_of_folds: number of fold into which the data is split
    :return: returns indices split into the number of folds
    """
    if num_of_folds == 0:
        num_of_folds = data.num_of_folds
    indices = np.array([x for x in range(0, data.num_of_folds)])
    indices = np.random.permutation(indices)
    indices = np.array_split(indices, num_of_folds)
    return indices


# set_valid_indices(5)