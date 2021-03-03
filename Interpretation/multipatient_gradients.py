from Interpretation.interpretation import get_outs, plot_correlation, plot_gradients
from Interpretation.manual_manipulation import prepare_for_gradients
import numpy as np

from global_config import output_dir

model_prefix = 'm_'
variable = 'absVel'
model_names = [f'lr_0.001/{model_prefix}_{variable}_k_1111', f'lr_0.001/{model_prefix}_{variable}_k_2222',
               f'lr_0.001/{model_prefix}_{variable}_k_3333', f'lr_0.001/{model_prefix}_{variable}_k_4444',
               f'lr_0.001/{model_prefix}_{variable}_k_2222_dilations_1111',
               f'lr_0.001/{model_prefix}_{variable}_k_3333_dilations_1111',
               f'lr_0.001/{model_prefix}_{variable}_k_2222_dilations_24816',
               f'lr_0.001/{model_prefix}_{variable}_k_3333_dilations_24816',
               f'lr_0.001/{model_prefix}_{variable}_k_4444_dilations_1111',
               f'lr_0.001/{model_prefix}_{variable}_k_4444_dilations_24816']


if __name__ == '__main__':
    for model_name in model_names:
        full_windows, small_windows, corrcoefs = [], [], []
        last_batch = None
        output = f'{output_dir}/hp_graphs/avg_{model_name}/'
        for patient_index in range(1, 13):
            corrcoef, new_model, X_reshaped, small_window, output = prepare_for_gradients(patient_index,
                                                                                          model_name,
                                                                                          eval_mode='validation',
                                                                                          trained_mode='trained')
            full_batch_X = X_reshaped[:1]
            smaller_batch_X = X_reshaped[:1, :, :small_window]
            amp_grads, _ = get_outs(full_batch_X, new_model, None)
            full_windows.append(amp_grads)
            amp_grads, _ = get_outs(smaller_batch_X, new_model, None)
            small_windows.append(amp_grads)
            last_batch = full_batch_X, smaller_batch_X
        full_gradients = np.mean(np.asarray(full_windows))
        window_gradients = np.mean(np.asarray(small_windows))
        corrcoef = sum(corrcoefs)/len(corrcoefs)

        plot_gradients(last_batch[0], full_gradients, corrcoef, f'{output}')
        plot_gradients(last_batch[1], window_gradients, corrcoef, f'{output}', None, setname='Smaller window')

