import numpy as np
from skorch.callbacks import Callback


class CorrelationMonitor1D(Callback):
    """
    Compute correlation between 1d predictions

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None, setname=None):
        self.input_time_length = input_time_length
        self.setname = setname
        self.step_number = 0

    def monitor_batch(self, msg):
        print(msg)

    ### from braindecode library 0.4.85
    def compute_preds_per_trial_from_n_preds_per_trial(self,
                                                       all_preds, n_preds_per_trial):
        """
        Compute predictions per trial from predictions for crops.
        Parameters
        ----------
        all_preds: list of 2darrays (classes x time)
            All predictions for the crops.
        input_time_length: int
            Temporal length of one input to the model.
        n_preds_per_trial: list of int
            Number of predictions for each trial.
        Returns
        -------
        preds_per_trial: list of 2darrays (classes x time)
            Predictions for each trial, without overlapping predictions.
        """
        # all_preds_arr has shape forward_passes x classes x time
        # all_preds_arr = np.concatenate(all_preds[0].detach().numpy(), axis=0)
        all_preds_arr = all_preds[0]
        preds_per_trial = []
        i_pred_block = 0
        for i_trial in range(len(n_preds_per_trial)):
            n_needed_preds = n_preds_per_trial[i_trial]
            preds_this_trial = []
            while n_needed_preds > 0:
                # - n_needed_preds: only has an effect
                # in case there are more samples than we actually still need
                # in the block.
                # That can happen since final block of a trial can overlap
                # with block before so we can have some redundant preds.
                pred_samples = all_preds_arr[i_pred_block, :, -n_needed_preds:]
                preds_this_trial.append(pred_samples)
                n_needed_preds -= pred_samples.shape[1]
                i_pred_block += 1

            # preds_this_trial = np.concatenate(preds_this_trial, axis=1)
            preds_this_trial = preds_this_trial[0]
            preds_per_trial.append(preds_this_trial)
        assert i_pred_block == len(all_preds_arr), (
            "Expect that all prediction forward passes are needed, "
            "used {:d}, existing {:d}".format(i_pred_block, len(all_preds_arr))
        )
        return preds_per_trial

    ### from braindecode library 0.4.85
    def compute_preds_per_trial_from_crops(self, all_preds, input_time_length, X):
        n_preds_per_input = all_preds[0].shape[2]
        n_receptive_field = input_time_length - n_preds_per_input + 1
        n_preds_per_trial = [trial.shape[1] - n_receptive_field + 1 for trial in X]
        preds_per_trial = self.compute_preds_per_trial_from_n_preds_per_trial(
            all_preds, n_preds_per_trial
        )
        return preds_per_trial

    def on_epoch_end(self, net, **kwargs):
        writer = net.callbacks[2][1].writer
        train_X = kwargs['dataset_train'].X
        train_y = kwargs['dataset_train'].y
        valid_X = kwargs['dataset_valid'].X
        valid_y = kwargs['dataset_valid'].y

        train_preds = net.predict(train_X)
        valid_preds = net.predict(valid_X)

        train_corr = self.calculate_correlation(train_preds, train_y, train_X)
        valid_corr = self.calculate_correlation(valid_preds, valid_y, valid_X)
        names = ['train correlation', 'validation correlation']

        for name, value in zip(names, [train_corr, valid_corr]):
            writer.add_scalar(name, value, self.step_number)
            writer.flush()
        print(f'train correlation: {train_corr}')
        print(f'validation correlation: {valid_corr}')
        self.step_number += 1

    # def on_batch_end(self, net, **kwargs):
    #     """Assuming one hot encoding for now"""
    #     assert self.input_time_length is not None, "Need to know input time length..."
    #     # this will be timeseries of predictions
    #     # for each trial
    #     # braindecode functions expect classes x time predictions
    #     # so add fake class dimension and remove it again
    #     msg = self.calculate_correlation(kwargs['y_pred'], kwargs['y'], kwargs['X'])
    #     self.monitor_batch(msg)

    def calculate_correlation(self, predictions, targets, inputs):
        all_preds = [predictions]
        preds_2d = [p[:, None] for p in all_preds]
        preds_per_trial = self.compute_preds_per_trial_from_crops(preds_2d,
                                                                  self.input_time_length,
                                                                  inputs)
        if isinstance(preds_per_trial[0], np.ndarray):
            preds_per_trial = [p[0] for p in preds_per_trial]
        else:
            preds_per_trial = [p.detach().numpy()[0] for p in preds_per_trial]
        pred_timeseries = np.concatenate(preds_per_trial, axis=0)
        ys_2d = [y[:, None] for y in [targets]]
        targets_per_trial = self.compute_preds_per_trial_from_crops(ys_2d,
                                                                    self.input_time_length,
                                                                    inputs)
        if isinstance(targets_per_trial[0], np.ndarray):
            targets_per_trial = [t[0] for t in targets_per_trial]
        else:
            targets_per_trial = [t.detach().numpy()[0] for t in targets_per_trial]
        target_timeseries = np.concatenate(targets_per_trial, axis=0)

        corr = np.corrcoef(target_timeseries, pred_timeseries)[0, 1]
        key = self.setname + '_corr'
        msg = {key: float(corr)}
        return corr
