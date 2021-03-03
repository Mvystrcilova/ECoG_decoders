from braindecode.models.util import get_output_shape
from Interpretation.interpretation import get_corr_coef
from data.pre_processing import Data
from global_config import home, input_time_length
from models.Model import load_model
import pandas

from visualization.performance_visualization import plot_df_boxplot


def get_lowpass_corr(model, data):
    best_corr = get_corr_coef(data.test_set, model, cuda=False)
    lp_corr = get_corr_coef(data.low_pass_test, model, cuda=False)
    return best_corr, lp_corr


def get_correlation_for_model(data_file, model):
    data = Data(home + data_file, num_of_folds=-1, low_pass=False,
                trajectory_index=1)
    n_preds_per_input = get_output_shape(model, data.in_channels, input_time_length)[1]

    data.cut_input(input_time_length, n_preds_per_input, False)
    return get_lowpass_corr(model, data)


if __name__ == '__main__':
    # model_files = ['m_vel_k_3333_p_', 'm_vel_k_3333_dilations_1111_p_', 'm_vel_k_3333_dilations_24816_p_',
    #                'm_vel_k_2222_p_', 'm_vel_k_2222_dilations_1111_p_', 'm_vel_k_2222_dilations_24816_p_']
    model_files = ['m_absVel_k_3333_p_', 'm_absVel_k_3333_dilations_1111_p_', 'm_absVel_k_3333_dilations_24816_p_',
                   'm_absVel_k_2222_p_', 'm_absVel_k_2222_dilations_1111_p_', 'm_absVel_k_2222_dilations_24816_p_']
    big_df = pandas.DataFrame()
    for file in model_files:
        df = pandas.DataFrame()
        for i in range(1, 13):
            model = load_model(f'/models/saved_models/{file}{i}/best_model_split_0')
            best_corr, lp_corr = get_correlation_for_model(f'/previous_work/P{i}_data.mat', model)
            df = df.append({f'best_corr_{file}': best_corr, f'lp_corr_{file}': lp_corr}, ignore_index=True)
        big_df[f'best_corr_{file}'] = df[f'best_corr_{file}']
        big_df[f'lp_corr_{file}'] = df[f'lp_corr_{file}']
    big_df.to_csv(home + '/results/absVel_corr_lp_corr_df.csv', sep=';')
    plot_df_boxplot(big_df)
