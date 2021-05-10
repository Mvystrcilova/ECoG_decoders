import pandas
import matplotlib.pyplot as plt
from global_config import home
import numpy


def get_df_correlations(df):
    corr_values = []
    for column in df.columns:
        list = df[column].tolist()
        cleaned_list = [x for x in list if str(x) != 'nan']
        corr_values += cleaned_list
    return corr_values


def create_df_from_files(files: list, mean=False):
    df = pandas.DataFrame()
    correlation_list = []
    for file in files:
        file_df = pandas.read_csv(f'{file}/performances.csv', sep=';', index_col=0)
        # file_df = file_df.T.drop_duplicates().T
        print(file_df.shape)
        # means = file_df.mean().tolist()
        correlations = get_df_correlations(file_df)
        # correlations = means
        if mean:
            means = file_df.mean().tolist()
            correlation_list.append(means)
            correlations = means
        else:
            correlation_list.append(correlations)
        # if file_df.shape[1] < 12:
        new_df = pandas.DataFrame()
        new_df[file.split('/')[10]] = correlations
        df = pandas.concat([df, new_df], axis=1)
        # else:
        #     df[file.split('/')[11]] = correlations
    return df, correlation_list


def get_5_fold_performance_df(variable, prefixes=None):
    big_df = pandas.DataFrame()
    if prefixes is None:
        prefixes = ['m_', 'lp_m_', 'hp_m_', 'hpv_m_', 'lpt_hpv_m_']
    # prefixes = [f'pw_{prefix}'for prefix in prefixes]
    file_names = [[f'{home}/outputs/performances_5/{prefix}{variable}_k1_d3',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k2_d3',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k3_d3',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k2_d1',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k3_d1',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k2_d2',
                   f'{home}/outputs/performances_5/{prefix}{variable}_k3_d2'] for prefix in prefixes]

    for i, file_sets in enumerate(file_names):
        df, cols = create_df_from_files(file_sets, mean=True)
        big_df = pandas.concat([big_df, df], axis=1)
    big_df = big_df.loc[:,~big_df.columns.duplicated()]
    return big_df


if __name__ == '__main__':
    variable = 'absVel'
    prefixes = ['m_', 'lp_m_', 'hp_m_', 'hpv_m_', 'lpt_hpv_m_']
    file_names = [[f'{home}/outputs/performances_5/{prefix}{variable}_k1_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k3_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d1',
                 f'{home}/outputs/performances_5/{prefix}{variable}_k3_d1', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d2', f'{home}/outputs/performances_5/{prefix}{variable}_k3_d3'] for prefix in prefixes]

    for i, file_sets in enumerate(file_names):
        df, cols = create_df_from_files(file_sets)

        plt.boxplot(cols, showfliers=False)
        plt.xticks(numpy.arange(1, df.shape[1]+1), labels=df.columns, rotation=90)
        plt.tight_layout()
        plt.savefig(f'{home}/results/performances_5/{prefixes[i]}{variable}_fold_perfromance.png')
        plt.show()
