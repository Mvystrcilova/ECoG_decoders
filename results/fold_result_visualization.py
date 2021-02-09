import pandas
import matplotlib.pyplot as plt
from global_config import home
import numpy
from results.performance_visualization import plot_df_boxplot


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
        file_df = pandas.read_csv(file, sep=';', index_col=0)
        file_df = file_df.T.drop_duplicates().T
        print(file_df.shape)
        means = file_df.mean().tolist()
        correlations = get_df_correlations(file_df)
        if mean:
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


if __name__ == '__main__':
    variable = 'absVel'
    file_names = [f'm_{variable}_k_1111/{variable}_performance.csv', f'm_{variable}_k_2222/{variable}_performance.csv',
                  f'm_{variable}_k_3333/{variable}_performance.csv', f'm_{variable}_k_3333_dilations_392781/{variable}_performance.csv',
                  f'm_{variable}_k_1111_dilations_1111/{variable}_performance.csv',
                  f'm_{variable}_k_2222_dilations_1111/{variable}_performance.csv', f'm_{variable}_k_3333_dilations_1111/{variable}_performance.csv',
                  f'm_{variable}_k_1111_dilations_24816/{variable}_performance.csv', f'm_{variable}_k_2222_dilations_24816/{variable}_performance.csv',
                  f'm_{variable}_k_3333_dilations_24816/{variable}_performance.csv']

    files = [f'{home}/outputs/performance/{file_name}' for file_name in file_names]
    df, cols = create_df_from_files(files, False)
    plt.boxplot(cols, showfliers=False)
    plt.xticks(numpy.arange(1, df.shape[1]+1), labels=df.columns, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{home}/results/{variable}_fold_perfromance.png')
    plt.show()
