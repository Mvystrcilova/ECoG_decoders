import pandas
import matplotlib.pyplot as plt
from global_config import home
import numpy
import seaborn as sns
from scipy.stats import binom_test, wilcoxon, ttest_rel


def get_df_correlations(df):
    corr_values = []
    for column in df.columns:
        list = df[column].tolist()
        cleaned_list = [x for x in list if str(x) != 'nan']
        corr_values += cleaned_list
    return corr_values


def get_statistical_significance(random_correlations, correlations, alternative='greater'):
    """
    calculates if the difference between random correlations and correlations is significant
    :param random_correlations: values of the first variable
    :param correlations: values of the second variable
    :param alternative: which sided test to perform
    :return: returns a string representing the wether the difference is significant
    """
    # success = [1 if corr > random_corr else 0 for corr, random_corr in zip(correlations, random_correlations)]
    # p = binom_test(x=sum(success), n=len(success), p=0.05, alternative='greater')
    # value, p = ttest_rel(correlations, random_correlations, )
    value, p = wilcoxon(correlations, random_correlations, correction=True, alternative=alternative)
    string = ''
    # if alternative == 'greater':
    #     if (p / 2 < 0.05) and (value > 0):
    #         string = ' *'
    #     if (p / 2 < 0.01) and (value > 0):
    #         string = ' **'
    # else:
    #     if (p / 2 < 0.05) and (value < 0):
    #         string = ' *'
    #     if (p / 2 < 0.01) and (value < 0):
    #         string = ' **'

    if p / 2 < 0.05:
        string = ' *'
    if p / 2 < 0.01:
        string = ' **'
    return string


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
    """
    Based on the specified prefixes loads the DataFrames in which the results for the models are saved.
    It concatenates the DataFrames for each of the specified prefix into one large DataFrame.
    The prefixes are concatenated with model names for all the created architectural modifications
    (i.e. kernel sizes 1, 2, 3 and dilations powers of 1, 2 and 3)
    :param variable: The variable for which comparison is done ('vel' for velocity or 'absVel' for absolute velocity)
    :param prefixes: The prefixes specifying the setting which we want to compare.
    The prefixes for the different settings are described in README.md
    :return: A DataFrame with all results for the specified prefixes for all architectural modifications
    """
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
    # prefixes = ['m_', 'lp_m_', 'hp_m_', 'hpv_m_', 'lpt_hpv_m_']
    prefixes = ['m_', 'abs_m_', 'sm_', 'abs_sm_']
    titles = ['Absolute velocity', 'Absolute value of velocity',
              'Shifted - absolute velocity', 'Shifted - absolute value\nof velocity']
    file_names = [[f'{home}/outputs/performances_5/{prefix}{variable}_k1_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k3_d3', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d1',
                 f'{home}/outputs/performances_5/{prefix}{variable}_k3_d1', f'{home}/outputs/performances_5/{prefix}{variable}_k2_d2', f'{home}/outputs/performances_5/{prefix}{variable}_k3_d3'] for prefix in prefixes]
    abs_vel = get_5_fold_performance_df('vel', ['abs_m_', 'abs_sm_'])
    absVel = get_5_fold_performance_df('absVel', ['m_', 'sm_'])
    df = pandas.DataFrame()
    for col in absVel.columns:
        df[col] = absVel[col]
    for col in abs_vel.columns:
        df[col] = abs_vel[col]

    fig, ax = plt.subplots(1, 4, sharey='row', figsize=(27, 12))
    indices = [i for i in range(len(prefixes))]
    letters = ['A', 'B', 'C', 'D']
    for i in range(len(prefixes)):
        print(prefixes[i])
        sub_df = df[[column for column in df.columns if
                    (column.startswith(prefixes[i])) and ('k4' not in column) and (
                             'sbp1' not in column)]]
        col_names = []
        for col in sub_df.columns:
            values = sub_df[col].tolist()
            col = col.replace(prefixes[i], '')
            significance_string = ''
            if prefixes[i].startswith('abs_'):
                absVel_col = col.replace('vel', 'absVel')
                chance_level_col = df[f'{prefixes[i-1]}{absVel_col}'].tolist()
                alternative = 'greater'
                significance_string = get_statistical_significance(values, chance_level_col, alternative)
            if 'k3_d3' in col:
                col = f'{col}_sbp0'
            col = f'{col}{significance_string}'
            if 'vel' in col:
                col = col.replace('vel', '|vel|')
            col_names.append(col)

        ax[indices[i]].axhline(0, color='k', linestyle='--')
        ax[indices[i]].text(-0, 1.0, letters[i],
                        size=30, weight='bold')
        sns.boxplot(data=sub_df, ax=ax[indices[i]])
        ax[indices[i]].set_title(titles[i % int(len(titles))], size=30, loc='center', pad=10)
        ax[indices[i]].tick_params('x', labelrotation=75, labelsize=30)
        ax[indices[i]].tick_params('y', labelsize=30)
        ax[indices[i]].set_xticklabels(labels=col_names)
        ax[indices[i]].get_xticklabels()[2].set_color("red")
        ax[indices[i]].grid(which='major', axis='y', linestyle='--')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.86)
    plt.savefig(f'{home}/results/performances_5_results/graphs/absVel_vs_abs_vel_performance_comparison.pdf')
    plt.show()
    # for i, file_sets in enumerate(file_names):
    #     df, cols = create_df_from_files(file_sets)
    #
    #     plt.boxplot(cols, showfliers=False)
    #     plt.xticks(numpy.arange(1, df.shape[1]+1), labels=df.columns, rotation=90)
    #     plt.tight_layout()
    #     plt.savefig(f'{home}/results/performances_5/{prefixes[i]}{variable}_fold_perfromance.png')
    #     plt.show()
