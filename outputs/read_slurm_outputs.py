from pathlib import Path

import pandas

from global_config import home


def get_results(file, starting_patient):
    f = open(file, 'r')
    lines = f.readlines()
    patients = {str(x): [] for x in range(starting_patient, 13)}
    for line in lines:
        words = line.split(' ')
        if (len(words) == 2) and (words[0] in list(patients.keys())):
            patients[words[0]].append(float(words[1][:-1]))
    return patients


def get_results_from_one_file(file_number):
    file = open(f'{home}/outputs/performance/slurm-{file_number}.out', 'r')
    lines = file.readlines()
    df = None
    df = pandas.DataFrame()

    for i, line in enumerate(lines):
        words = line.split(' ')
        if words[0] == 'starting':
            model_name = words[9]
            model_name = model_name.replace('k_k', 'k')
            model_name = model_name[:-1]

        if words[0] == 'whole_patient:':
            patient_index = words[1]
            performances = line.split('[')
            performances = performances[1]
            performances = performances.split(', ')
            performances[-1] = performances[-1][:-2]
            patient_df = pandas.DataFrame()
            patient_df[f'P_{patient_index}'] = performances
            df = pandas.concat([df, patient_df], axis=1)
            if patient_index == 12:
                df.to_csv(f'{home}/performances/{model_name}.csv', sep=';')
    Path(f'{home}/outputs/performances/{model_name}/').mkdir(exist_ok=True, parents=True)
    df.to_csv(f'{home}/outputs/performances/{model_name}/performances.csv', sep=';')



if __name__ == '__main__':
    # patients = get_results(f'{home}/outputs/performance/slurm-7401.out', 1)
    # print(patients)
    # df = pandas.read_csv(f'{home}/outputs/performance/m_vel_k_3333/vel_performance.csv', sep=';', index_col=0)
    # for patient, values in patients.items():
    #     patient_df = pandas.DataFrame()
    #     patient_df[patient] = values
    #     df = pandas.concat([df, patient_df], ignore_index=True, axis=1)
    # print(df.shape)
    # df.to_csv(f'{home}/outputs/performance/m_vel_k_3333/vel_performance.csv', sep=';')
    get_results_from_one_file(12436)