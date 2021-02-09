import pandas

from global_config import home


def get_results(file, starting_patient):
    f = open(file, 'r')
    lines = f.readlines()
    patients = {str(x):[] for x in range(starting_patient, 13)}
    for line in lines:
        words = line.split(' ')
        if (len(words) == 2) and (words[0] in list(patients.keys())):
            patients[words[0]].append(float(words[1][:-1]))
    return patients


if __name__ == '__main__':
    patients = get_results(f'{home}/outputs/performance/slurm-7401.out', 9)
    print(patients)
    df = pandas.read_csv(f'{home}/outputs/performance/m_vel_k_3333/vel_performance.csv', sep=';', index_col=0)
    for patient, values in patients.items():
        patient_df = pandas.DataFrame()
        patient_df[patient] = values
        df = pandas.concat([df, patient_df], ignore_index=True, axis=1)
    print(df.shape)
    df.to_csv(f'{home}/outputs/performance/m_vel_k_3333/vel_performance.csv', sep=';')