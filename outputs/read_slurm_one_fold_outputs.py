from global_config import home
import pandas


def read_model_performances(lines):
    performances = {}
    patients = [str(x) for x in range(13)]
    current_model = ''
    for line in lines:
        words = line.split(' ')
        if (len(words) == 10) and (words[0] == 'starting'):
            if words[-1][:-1] not in performances.keys():
                performances[words[-1][:-1]] = []
                current_model = words[-1][:-1]
        if (len(words) == 2) and (words[0] in patients):
            performances[current_model].append(float(words[1][:-1]))
    return performances


if __name__ == '__main__':
    # high-passed normally
    # numbers = [8534, 8533, 8521, 8888,  8535, 8530, 8529,  8889]

    # high-pass and shifted
    # numbers = [8758, 8754, 8753, 8750, 8757, 8756, 8752, 8751]

    # shifted correctly
    # numbers = [8685, 8688, 8689, 8692, 8686, 8687, 8690, 8691]

    # normal settings lr 0.01
    # numbers = [8855, 8856, 8857, 8859, 8862, 8861, 8860, 8858]

    # normal_settings lr 0.001
    # numbers = [8863, 8864, 8865, 8866, 8870, 8869, 8868, 8867]

    # low-pass
    # numbers = [8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994]

    # shifted low-pass
    # numbers = [8999, 9000, 9001, 9002, 9003, 9004, 9005, 9006]

    # high pass performance on full data training
    # numbers = [9152, 9153, 9154, 9156, 9187, 9188, 9159]

    # shifted high pass performance on full data training
    numbers = [9160, 9161, 9162, 9163, 9165, 9166, 9167, 9164]

    files = [f'{home}/outputs/performance/slurm-{number}.out' for number in numbers]
    performances = {}
    for file in files:
        f = open(file, 'r')
        lines = f.readlines()
        file_performance = read_model_performances(lines)
        performances = {**performances, **file_performance}
    df = pandas.DataFrame(performances)
    print(df.shape)
    df.to_csv(f'{home}/results/hp_shifted_valid_performance.csv', sep=';')
    print(performances)