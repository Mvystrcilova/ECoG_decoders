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


def read_model_shiftby_performances(lines):
    performances = {}
    patients = [str(x) for x in range(13)]
    # current_model = ''
    # shifts = [-100, -75, -50, -25, 25, 50, 100, 125, 150, 175, 200, 225, 250]
    for i, line in enumerate(lines):
        words = line.split(' ')
        if (len(words) == 10) and (words[0] == 'starting'):
            if 'vel' in words[-1][:-1]:
                variable = 'vel'
            else:
                variable = 'absVel'
        if (len(words) == 2) and (words[0] in patients):
            shift_words = lines[i+1].replace(':', '').split(' ')
            assert shift_words[0] == 'shift'
            col_name = f'{variable}_' + '_'.join(shift_words)[:-1]
            if col_name not in performances.keys():
                performances[col_name] = [float(words[1][:-1])]
            else:
                performances[col_name].append(float(words[1][:-1]))
    return performances




if __name__ == '__main__':
    # high-passed normally
    # numbers = [8534, 8533, 8521, 10152, 8888,  8535, 8530, 8529, 10153, 8889]

    # high-pass and shifted
    # numbers = [8758, 8754, 8753, 10156, 8750, 8757, 8756, 8752, 10157, 8751]

    # shifted correctly
    # numbers = [8685, 8688, 8689, 10146, 8692, 8686, 8687, 8690, 10147, 8691]

    # normal settings lr 0.01
    # numbers = [8855, 8856, 8857, 8859, 8862, 8861, 8860, 8858]

    # normal_settings lr 0.001
    # numbers = [8863, 8864, 8865, 10150, 8866, 8870, 8869, 8868, 8867, 10151]

    # low-pass
    # numbers = [8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994]
    # low-pass sbp0
    numbers = [10186, 10187]

    # shifted low-pass
    # numbers = [8999, 9000, 9001, 9002, 9003, 9004, 9005, 9006]
    # shifted low-pass sbp0
    # numbers = [10184, 10185]

    # high pass performance on full data training
    # numbers = [9152, 9153, 9154, 9156, 9187, 9188, 9159]
    # high pass valid performance on full data training sbp0
    # numbers = [10182, 10183]

    # shifted high pass valid performance on full data training
    # numbers = [9160, 9161, 9162, 9163, 9165, 9166, 9167, 9164]
    # shifted high pass valid performance on full data training sbp0
    # numbers = [10178, 10179]

    # stride before pool False high pass validation on lp training data
    # numbers = [10018, 10019]

    # shifted stride before pool False high pass validation on lp training data
    # numbers = [10020, 10021]

    # high pass validation on lp training data
    # numbers = [10096, 10097, 10098, 10110, 10100, 10099, 10101, 10111]

    # shifted high pass validation on lp training data
    # numbers = [10102, 10103, 10104, 10109, 10105, 10106, 10107, 10108]

    # shifted performances
    numbers = [11434, 11433, 11432, 11431, 10204, 10205, 10206, 10207, 11466, 11467, 10208, 10209, 10210, 10211, 10212, 10213]

    # high-passed shifted performaces
    numbers = [11501, 11502, 11503, 11504, 11505, 11517, 11518, 11507, 11508, 11509, 11510]
    files = [f'{home}/outputs/performance/slurm-{number}.out' for number in numbers]
    performances = {}
    for file in files:
        f = open(file, 'r')
        lines = f.readlines()
        # file_performance = read_model_performances(lines)
        file_performance = read_model_shiftby_performances(lines)
        if 'vel' in list(file_performance.keys())[0]:
            print(file_performance.keys())
            print(file)
        performances = {**performances, **file_performance}
    df = pandas.DataFrame(performances)
    print(df.shape)
    df.to_csv(f'{home}/results/shifted_hp_window_performances.csv', sep=';')
    print(performances)