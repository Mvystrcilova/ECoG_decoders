from global_config import home
import pandas


def read_model_performances(lines):
    performances = {}
    patients = [str(x) for x in range(13)]
    current_model = ''
    for line in lines:
        words = line.split(' ')
        if (len(words) == 10) and (words[0] == 'starting'):
            if words[-1][:-1].split('/')[0] not in performances.keys():
                performances[words[-1][:-1].split('/')[0]] = []
                current_model = words[-1][:-1].split('/')[0]
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
            col_name = float(words[1][:-1])
        if (len(words) == 2) and (words[0] in patients):
            # shift_words = lines[i+1].replace(':', '').split(' ')
            # assert shift_words[0] == 'shift'
            # col_name = f'{variable}_' + '_'.join(shift_words)[:-1]
            # col_name = float(words[1][:-1])
            if col_name not in performances.keys():
                performances[col_name] = [float(words[1][:-1])]
            else:
                performances[col_name].append(float(words[1][:-1]))
    return performances


if __name__ == '__main__':
    # high-passed normally
    # numbers = [8534, 8533, 8521, 10152, 8888,  8535, 8530, 8529, 10153, 8889]
    # high-order filter
    numbers = [11804, 11805, 11806, 11757, 11807, 11808, 11809, 11756]
    # 3rd order filter
    # numbers = [11905, 11907, 11908, 11906, 11909, 11910]

    # high-pass and shifted
    # numbers = [8758, 8754, 8753, 10156, 8750, 8757, 8756, 8752, 10157, 8751]
    # high-order filter
    # numbers = [11777, 11779, 11780, 11770, 11778, 11781, 11782, 11771]
    # 3rd order filter
    # numbers = [11918, 11919, 11920, 11922, 11923, 11924]

    # shifted correctly
    # numbers = [8685, 8688, 8689, 10146, 8692, 8686, 8687, 8690, 10147, 8691]

    # normal settings lr 0.01
    # numbers = [8855, 8856, 8857, 8859, 8862, 8861, 8860, 8858]

    # normal_settings lr 0.001
    # numbers = [8863, 8864, 8865, 10150, 8866, 8870, 8869, 8868, 8867, 10151]
    # numbers = [11633, 11632, 11631, 11753, 11634, 11635, 11636, 11752]

    # low-pass
    # numbers = [8987, 8988, 8989, 8990, 8991, 8992, 8993, 8994]
    # low-pass sbp0
    # numbers = [10186, 10187]

    # shifted low-pass
    # numbers = [8999, 9000, 9001, 9002, 9003, 9004, 9005, 9006]
    # shifted low-pass sbp0
    # numbers = [10184, 10185]
    # test set
    # numbers = [12014, 12015, 12016, 12017, 12018, 12019]

    # high pass performance on full data training
    # numbers = [9152, 9153, 9154, 9156, 9187, 9188, 9159]
    # high pass valid performance on full data training sbp0
    # numbers = [10182, 10183]
    # high-order filter, test
    # numbers = [11789, 11790, 11791, 11820, 11792, 11793, 11794, 11819]
    # 3rd orde filter test
    # numbers = [11931, 11932, 11933, 11934, 11935, 11936]

    # shifted high pass valid performance on full data training
    # numbers = [9160, 9161, 9162, 9163, 9165, 9166, 9167, 9164]
    # shifted high pass valid performance on full data training sbp0
    # numbers = [10178, 10179]
    # high-order filter, test
    # numbers = [11783, 11784, 11785, 11773, 11786, 11787, 11788, 11772]
    # TODO: 3rd order filter, test
    # numbers = [11962, 11963, 11964, 11965, 11966, 11967]

    # stride before pool False high pass validation on lp training data
    # numbers = [10018, 10019]

    # shifted stride before pool False high pass validation on lp training data
    # numbers = [10020, 10021]

    # high pass validation on lp training data
    # numbers = [10096, 10097, 10098, 10110, 10100, 10099, 10101, 10111]
    # numbers = [11710, 11705, 11711, 11707, 11708, 11709]

    # high pass validation on lp training data order 15 filter
    # numbers = [11712, 11713, 11714, 11717, 11716, 11715]

    # high pass validation on lp training data order 15 filter shifted
    # numbers = [11746, 11747, 11748, 117, 11750, 11749]

    # shifted high pass validation on lp training data
    # numbers = [10102, 10103, 10104, 10109, 10105, 10106, 10107, 10108]
    # numbers = [11740, 11741, 11742, 11745, 11744, 11743]

    # shifted performances
    # numbers = [11434, 11433, 11432, 11431, 10204, 10205, 10206, 10207, 11466, 11467, 10208, 10209, 10210, 10211, 10212, 10213]
    #  numbers = [11640, 11641, 11642, 11754, 11639, 11638, 11637, 11755]

    # high-passed shifted performaces
    # numbers = [11501, 11502, 11504, 11505, 11517, 11518, 11507, 11503, 11508, 11509, 11510]

    # double model initial trials
    # numbers = [11530]

    # low-pass validation, full training
    # numbers = [11686, 11685, 11684, 11687, 11688, 11689]

    # random
    # numbers = [11811, 11812, 11813, 11814, 11815, 11816]

    # pw normal setting
    # numbers = [11872, 11871, 11870, 11873, 11874, 11875]

    # pw normal setting shifted
    # numbers = [11876, 11877, 11878, 11879, 11880, 11881]

    # pw high-pass both strong filter
    # numbers = [11883, 11884, 11885, 11886, 11887, 11888]

    # pw high-pass both strong filter shifted

    # pw high-pass valid full data train strong filter
    # pw high-pass valid full data train strong filter shifted
    # numbers = [11943, 11944, 11945, 11946, 11947, 11948]

    numbers = [12069, 12070, 12072, 12073]
    files = [f'{home}/outputs/performance/slurm-{number}.out' for number in numbers]
    performances = {}
    for file in files:
        f = open(file, 'r')
        lines = f.readlines()
        file_performance = read_model_performances(lines)
        # file_performance = read_model_shiftby_performances(lines)
        # if 'vel' in list(file_performance.keys())[0]:
        #     print(file_performance.keys())
        #     print(file)
        performances = {**performances, **file_performance}
    df = pandas.DataFrame(performances)
    print(df.shape)
    df.to_csv(f'{home}/results/test_results/lpv_wide_performances.csv', sep=';')
    print(performances)