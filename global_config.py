import torch, os

cuda = torch.cuda.is_available()
home = os.path.dirname(os.path.abspath(__file__))
input_time_length = 1200
n_perturbations = 10
output_dir = home + '/outputs'
srate = 250
random_seed = 8
interpreted_model_name = 'strides_3333'
eval_mode = 'train'
trained_mode = 'untrained'


def get_model_name_from_kernel_and_dilation(kernel_size, dilation):
    if kernel_size == [1, 1, 1, 1]:
        model_name = 'k1'
    elif kernel_size == [2, 2, 2, 2]:
        model_name = 'k2'
    elif kernel_size == [3, 3, 3, 3]:
        model_name = 'k3'
    elif kernel_size == [4, 4, 4, 4]:
        model_name = 'k4'

    if dilation is not None:
        if dilation == [1, 1, 1, 1]:
            dilations_name = '1'
        elif dilation == [2, 4, 8, 16]:
            dilations_name = '2'
        else:
            dilations_name = '3'
        # dilations_name = ''.join(str(x) for x in dilation)
    else:
        dilations_name = '3'

    model_name = f'{model_name}_d{dilations_name}'

    return model_name