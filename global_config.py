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
