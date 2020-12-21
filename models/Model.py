from braindecode.models.deep4 import Deep4Net
from torch import nn, optim
from braindecode.models.util import to_dense_prediction_model
from torch.nn import functional
import logging, sys, torch
from global_config import home
log = logging.getLogger()
log.setLevel('DEBUG')
from torchsummary import summary

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)


def load_model(model_file):
    log.info("Loading CNN model...")
    if not torch.cuda.is_available():
        model = torch.load(home + model_file, map_location=torch.device('cpu'))
    else:
        model = torch.load(home + model_file)
    # fix for new pytorch
    for m in model.modules():
        if m.__class__.__name__ == 'Conv2d':
            m.padding_mode = 'zeros'
    log.info("Loading done.")
    model.double()
    return model


def create_new_model(model, module_name):
    new_model = nn.Sequential()
    found_selected = False
    for name, child in model.named_children():
        new_model.add_module(name, child)
        if name == module_name:
            found_selected = True
            break
    assert found_selected
    print(model)
    print(new_model)
    return new_model


def change_network_stride(model, kernel_sizes=None, dilations=None):
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]

    new_model = nn.Sequential()
    strides = [1, 1, 1, 1]
    i = 0
    for name, child in model.named_children():
        # print(name)
        if ('pool' in name) and (len(name) <= 8):
            print(child.kernel_size, child.stride)
            if dilations is None:
                new_model.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1), padding=child.padding, dilation=child.dilation, ceil_mode=child.ceil_mode))
            else:
                new_model.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1), padding=child.padding, dilation=(dilations[i], 1), ceil_mode=child.ceil_mode))

            print(name, strides[i], kernel_sizes[i])
            i += 1
        else:
            new_model.add_module(name, child)
    return new_model


def squeeze_out(x):
    assert x.size()[1] == 1 and x.size()[3] == 1
    return x[:, 0, :, 0]


### From braindecode library 0.4.85
class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Model:
    def __init__(self, input_channels, n_classes, input_time_length, final_conv_length, stride_before_pool):
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.final_conv_lenght = final_conv_length
        self.model = Deep4Net(in_chans=self.input_channels, n_classes=self.n_classes,
                              input_window_samples=input_time_length,
                              final_conv_length=self.final_conv_lenght,
                              stride_before_pool=stride_before_pool).train()
        self.regressed = False
        self.optimizer = optim.Adam
        self.loss_function = nn.MSELoss

    def make_regressor(self):
        if not self.regressed:
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                print(name)
                if 'softmax' in name:
                    break
                new_model.add_module(name, module)
            new_model.add_module('squeeze_out', Expression(squeeze_out))
            self.model = new_model
            to_dense_prediction_model(self.model)
            self.regressed = True

    def get_layer_activations(self, input, activations):
        for name, module in self.model.items():
            x = module(input)
            if name in activations.keys():
                activations[name].append(x)
            else:
                activations[name] = [x]
        return activations
