from braindecode.models.deep4 import Deep4Net
from braindecode.training.losses import CroppedLoss
from torch import nn, optim
from braindecode.models.util import to_dense_prediction_model
from torch.nn import functional
import logging, sys, torch
from global_config import home
log = logging.getLogger()
log.setLevel('DEBUG')
from torchsummary import summary

import sys

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


def change_network_stride(model, kernel_sizes=None, dilations=None, remove_maxpool=False, change_conv_layers=False, conv_dilations=None):
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]

    new_model = nn.Sequential()
    strides = [1, 1, 1, 1]
    i = 0
    j = 0
    for name, child in model.named_children():
        # print(name)
        add = True
        if ('pool' in name) and (len(name) <= 8):
            add = False
            if dilations is None:
                if not remove_maxpool:
                    new_model.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1), padding=child.padding, dilation=child.dilation, ceil_mode=child.ceil_mode))
            else:
                new_model.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1), padding=child.padding, dilation=(dilations[i], 1), ceil_mode=child.ceil_mode))
            print('previous', name, child.stride, child.kernel_size)
            print('now', name, strides[i], kernel_sizes[i])
            i += 1


        if change_conv_layers:
            if (('conv' in name) and (len(name) < 8)) or ('conv_classifier' == name):
                add = False
                if conv_dilations is None:
                    new_model.add_module(f'conv_{j}',
                                         nn.Conv2d(in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=child.kernel_size, stride=(strides[j], 1),
                                                      padding=child.padding, dilation=child.dilation))
                else:
                    new_model.add_module(f'conv_{j}',
                                         nn.Conv2d(in_channels=child.in_channels, out_channels=child.out_channels,
                                                   kernel_size=child.kernel_size, stride=(strides[j], 1),
                                                   padding=child.padding, dilation=(conv_dilations[j], 1)))
                print('previous', name, child.stride, child.kernel_size, child.dilation)
                print('now', name, strides[j], child.kernel_size, conv_dilations[j])
                j += 1

        if add:
            new_model.add_module(name, child)

    return new_model


def squeeze_out(x):
    assert x.size()[1] == 1 and x.size()[3] == 1
    return x[:, 0, :, 0]


def add_layer_to_graph(name, layer):
    pass


### From braindecode library 0.4.85
class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)


class Model:
    def __init__(self, input_channels, n_classes, input_time_length, final_conv_length, stride_before_pool, cropped=True):
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.final_conv_lenght = final_conv_length
        self.model = Deep4Net(in_chans=self.input_channels, n_classes=self.n_classes,
                              input_window_samples=1000,
                              final_conv_length=self.final_conv_lenght,
                              stride_before_pool=stride_before_pool).train()
        # print(self.model)
        # summary(self.model, input_size=(input_channels, input_time_length, 1))
        self.regressed = False
        self.optimizer = optim.Adam
        self.cropped = cropped
        self.loss_function = torch.nn.MSELoss

    def make_regressor(self):
        if not self.regressed:
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                # print(name)
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



