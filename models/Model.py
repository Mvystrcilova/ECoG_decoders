from braindecode.models.deep4 import Deep4Net
from braindecode.training.losses import CroppedLoss
from braindecode.util import np_to_var
from torch import nn, optim
import numpy as np
from braindecode.models.util import to_dense_prediction_model, get_output_shape
from torch.nn import functional
import logging, sys, torch
from global_config import home
from torchviz import make_dot

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
    log.info('Double done')
    return model


def add_padding(model, input_channels):
    """
    intended to add padding to the model, not a part of the thesis.
    :param model:
    :param input_channels:
    :return:
    """
    new_model = nn.Sequential()
    i = 0
    last_out = None
    for name, module in model.named_children():
        if hasattr(module, "dilation") and hasattr(module, 'kernel_size') and ('spat' not in name):
            dilation = module.dilation
            kernel_size = module.kernel_size
            right_padding = 0, 0, 0, (kernel_size[0] - 1) * dilation[0]
            new_model.add_module(name=f'{name}_pad', module=nn.ZeroPad2d(padding=right_padding))

            module.stride = (2, 1)
            new_model.add_module(name, module)
        else:
            new_model.add_module(name, module)
    n_preds_per_input = get_output_shape(new_model, input_channels, 1000)[1]
    new_model.add_module(name='last', module=nn.Linear(n_preds_per_input, 1))
    summary(new_model.cuda(device='cuda'), (85, 1000))
    print(new_model)
    summary(model, (85, 1200, 1))

    return new_model


def create_new_model(model, module_name, input_channels=None):
    new_model = nn.Sequential()
    found_selected = False
    for name, child in model.named_children():
        if name == 'conv_spat':
            if input_channels is not None:
                child.kernel_size = (1, input_channels)

        new_model.add_module(name, child)
        if name == module_name:
            found_selected = True
            break
    assert found_selected
    print(model)
    # test_input = np_to_var(
    #     np.ones((2, 85, 1200, 1), dtype=np.float32))
    # test_out = model(test_input)
    # make_dot(test_out, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    # summary(model, (85, 1200, 1))
    # print(new_model)
    return new_model


def test_padding():
    m = nn.ZeroPad2d((0, 0, 0, 10))
    input = torch.randn(1, 1, 20, 1)
    print(input.shape)
    print(input)
    out = m(input)
    conv_layer = nn.Conv2d(1, 25, kernel_size=(3, 1), )
    print(out)
    print(out.shape)


def change_network_kernel_and_dilation(model, kernel_sizes=None, dilations=None, remove_maxpool=False):
    """
    changes the dilation and kernel sizes of the given model based on the given kernel sizes and dilations

    :return:
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 3, 3]

    new_model = nn.Sequential()
    strides = [1, 1, 1, 1]
    i = 0
    for name, child in model.named_children():
        # print(name)
        add = True
        if ('pool' in name) and (len(name) <= 8):
            add = False
            if dilations is None:
                if not remove_maxpool:
                    new_model.add_module(f'pool_{i}',
                                         nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1),
                                                      padding=child.padding, dilation=child.dilation,
                                                      ceil_mode=child.ceil_mode))
            else:
                new_model.add_module(f'pool_{i}', nn.MaxPool2d(kernel_size=(kernel_sizes[i], 1), stride=(strides[i], 1),
                                                               padding=child.padding, dilation=(dilations[i], 1),
                                                               ceil_mode=child.ceil_mode))
            print('previous', name, child.stride, child.kernel_size)
            print('now', name, strides[i], kernel_sizes[i])
            i += 1
        if add:
            new_model.add_module(name, child)

    return new_model


def squeeze_out(x):
    assert x.size()[1] == 1 and x.size()[3] == 1
    return x[:, 0, :, 0]


# From braindecode library 0.4.85
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
                              input_window_samples=1200,
                              final_conv_length=self.final_conv_lenght,
                              stride_before_pool=stride_before_pool).train()
        # summary(self.model, (85, 1200, 1))
        # test_input = np_to_var(
        #     np.ones((2, 85, 1200, 1), dtype=np.float32))
        # test_out = self.model(test_input)
        # make_dot(test_out, params=dict(list(self.model.named_parameters()))).render("rnn_torchviz", format="png")
        # print(self.model)
        self.regressed = False
        self.optimizer = optim.Adam
        self.loss_function = torch.nn.MSELoss

    def make_regressor(self):
        """
        Transforms the model from a network suited for classification into a network suited
        for regression by removing the last softmax layer
        :return:
        """
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
            # uncomment section below if interested in the regressed model details

            test_input = np_to_var(
                np.ones((2, 85, 1200, 1), dtype=np.float32))
            test_out = self.model(test_input)
            make_dot(test_out, params=dict(list(self.model.named_parameters()))).render("rnn_torchviz_regressed", format="png")
            summary(self.model, input_size=(self.input_channels, self.input_time_length, 1))
            summary(self.new_model, input_size=(self.input_channels, self.input_time_length, 1))


if __name__ == '__main__':
    """
    Toy examples of testing the Model class.
    """
    model = Model(85, n_classes=1, input_time_length=1200, final_conv_length=2, stride_before_pool=False)
    print(model.model)
    model.make_regressor()
    print(model.model)
    model_2 = Model(85, n_classes=1, input_time_length=1000, final_conv_length=2, stride_before_pool=False)
    print(model_2.model)
    model_2.make_regressor()
    print(model_2.model)

    test_padding()
    add_padding(model.model, 85)
    print('done')
