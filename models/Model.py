from braindecode.models.deep4 import Deep4Net
from torch import nn, optim
from braindecode.torch_ext.modules import Expression
from braindecode.models.util import to_dense_prediction_model
from torch.nn import functional

def squeeze_out(x):
    assert x.size()[1] == 1 and x.size()[3] == 1
    return x[:, 0, :, 0]


class Model:
    def __init__(self, input_channels, n_classes, input_time_length, final_conv_length, stride_before_pool):
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.input_time_length = input_time_length
        self.final_conv_lenght = final_conv_length
        self.model = Deep4Net(in_chans=self.input_channels, n_classes=self.n_classes,
                              input_time_length=self.input_time_length, final_conv_length=self.final_conv_lenght,
                              stride_before_pool=stride_before_pool).create_network()
        self.regressed = False
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = functional.mse_loss

    def make_regressor(self):
        if not self.regressed:
            new_model = nn.Sequential()
            for name, module in self.model.named_children():
                if name == 'softmax':
                    break
                new_model.add_module(name, module)
            new_model.add_module('squeeze', Expression(squeeze_out))
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










