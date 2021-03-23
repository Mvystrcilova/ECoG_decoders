from braindecode.util import np_to_var
from torch import nn, optim
import torch

from data.pre_processing import band_pass_data


class DoubleModel(nn.Module):
    def __init__(self, model_1, model_2):
        super(DoubleModel, self).__init__()
        self.model_1 = model_1.float()
        self.model_2 = model_2.float()
        for param in self.model_1.parameters():
            param.requires_grad = False
        for param in self.model_2.parameters():
            param.requires_grad = False
        # self.put_together0 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.dropout0 = nn.Dropout(0.5)
        self.put_together = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm2d(1)
        self.put_together2 = nn.Linear(1500, 1000)
        self.dropout2 = nn.Dropout(0.3)
        self.put_together3 = nn.Linear(1000, 679)
        self.optimizer = optim.Adam
        self.loss_function = torch.nn.MSELoss

    # def freeze_models(self):
    #     for param in self.param
    def forward(self, input):
        # length = int(input.size[0]/2)
        # x1 = self.model_1.double()(inputs[0].reshape([1, inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]]))
        # x2 = self.model_2.double()(inputs[1].reshape([1, inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]]))
        x1 = self.model_1(input[:, :, :, 0].float())
        # x2, _ = band_pass_data(input.double(), None, order=3, cut_off_frequency=60, btype='hp')
        # x2 = np_to_var(x2)
        # x2 = x2.double()
        x2 = self.model_2(input[:, :, :, 1].float())
        out = torch.stack((x1, x2), 0)
        out = torch.transpose(out, 0, 2)
        # x = self.put_together0(x)
        # x = self.batch_norm(x)
        #  out = torch.nn.functional.elu(x)
        # x = self.dropout0(x)
        # # x = x.view(x.size()[1], x.size()[0], x.size()[2])
        out = torch.nn.functional.elu(self.put_together(out))
        # out = self.put_together(out)

        out = out.view([out.shape[0], out.shape[1]])
        out = torch.transpose(out, 0, 1)
        # out = self.dropout(out)
        # out = torch.nn.functional.elu(self.put_together2(out))
        # out = self.dropout2(out)
        # out = self.put_together3(out)
        # out = out.view(out.size()[0], out.size()[2])
        return out

