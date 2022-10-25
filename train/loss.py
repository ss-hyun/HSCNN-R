#!/usr/local/bin/python

from __future__ import division

import math

import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import decimal


class reconstruct_loss(nn.Module):
    """the loss between the input and synthesized input"""

    def __init__(self, cie_matrix, batchsize):
        super(reconstruct_loss, self).__init__()
        self.cie = Variable(torch.from_numpy(cie_matrix).float().cuda(), requires_grad=False)
        self.batchsize = batchsize

    def forward(self, network_input, network_output):
        network_output = network_output.permute(3, 2, 0, 1)
        network_output = network_output.contiguous().view(-1, 31)
        reconsturct_input = torch.mm(network_output, self.cie)
        reconsturct_input = reconsturct_input.view(50, 50, 64, 3)
        reconsturct_input = reconsturct_input.permute(2, 3, 1, 0)
        reconstruction_loss = torch.mean(torch.abs(reconsturct_input - network_input))
        return reconstruction_loss


def loss(outputs, label):
    if outputs.dim() == 4:
        outputs = outputs.squeeze()
        label = label.squeeze()

    reflect_t = torch.transpose(outputs.view([label.shape[0], -1]), 0, 1)
    reflect_r = torch.transpose(label.view([label.shape[0], -1]), 0, 1)
    sam = torch.arccos((reflect_r * reflect_t).sum(dim=1)
                       / torch.sqrt((reflect_t ** 2).sum(dim=1) * (reflect_r ** 2).sum(dim=1) + 1e-9))
    # check = torch.isnan(sam)
    # if check[check == True].size(0):
    #     print(reflect_r[check == True], reflect_t[check == True])

    error = torch.abs(outputs - label) / label

    return torch.mean(sam) + torch.mean(error.view(-1))


def rrmse_loss(outputs, label):
    """Computes the rrmse value"""
    if outputs.dim() == 4:
        outputs = outputs.squeeze()
        label = label.squeeze()
    error = torch.abs(outputs - label) / label
    rrmse = torch.mean(error.view(-1))
    return rrmse
