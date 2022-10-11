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


def rrmse_loss(outputs, label):
    """Computes the rrmse value"""
    if outputs.dim() == 4:
        outputs = outputs.squeeze()
        label = label.squeeze()
    error = torch.abs(outputs - label) / label
    # rrmse = torch.mean(error.view(-1))

    # if rrmse < 0.002:
    chan = label.shape[0]
    # flat_error = error.clone().detach().view([chan, -1])
    # flat_outputs = outputs.clone().detach().view([chan, -1])
    # flat_label = label.clone().detach().view([chan, -1])

    flat_error = torch.zeros_like(error)
    flat_outputs = torch.zeros_like(outputs)
    flat_label = torch.zeros_like(label)

    for i in range(chan - 1):
        # lf = i - 1 if i > 0 else 0
        r = i + 1 if i < chan - 1 else chan - 1

        flat_error[i] = outputs[i] * label[i]
        flat_outputs[i] = outputs[i] ** 2
        flat_label[i] = label[i] ** 2

        # y1 = lambda a, b: (flat_outputs[a] - flat_outputs[b])
        # y2 = lambda a, b: (flat_label[a] - flat_label[b])

        # val = (
        #         torch.abs(torch.arccos(torch.round(
        #             torch.sqrt(((flat_rrmse ** 2 + y1(r, i) * y2(r, i)) ** 2)
        #                        / (flat_rrmse ** 2 + y1(r, i) ** 2) / (flat_rrmse ** 2 + y2(r, i) ** 2))
        #             , decimals=3)))
        #         +
        #         torch.abs(torch.arccos(torch.round(
        #             torch.sqrt(((flat_rrmse ** 2 + y1(i, lf) * y2(i, lf)) ** 2)
        #                        / (flat_rrmse ** 2 + y1(i, lf) ** 2) / (flat_rrmse ** 2 + y2(i, lf) ** 2))
        #             , decimals=3)))
        # )
        # check = torch.isnan(val)
        # if check[check==True].size(0):
        #     temp = torch.sqrt(((flat_rrmse ** 2 + y1(r, i) * y2(r, i)) ** 2)
        #                       / (flat_rrmse ** 2 + y1(r, i) ** 2) / (flat_rrmse ** 2 + y2(r, i) ** 2))
        #     for j, nan in enumerate(temp):
        #         if nan > 1 or nan < -1:
        #             print("1: ", i, j, "%.20f" % temp[j], (flat_rrmse ** 2 + y1(r, i) * y2(r, i))[j]
        #                   , torch.sqrt((flat_rrmse ** 2 + y1(r, i) ** 2) * (flat_rrmse ** 2 + y2(r, i) ** 2))[j]
        #                   , y1(r, i)[j], y2(r, i)[j], flat_rrmse)
        #     temp = torch.sqrt(((flat_rrmse ** 2 + y1(i, lf) * y2(i, lf)) ** 2)
        #                       / (flat_rrmse ** 2 + y1(i, lf) ** 2) / (flat_rrmse ** 2 + y2(i, lf) ** 2))
        #     for j, nan in enumerate(temp):
        #         if nan > 1 or nan < -1:
        #             print("2: ", i, j, "%.20f" % temp[j], (flat_rrmse ** 2 + y1(i, lf) * y2(i, lf))[j]
        #                   , torch.sqrt((flat_rrmse ** 2 + y1(i, lf) ** 2) * (flat_rrmse ** 2 + y2(i, lf) ** 2))[j]
        #                   , y1(i, lf)[j], y2(i, lf)[j], flat_rrmse)

        # flat_error[i] = (
        # torch.arccos(torch.round(
        #     torch.sqrt(((flat_rrmse ** 2 + y1(r, i) * y2(r, i)) ** 2)
        #                / (flat_rrmse ** 2 + y1(r, i) ** 2) / (flat_rrmse ** 2 + y2(r, i) ** 2))
        #     , decimals=3))
        # +
        # torch.arccos(torch.round(
        #     torch.sqrt(((flat_rrmse ** 2 + y1(i, lf) * y2(i, lf)) ** 2)
        #                / (flat_rrmse ** 2 + y1(i, lf) ** 2) / (flat_rrmse ** 2 + y2(i, lf) ** 2))
        #     , decimals=3))
        # )

    flat_error = torch.sum(flat_error, dim=0)
    flat_outputs = torch.sum(flat_outputs, dim=0)
    flat_label = torch.sum(flat_label, dim=0)

    # t=torch.sqrt(flat_outputs * flat_label)
    # t = torch.arccos(torch.round(flat_error/torch.sqrt(flat_outputs*flat_label), decimals=4))
    # check = torch.isnan(t)
    # if check[check==True].size(0):
    #     for i, nan in enumerate(t):
    #         if torch.isnan(nan):
    #             print(flat_error[i], flat_outputs[i], flat_label[i], flat_error[i]/torch.sqrt(flat_outputs[i]*flat_label[i]))
    #     exit()
    #     print(rrmse, t)

    flat_outputs[flat_outputs < 0] = 0
    flat_error = flat_error / torch.sqrt(flat_outputs * flat_label)
    flat_error[flat_error > 1] = 0
    check = torch.isnan(flat_error)
    # for i, fe in enumerate(flat_error):
    #     for j, nan in enumerate(fe):
    #         print(nan)
    #         print(torch.isnan(nan))
    #         exit()
    if check[check==True].size(0):
        for i, fe in enumerate(flat_error):
            for j, nan in enumerate(fe):
                if torch.isnan(nan):
                    print(flat_error[i][j], flat_outputs[i][j], flat_label[i][j], flat_error[i][j]/torch.sqrt(flat_outputs[i][j]*flat_label[i][j]))
        exit()

    sam = Variable(torch.mean(torch.arccos(flat_error)), requires_grad=True)

    return sam

    # return rrmse
