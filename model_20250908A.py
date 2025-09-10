#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    
"""

import torch

###############################################################################
class cnnModel(torch.nn.Module):
    '''
    CNN that takes one-hot rep of sequence as input and outputs a one-hot
    rep of the secondary structure prediction. each layer is a 1d conv.
    followed by an activation.
    '''

    def __init__(self):
        super(cnnModel, self).__init__()

        # input channels = number of AA's +1 for padding. pad so that
        # output is same length as input, padding character = 0
        # layer parameters = (input chans, output chans, kernel)
        self.layer1 = torch.nn.Conv1d(in_channels=20,
                                      out_channels=11,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Conv1d(in_channels=11,
                                      out_channels=11,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Conv1d(in_channels=11,
                                      out_channels=3,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.softMax3 = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)
               
        x = self.layer2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        x = self.softMax3(x)

        return x
