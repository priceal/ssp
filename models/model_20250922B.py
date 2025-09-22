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

        # embedding dimension = ed
        self.ed = 7
        
        # embedding transformation, input tensor must have dims (*,20), output
        # will have (*,5). since CNN uses (*,channels,length), must swap last
        # two dims before and after use
        self.embedding = torch.nn.Linear( 20, self.ed, bias=False )
     
        # input channels = number of AA's +1 for padding. pad so that
        # output is same length as input, padding character = 0
        # layer parameters = (input chans, output chans, kernel)
        self.layer1 = torch.nn.Conv1d(in_channels=self.ed,
                                      out_channels=11,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.batchnorm1 = torch.nn.BatchNorm1d(num_features=11)
        self.relu1 = torch.nn.ReLU()

        #######################################################################
        self.layer2 = torch.nn.Conv1d(in_channels=11,
                                      out_channels=11,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.batchnorm2 = torch.nn.BatchNorm1d(num_features=11)
        self.relu2 = torch.nn.ReLU()

        #######################################################################
        self.layer3 = torch.nn.Conv1d(in_channels=11,
                                      out_channels=3,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.batchnorm3 = torch.nn.BatchNorm1d(num_features=3)
        self.softMax3 = torch.nn.Softmax(dim=1)

    ###########################################################################
    def forward(self, x):

        # since CNN uses (*,channels,length), must swap last
        # two dims before and after embedding multiplication
        x = self.embedding(x.transpose(-1,-2))
        x = torch.transpose(x,-1,-2)
        
        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
               
        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.softMax3(x)

        return x
