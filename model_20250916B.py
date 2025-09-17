#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

deep narrow CNN for ssp.
add embedding layer
    
"""

import torch
import math

###############################################################################
class cnnModel(torch.nn.Module):
    '''
    CNN that takes one-hot rep of sequence as input and outputs a one-hot
    rep of the secondary structure prediction. each layer is a 1d conv.
    followed by an activation.
    '''

    def __init__(self):
        super(cnnModel, self).__init__()

        # embedding dimension = ed, key/query space dim = kd
        self.ed = 7
        self.kd = 4
       
        # embedding transformation, input tensor must have dims (*,20), output
        # will have (*,ed). since CNN uses (*,channels,length), must swap last
        # two dims before use
        self.embedding = torch.nn.Linear( 20, self.ed, bias=False )
    
        # setup query, key and value matrices. since these also work on dim=-1
        # last two dims must be swapped from CNN order
        self.query = torch.nn.Linear( self.ed, self.kd, bias = False )
        self.key =   torch.nn.Linear( self.ed, self.kd, bias = False  )
        self.value = torch.nn.Linear( self.ed, self.ed, bias = False  )
        self.softMaxAtt = torch.nn.Softmax(dim=-1)

        # input channels = number of AA's +1 for padding. pad so that
        # output is same length as input, padding character = 0
        # layer parameters = (input chans, output chans, kernel)
        self.layer1 = torch.nn.Conv1d(in_channels=self.ed,
                                      out_channels=5,
                                      kernel_size=11,
                                      stride=1,
                                      bias=False,
                                      padding='same')
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Conv1d(in_channels=5,
                                      out_channels=5,
                                      kernel_size=11,
                                      stride=1,
                                      bias=False,
                                      padding='same')
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Conv1d(in_channels=5,
                                      out_channels=5,
                                      kernel_size=11,
                                      stride=1,
                                      bias=False,
                                      padding='same')
        self.relu3 = torch.nn.ReLU()

        self.layer4 = torch.nn.Conv1d(in_channels=5,
                                      out_channels=3,
                                      kernel_size=11,
                                      stride=1,
                                      bias=False, 
                                      padding='same')
        
        self.softMax4 = torch.nn.Softmax(dim=1)

    def forward(self, x):

        # since CNN uses (*,channels,length), must swap last
        # two dims before embedding multiplication, note that x will
        # exit this transformation transposed!
        x = self.embedding(torch.transpose(x,-2,-1))
         
        # since output of embedding will have dims (N,length,embedding dim)
        # can use directly in attention algorithm
        q = self.query( x )   # (N(0),length(1),kd dim(2))
        k = self.key( x )     # (N(0),length(1),kd dim(2))
        v = self.value( x )   # (N(0),length(1),embedding dim(2))
       
        # we want inner product over dim(2)
        # dims of weights should be ( N(0), length(1), length(2) )
        # transpose is to contract over kd dim in both tensors
        logits = q @ torch.transpose(k,-2,-1) / math.sqrt(self.kd)
        weights = self.softMaxAtt( logits )
        
        # for attention we need contraction of last dim of weights (source)
        # with second to last dim of v (size=embedding dim), add result to x
        x = x + weights @ v
         
        # proceed with CNN layers, swap last two indices to get back to 
        # (N,channels,length) order for first layer
        x = self.layer1(torch.transpose(x,-2,-1))
        x = self.relu1(x)
               
        x = self.layer2(x)
        x = self.relu2(x)
        
        x = self.layer3(x)
        x = self.relu3(x)

        x = self.layer4(x)
        x = self.softMax4(x)

        return x
