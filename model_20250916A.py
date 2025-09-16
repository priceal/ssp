#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

deep narrow CNN for ssp.
add embedding layer
    
"""

import torch, math

###############################################################################
class attModel(torch.nn.Module):
    '''
    CNN that takes one-hot rep of sequence as input and outputs a one-hot
    rep of the secondary structure prediction. each layer is a 1d conv.
    followed by an activation.
    '''

    def __init__(self, ed, kd ):
        super(attModel, self).__init__()

        # query/key/value
        # embedding dimension = ed
        # key/query space dim = kd
        self.kd = kd
        self.ed = ed
        
        self.query = torch.nn.Linear( ed, kd, bias = False )
        self.key =   torch.nn.Linear( ed, kd, bias = False  )
        self.value = torch.nn.Linear( ed, ed, bias = False  )
        
        self.softMax = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        # in general, dims of x will be (Nseqs(0),seqlength(1),embedding dim(2))
        q = self.query( x )   # (Nseqs(0),seqlength(1),kd dim(2))
        k = self.key( x )     # (Nseqs(0),seqlength(1),kd dim(2))
        v = self.value( x )   # (Nseqs(0),seqlength(1),embedding dim(2))
       
        # dims of q and k will be ( Nseqs(0),seqlength(1),key/query dim(2) )
        # we want inner product over dim(2)
        # dims of weights should be ( Nseqs(0), seqlength(1), seqlength(2) )
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.kd)
        weights = self.softMax( logits )
        
        # for attention we need contraction of last dim of weights (source)
        # with second to last dim of v (size=embedding dim)
        attention = weights @ v
        
        return attention
