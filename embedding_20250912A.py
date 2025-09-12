#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep narrow CNN for ssp.
add embedding layer
    
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

        # embedding transformation, input tensor must have dims (*,5), output
        # will have (*,2)
        self.embedding = torch.nn.Linear( 5, 2, bias=False )
       
    def forward(self, x):

        x = self.embedding( x )

        return x
