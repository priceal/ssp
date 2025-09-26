#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    
"""

import torch

###############################################################################
class cnnModel(torch.nn.Module):
    '''
    CNN that takes a numerical coding of sequence as input and outputs a one-hot
    rep of the secondary structure prediction. 
    input is expected to be (N,length), where each element is an index from 0
    to 20. 0 is assumed padding symbol.
    each layer is a 1d conv.
    followed by an activation.
    '''

    def __init__(self):
        super(cnnModel, self).__init__()

        # embedding dimension = ed
        self.ed = 7
        self.padidx = 0    # change this for another padding index
        
        # convolutional layers, in format (in_channels, out_channels, kernel_size)
        paramsHidden = (
                         ( self.ed, 5, 11, torch.nn.ReLU() ),
                         ( 5, 5, 11, torch.nn.ReLU() ),
                         ( 5, 5, 11, torch.nn.ReLU() ),
                         ( 5, 5, 11, torch.nn.ReLU() )
                       )
        paramsLast = ( 5, 3, 11 )  # lacks activation
        
        # embedding transformation, input tensor must have dims (*,20), output
        # will have (*,5). since CNN uses (*,channels,length), must swap last
        # two dims before and after use
        
        # try using torch.nn.Embedding... will convert list of indices to 
        # embedding vector (so input rep. is not oneHot vector, but list of
        # res. #'s). Has feature that can fix padding index so that embedding
        # coeffs for that are not updated---converts to 0.
        self.embedding = torch.nn.Embedding( 21, self.ed, padding_idx=self.padidx )
     
        # input channels = number of AA's +1 for padding. pad so that
        # output is same length as input, padding character = 0
        # layer parameters = (input chans, output chans, kernel)
        convLayers= []; batchLayers = []; activLayers =[]
        for inch, outch, ksize, afunc in paramsHidden:
            convLayers.append( torch.nn.Conv1d(in_channels=inch,
                                           out_channels=outch,
                                           kernel_size=ksize,
                                           stride=1,
                                           bias=False,
                                           padding='same' )
                              )
            batchLayers.append( torch.nn.BatchNorm1d(num_features=outch) )
            activLayers.append( afunc )
            
        self.conv = torch.nn.ModuleList( convLayers )
        self.batch = torch.nn.ModuleList( batchLayers )
        self.activ = torch.nn.ModuleList( activLayers )
        
        self.outLayer = torch.nn.Conv1d(in_channels=paramsLast[0],
                                       out_channels=paramsLast[1],
                                       kernel_size=paramsLast[2],
                                       stride=1,
                                       bias=False,
                                       padding='same' )
        self.outBatch = torch.nn.BatchNorm1d(num_features=paramsLast[1])
        self.outActiv = torch.nn.Softmax( dim = 1 )

    ###########################################################################
    def forward(self, x):
        '''
        
        Args:
            x (TYPE): DESCRIPTION.
            mask (TYPE): DESCRIPTION.

        Returns:
            x (TYPE): DESCRIPTION.

        '''

        # since CNN uses (*, channels, length), must swap last two dims after 
        # embedding operation which produces (*, length, embedding_dims)
        x = self.embedding(x)
        x = torch.transpose(x,-1,-2)
        
        # convolutional layers --- note, if manually padded on C-term, the
        # layers after first hidden will have c-term residues influenced by
        # a non-zero padding, which will propagate through layers!
        # might need to zero out the padding after each layer!
        for cl, bl, al in zip( self.conv, self.batch, self.activ):
            x = cl(x)
            x = bl(x)
            x = al(x)
            # add a zero-ing out step here, to set padded regions=0
            
        x = self.outLayer(x)
        x = self.outBatch(x)
        x = self.outActiv(x)

        return x
