#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

deep narrow CNN for ssp.
add embedding layer
    
"""

import torch
import math

# key/query dimension
kd = 2

# example has dims (N,length,embedding dims) = (2,6,3)
x = [
     [ [1.,0.,0.],
     [0.,1.,0.],
     [0.,0.,1.],
     [2.,0.,0.],
     [0.,2.,0.],
     [0.,0.,2.] ], 
    [ [-1.,0.,0.],
     [0.,-1.,0.],
     [0.,0.,-1.],
     [-2.,0.,0.],
     [0.,-2.,0.],
     [0.,0.,-2.] ] ] 

x=torch.tensor(x)

# embedding dimension = ed
ed = x.shape[-1]

query = torch.nn.Linear( ed, kd, bias = False )
key =   torch.nn.Linear( ed, kd, bias = False  )
value = torch.nn.Linear( ed, ed, bias = False  )
        
# in general, dims of x will be (Nseqs(0),seqlength(1),embedding dim(2))
q = query( x )   # (N(0),length(1),kd dim(2))
k = key( x )     # (N(0),length(1),kd dim(2))
v = value( x )   # (N(0),length(1),embedding dim(2))
   
# dims of q and k will be ( Nseqs(0),seqlength(1),key/query dim(2) )
# we want inner product over dim(2)
# dims of weights should be ( Nseqs(0), seqlength(1), seqlength(2) )
logits = q @ k.transpose(-2, -1) / math.sqrt(kd)
weights = torch.softmax( logits, dim=-1 )

# for attention we need contraction of last dim of weights (source)
# with second to last dim of v (size=embedding dim)
attention = weights @ v

