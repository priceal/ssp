#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025
@author: allen

utilities for ssp
    
"""
import numpy as np
import torch
from torch.utils.data import Dataset

'''
###############################################################################
######################### functions/classes ###################################
###############################################################################
'''
def oneHot(string, vocab="ARNDCEQGHILKMFPSTWYV"):
    '''
    create the one-hot array for the string. Any un-recognized character 
    (or space) will return as a zero vector.

    Args:
        string (TYPE): input sequence
        vocab (TYPE, optional): symbol list. order defines encoding. 
        Defaults to "ARNDCEQGHILKMFPSTWYV".

    Returns:
        array: NxM where N is length of input string and M is length of vocab 
        string

    '''
    result = []
    for c in string:
        code = np.zeros(len(vocab))
        if c not in vocab:
            result.append(code)
            continue
        code[ vocab.find(c) ] = 1
        result.append(code)
        
    return np.array(result)

#######################################################################
def dataReader(filePath, lengths=(0,800), crop=800, swap=False):
    '''
    reads text file of protein sequences with ss assignments.
    format of file should be each entry 2 lines, the first is the sequence
    and the second is the secondary struture (target). Assumes ss assignments
    are from (H,E,C). If not, must modify below.
    
    Args:
        filePath (string): input file path.
        lengths (tuple, optional): (min,max) rejects sequences outside range
        Defaults to (0,800).
        crop (integer, optional): all output sequences will be this length, 
        either cropped or padded. Defaults to 800.
        swap (boolean): True to put output in correct order (N,channels,position)
        for CNN modules

    Returns:
        tensor, tensor: the first is the one-hot rep of the sequence of size
        NxM where N=number of sequences and M=length of sequence vocab, and 
        the second is the one-hot rep of the secondary structure of size NxK
        where K=length of ss vocab (here assumed 3)
    '''
    minLen,maxLen = lengths
    with open(filePath, 'r') as f:
        seqOneHot = []
        tarOneHot = []
        while True:
            sequence = f.readline()[:-1]   # remove trailing newline
            if sequence == '':    # test if EOF
                break     
            target = f.readline()[:-1]
            if not minLen<=len(sequence)<=maxLen:   # reject if outside range
                continue
            # reduce to crop or fill in with spaces to crop, padding on right
            sequence = f'{sequence[:crop]:<{crop}}'   
            target = f'{target[:crop]:<{crop}}'
            # convert data strings to one-hot, add to lists of one-hot reps, 
            # creating lists of crop x len(vocab) arrays. note on padding:
            # space is recognized as padding and sent to one hot vector <0>
            seqOneHot.append(oneHot(sequence, vocab="ARNDCEQGHILKMFPSTWYV"))
            tarOneHot.append(oneHot(target, vocab="HEC"))

    if swap:
        # arrange in arrays of shape (Nseqs(0),Nclasses(1),seqlength(2) . Note, 
        # converting lists to arrays w/o swapaxes yields indices in order of
        # sequence number(0), sequence position(1), one hot code(2). Eg, if 
        # there are 1000 sequences in data, and crop=800, then size is 
        # (1000,800,20) for protein sequence and (1000,800,3) for ss data. We need
        # to swap axes 1&2 for CNN, so shapes are (1000,20,800) and (1000,3,800)
        x = np.array(seqOneHot).swapaxes(1, 2)
        y = np.array(tarOneHot).swapaxes(1, 2)
    else:
        x = np.array(seqOneHot)
        y = np.array(tarOneHot)

    # return tensors
    return torch.tensor(x, dtype=torch.float32, requires_grad=False), \
        torch.tensor(y, dtype=torch.float32, requires_grad=False)

###############################################################################
class seqDataset(Dataset):
    """  """

    def __init__(self, x, y):
        '''

        Parameters
        ----------
        
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]
