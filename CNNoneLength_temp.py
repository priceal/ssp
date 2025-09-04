#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025
@author: allen

secondary structure classification model. classifies protein sequences at 
residue level as 

E - strand
H - helix
C - coil/turn

uses class weights to correct imbalance in data

features to include:
    1) employ torch optimization tools      v/
    2) use scikit learn metrics             v/
    3) split test/train                     (pre-calculated)
    4) employ batch optimization            v/
    5) drop out back propagation
    6) use weights to correct imbalance     v/
    
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader

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
        string (TYPE): DESCRIPTION.

    Returns:
        array: shape = ( length of sequence, length of vocab )

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
def dataReader(filePath, lengths=(0,800), crop=800):
    '''
    

    Args:
        filePath (TYPE): DESCRIPTION.
        lengths (TYPE, optional): range of sequence lengths to accept. 
                                    Defaults to (0,800).
        crop (TYPE, optional): crop/padding length. Defaults to 800.

    Returns:
        tensor, tensor: the first is the one-hot rep of the sequence (data), 
                and the second is the one-hot rep of the secondary structure 
                (labels)
    '''
    # format of file should be each entry 2 lines, the first is the sequence
    # and the second is the secondary struture (target)
    
    minLen,maxLen = lengths
    with open(filePath, 'r') as f:
        seqOneHot = []
        tarOneHot = []
        while True:
            sequence = f.readline()[:-1]   # remove trailing newline
            if sequence == '':    # test if EOF
                break     
            target = f.readline()[:-1]
            
            if len(sequence)<minLen:   # reject less than minLength
                continue
            if len(sequence)>maxLen:   # reject greater than maxLength
                continue
            
            # reduce to crop or fill in with spaces to crop
            sequence = f'{sequence[:crop]:<{crop}}'   
            target = f'{target[:crop]:<{crop}}'
            
            # convert data strings to one-hot, add to lists of one-hot reps, 
            seqOneHot.append(oneHot(sequence, vocab="ARNDCEQGHILKMFPSTWYV"))
            tarOneHot.append(oneHot(target, vocab="HEC"))

    # arrange in arrays of shape (Nseqs(0),Nclasses(1),maxLen(2)) 
    x = np.array(seqOneHot).swapaxes(1, 2)
    y = np.array(tarOneHot).swapaxes(1, 2)

    # return tensors
    return torch.tensor(x, dtype=torch.float32, requires_grad=True), \
        torch.tensor(y, dtype=torch.float32, requires_grad=True)

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
                                      out_channels=64,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.relu1 = torch.nn.ReLU()

        self.layer2 = torch.nn.Conv1d(in_channels=64,
                                      out_channels=32,
                                      kernel_size=31,
                                      stride=1,
                                      padding='same')
        self.relu2 = torch.nn.ReLU()

        self.layer3 = torch.nn.Conv1d(in_channels=32,
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

'''
##############################################################################
###############################################################################
############################# main ############################################
###############################################################################
##############################################################################
'''
if __name__ == "__main__":

    # learning parameters
    lengthLimits = (100,400)  # screen for seq lengths in this interval
    cropSize = 100  # crop/pad all accepted seqs to this length
    numBatches = 1  # if non-zero, ignore batchSize and set to N/numBatches
    batchSize = 0  # only use if numBatches = 0
    numberEpochs = 2
    reportCycle = 1
    learningRate = 0.0000001
   # classWeights = ( 0.8, 1.4, 0.7 )  # (H, E, C)
    classWeights = ( 2.975837 , 4.283632 , 2.3228085 )  # (H, E, C)
   
    # file to load and optional file directory---can leave undefined '' or '.'
    inputTrain = 'pisces100to400.train.txt'
    inputTest = 'pisces100to400.test.txt'
    fileDirectory = 'data'

    ###########################################################################

    # load data
    xTest, yTest = dataReader(os.path.join(
        fileDirectory, inputTest), lengths=lengthLimits, crop=cropSize)
    xTrain, yTrain = dataReader(os.path.join(
        fileDirectory, inputTrain), lengths=lengthLimits, crop=cropSize)
    dataTrain = seqDataset(xTrain, yTrain)

    print("DATA SET ")
    rows = ['training data', 'training labels', 'test data', 'test labels']
    ds = [xTrain, yTrain, xTest, yTest]
    print("{:20} {:15} {:15} {:15}".format(
        'DATA', 'ENTRIES', 'CHANNELS', 'LENGTH'))
    for r, d in zip(rows, ds):
        a, b, c = d.shape
        print(f"{r:<20} {a:<15} {b:<15} {c:<15}")

    if numBatches > 0:
        batchSize = int(len(xTrain)/numBatches)
    else:
        numBatches = int(len(xTrain)/batchSize) + 1
    dataloader = DataLoader(dataTrain, batch_size=batchSize, shuffle=True)
    print('number of batches:', numBatches)
    print('size of batches:', batchSize)

    # create model
    model = cnnModel()
    print('\nMODEL ')
    print("{0:20} {1:20}".format("MODULES", "PARAMETERS"))
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print("{0:20} {1:<20}".format(name, params))
        total_params += params
    print("{0:20} {1:<20}".format("TOTAL", total_params))
    
    # create weights for classes
    junk, nClasses, nLengths = yTrain.shape
    ww=np.ones([nClasses,nLengths])*np.array(classWeights)[:,np.newaxis]
    weights = torch.tensor(ww)


