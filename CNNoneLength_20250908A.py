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
#from sklearn.utils.class_weight import compute_class_weight

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
def dataReader(filePath, lengths=(0,800), crop=800):
    '''
    reads text file of protein sequences with ss assignments.
    format of file should be each entry 2 lines, the first is the sequence
    and the second is the secondary struture (target). Assumes ss assignments
    are from (H,E,C). If not, must modify below.
    
    Args:
        filePath (string): input file path.
        lengths (tuple, optional): (min,max) rejects sequences outside range
        Defaults to (0,800).
        crop (integer, optional): all output sequences will be this lengthm, 
        either cropped or padded. Defaults to 800.

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
            # reduce to crop or fill in with spaces to crop
            sequence = f'{sequence[:crop]:<{crop}}'   
            target = f'{target[:crop]:<{crop}}'
            # convert data strings to one-hot, add to lists of one-hot reps, 
            # creating lists of crop x len(vocab) arrays.
            seqOneHot.append(oneHot(sequence, vocab="ARNDCEQGHILKMFPSTWYV"))
            tarOneHot.append(oneHot(target, vocab="HEC"))

    # arrange in arrays of shape (Nseqs(0),Nclasses(1),seqlength(2) . Note, 
    # converting lists to arrays w/o swapaxes yields indices in order of
    # sequence number(0), sequence position(1), one hot code(2). Eg, if 
    # there are 1000 sequences in data, and crop=800, then size is 
    # (1000,800,20) for protein sequence and (1000,800,3) for ss data. We need
    # to swap axes 1&2 for CNN, so shapes are (1000,20,800) and (1000,3,800)
    x = np.array(seqOneHot).swapaxes(1, 2)
    y = np.array(tarOneHot).swapaxes(1, 2)

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

'''
##############################################################################
###############################################################################
############################# main ############################################
###############################################################################
##############################################################################
'''
if __name__ == "__main__":

    # learning parameters
    lengthLimits = (200,600)  # screen for seq lengths in this interval
    cropSize = 200  # crop/pad all accepted seqs to this length
    numBatches = 1  # if non-zero, ignore batchSize and set to N/numBatches
    batchSize = 0  # only use if numBatches = 0
    numberEpochs = 400
    reportCycle = 10
    learningRate = 0.1
#    classWeights = ( 2.975837 , 4.283632 , 2.3228085 )  # (H, E, C)
#   classWeights = ( 1.0, 1.0, 1.0 )  # (H, E, C)
    weights = 'calc'    # None: unweighted. 
                        # (WH, WE, WC): use fixed weights
                        # 'calc' : use calculated weights
                    
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

    # print data/batch stats
    print("DATA SET ")
    rows = ['training data', 'training labels', 'test data', 'test labels']
    ds = [xTrain, yTrain, xTest, yTest]
    print("{:20} {:15} {:15} {:15}".format('DATA', 'ENTRIES', 'CHANNELS', 'LENGTH'))
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
#    model = cnnModel()
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
                    
    # create weights for classes--should broadcast correctly in loss calc
    if not weights:
        weights=torch.tensor((1.0,1.0,1.0))
    elif weights=='calc':
        numClasses = yTrain.sum( dim=(0,2) )  # sums over entries and AA positions
        weights = numClasses.sum()/numClasses/3 # dims=(3)
    else:
        weights = torch.tensor(weights)
    weights.unsqueeze_(1)   # add dim in place to get dims = (3,1) for broadcasting

    # run cycles of optimization
    plt.figure(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    print('\nOPTIMIZATION')
    print('{:10} {:10} {:10} {:10}'.format('epoch','batch','loss-train','loss-test') )
    stepCount = 0
    for i in range(numberEpochs):
        for j, batch in enumerate(dataloader):
            
            # calculate and display loss, then back propagate
            xx, yy = batch[0], batch[1]
            optimizer.zero_grad()
            prediction = model(xx)
            lossTerms = -yy*torch.log(prediction)*weights
            loss = lossTerms.sum()/yy.shape.numel() # normalize by num of AAs?
#            loss = lossTerms.sum()

            loss.backward()
            optimizer.step()
            
            if stepCount % reportCycle == 0:
                print(f"{i:<10} {j:<10} {loss.item():<10.5}")
                plt.plot([stepCount], [loss.item()], '.k')
            
            stepCount += 1
    
    plt.show()
    
    # metrics
    
    # must convert probability-logits to one-hots ---
    # convert max logit value to 1, others 0
    targetLabels = ['H', 'E', 'C']
    print('\nFINAL METRICS')
    print('training set performance')
    yCheck = np.argmax(yTrain.detach().numpy(), axis=1).flatten()
    prediction = model(xTrain)
    pCheck = np.argmax(prediction.detach().numpy(), axis=1).flatten()
    cm = confusion_matrix(yCheck, pCheck)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=targetLabels)
    disp.plot()
    recall = np.diagonal(cm)/cm.sum(axis=1)
    precision = np.diagonal(cm)/cm.sum(axis=0)
    print('{:10} {:10} {:10}'.format('class', 'recall', 'precision'))
    for n, r, p in zip(targetLabels, recall, precision):
        print(f'{n:<10} {r:<10.4} {p:<10.4}')
    
    labelCounts = cm.sum(axis=1)
    print('label','count','fraction')
    for i,tl in enumerate(targetLabels):
        print(tl,labelCounts[i],labelCounts[i]/labelCounts.sum())
'''
    print('\ntest set performance')
    yCheck = np.argmax(yTest.detach().numpy(), axis=1).flatten()
    prediction = model(xTest)
    pCheck = np.argmax(prediction.detach().numpy(), axis=1).flatten()
    cm = confusion_matrix(yCheck, pCheck)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['H', 'E', 'C'])
    disp.plot()
    recall = np.diagonal(cm)/cm.sum(axis=1)
    precision = np.diagonal(cm)/cm.sum(axis=0)
    print('{:10} {:10} {:10}'.format('class', 'recall', 'precision'))
    for n, r, p in zip(['H', 'E', 'C'], recall, precision):
        print(f'{n:10} {r:<10.4} {p:<10.4}')
'''
