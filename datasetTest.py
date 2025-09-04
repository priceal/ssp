#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:24:09 2025
@author: allen

secondary structure classification model
classifies protein sequences at residue level as 

E - strand
H - helix
C - coil/turn

features to include:
    1) employ torch optimization tools
    2) use scikit learn metrics
    3) split test/train
    4) employ batch optimization
    5) drop out back propagation
    
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader


'''
###############################################################################
######################### functions ###########################################
###############################################################################
'''
#######################################################################


def oneHot(string, vocab="ARNDCEQGHILKMFPSTWYV"):
    '''
    create the one-hot array for the string. Any un-recognized character will
    return as a 'space' or the last character in vocab due to .find() returning
    -1

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
def dataReader(filePath, maxLen=800):
    '''
    Args:
        filePath (TYPE): DESCRIPTION.
        maxLen (TYPE, optional): DESCRIPTION. Defaults to 800.

    Returns:
        tensor, tensor: the first is the one-hot rep of the sequence, and
        the second is the one-hot rep of the secondary structure (target)

    '''
    # format of file should be each entry 2 lines, the first is the sequence
    # and the second is the secondary struture (target)
    with open(filePath, 'r') as f:
        lines = f.readlines()

    # extract every other line, starting at appropriate position
    sequences = lines[::2]
    classes = lines[1::2]

    # convert data strings to one-hot, create lists of one-hot reps, with
    # max length = maxLength. fills in to maxLength for uniform length
    seqsOneHot = []
    for sequence in sequences:
        sequence = f'{sequence[:maxLen]:<{maxLen}}'  # fill out to maxLength
        seqsOneHot.append(oneHot(sequence, vocab="ARNDCEQGHILKMFPSTWYV"))
    classesOneHot = []
    for clss in classes:
        clss = f'{clss[:maxLen]:<{maxLen}}'  # cut-off to maxLen
        classesOneHot.append(oneHot(clss, vocab="HEC"))

    # arrange in arrays of shape (Nseqs(0),Nclasses(1),maxLen(2)) -needed for CNN
    x = np.array(seqsOneHot).swapaxes(1, 2)
    y = np.array(classesOneHot).swapaxes(1, 2)

    # return tensors
    return torch.tensor(x, dtype=torch.float32, requires_grad=True), \
        torch.tensor(y, dtype=torch.float32, requires_grad=True)



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
                                      out_channels=31,
                                      kernel_size=11,
                                      stride=1,
                                      padding='same')
        self.relu1 = torch.nn.ReLU()

        '''        
        self.layer2 = torch.nn.Conv1d(10, 21, 101, padding='same')
        self.relu2 = torch.nn.ReLU()
        '''
        self.layer3 = torch.nn.Conv1d(in_channels=31,
                                      out_channels=3,
                                      kernel_size=51,
                                      stride=1,
                                      padding='same')
        self.softMax3 = torch.nn.Softmax(dim=1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)
        '''        
        x = self.layer2(x)
        x = self.relu2(x)
        '''
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
    maxLength = 400
    numBatches = 0  # if non-zero, ignore batchSize and set to N/numBatches
    batchSize = 1  # only use if numBatches = 0
    numberEpochs = 1
    reportCycle = 100
    learningRate = 0.0000001

    # file to load and optional file directory---can leave undefined '' or '.'
    inputTrain = 'pisces100to400.train.txt'
    inputTest = 'pisces100to400.test.txt'
    fileDirectory = 'data'

    ###########################################################################

    # load data
    xTest, yTest = dataReader(os.path.join(
        fileDirectory, inputTest), maxLen=maxLength)
    xTrain, yTrain = dataReader(os.path.join(
        fileDirectory, inputTrain), maxLen=maxLength)
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

    # run cycles of optimization
    plt.figure(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    print('\nOPTIMIZATION')
    print('{:10} {:10} {:10} {:10}'.format('epoch','batch','loss-train','loss-test') )
    stepCount = 0
    for i in range(numberEpochs):

        for j, batch in enumerate(dataloader):

            # calculate and display loss, then back propagate
            
            # trim to length of sequence, only works for batch size 1 !!
            sampleLength = int(batch[0].sum())
            xx=batch[0][:,:,:sampleLength]
            yy=batch[1][:,:,:sampleLength]
            
            prediction = model(xx)
            lossTerms = - \
                yy*torch.log(prediction)-(1.0-yy) * \
                torch.log(1.0-prediction)
            loss = lossTerms.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if stepCount % reportCycle == 0:
                
                # trim to length of sequence, only works for batch size 1 !!
  
                #predictionTest = model(xTest)
                #lossTestTerms = - \
                 #   yTest*torch.log(predictionTest)-(1.0-yTest) * \
                 #   torch.log(1.0-predictionTest)
               # lossTest = lossTestTerms.sum()
                #print(f"{i:<10} {j:<10} {loss.item():<10.5} {lossTest.item():<10.5}")
                print(f"{i:<10} {j:<10} {loss.item():<10.5}")
                plt.plot([stepCount], [loss.item()], '.k')
               # plt.plot([stepCount], [lossTest.item()], '.r')
               
            stepCount += 1

    plt.show()

    # metrics

    # must convert probability-logits to one-hots ---
    # convert max logit value to 1, others 0
    print('\nFINAL METRICS')
    print('training set performance')
    yCheck = np.argmax(yTrain.detach().numpy(), axis=1).flatten()
    prediction = model(xTrain)
    pCheck = np.argmax(prediction.detach().numpy(), axis=1).flatten()
    cm = confusion_matrix(yCheck, pCheck)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['H', 'E', 'C'])
    disp.plot()
    recall = np.diagonal(cm)/cm.sum(axis=1)
    precision = np.diagonal(cm)/cm.sum(axis=0)
    print('{:10} {:10} {:10}'.format('class', 'recall', 'precision'))
    for n, r, p in zip(['H', 'E', 'C'], recall, precision):
        print(f'{n:<10} {r:<10.4} {p:<10.4}')

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
