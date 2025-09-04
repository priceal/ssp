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
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
###############################################################################
######################### functions ###########################################
###############################################################################
'''
#######################################################################
def oneHot( string, vocab=" ARNDCEQGHILKMFPSTWYV" ):
    '''
    create the one-hot array for the string. 

    Args:
        string (TYPE): DESCRIPTION.

    Returns:
        array: shape = ( length of sequence, length of vocab )

    '''
    result = []
    for c in string:
        code = np.zeros( len(vocab) )      
        index=vocab.find(c) 
        code[index] = 1
        result.append(code)
        
    return np.array(result)

#######################################################################
def dataLoader( filePath, maxLen=800 ):
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
    with open( filePath, 'r' ) as f:
        lines = f.readlines()
    
    # extract every other line, starting at appropriate position
    sequences = lines[::2]
    classes = lines[1::2]
    
    # convert data strings to one-hot, create lists of one-hot reps, with 
    # uniform lengths = maxLen, left justified
    seqsOneHot = []
    for sequence in sequences:
        sequence = f'{sequence[:maxLen]:<{maxLen}}' # cut-off to maxLe
        seqsOneHot.append( oneHot(sequence, vocab=" ARNDCEQGHILKMFPSTWYV") )
    classesOneHot = []
    for clss in classes:
        clss = f'{clss[:maxLen]:<{maxLen}}' # cut-off to maxLen
        classesOneHot.append( oneHot(clss, vocab=" HEC") )
       
    # arrange in arrays of shape (Nseqs(0),Nclasses(1),maxLen(2)) -needed for CNN
    x = np.array(seqsOneHot).swapaxes(1,2)
    y = np.array(classesOneHot).swapaxes(1,2)
    
    # return tensors
    return torch.tensor( x, dtype=torch.float32, requires_grad=True ),\
        torch.tensor( y, dtype=torch.float32, requires_grad=True) 
          
        

class dataset(Dataset):
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

        return self.x[idx],self.y[idx]
        
        
        
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
        self.layer1 = torch.nn.Conv1d(21, 31, 11, padding='same')
        self.relu1 = torch.nn.ReLU()
        
        '''        
        self.layer2 = torch.nn.Conv1d(10, 21, 101, padding='same')
        self.relu2 = torch.nn.ReLU()
        '''
        self.layer3 = torch.nn.Conv1d(31, 4, 51, padding='same')
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
###############################################################################
############################# main ############################################
###############################################################################
'''

if __name__ == "__main__":

    # learning parameters
    maxLength = 150
    numberIterations = 10
    reportCycle = 1
    learningRate = 0.0000001
    testSize = 0.15

    # file to load and optional file directory---can leave undefined '' or '.'
    inputFile = 'pisces50to150.data.txt'
    fileDirectory = 'data'

    ###########################################################################
    # load data, create model
    x, y = dataLoader(os.path.join(fileDirectory,inputFile), maxLen=maxLength)
    data = dataset(x, y)
    train,test = random_split(data, [0.85,0.15])
    xTrain, yTrain = train[:][0], train[:][1]
    xTest, yTest = test[:][0], test[:][1]
    
    '''
    xTrain=x[:2000]
    xTest=x[2000:]
    yTrain=y[:2000]
    yTest=y[2000:]
    '''
    model = cnnModel()

    # display size of model
    print("Modules \t\tParameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(name,'\t\t', params)
        total_params += params
    print(f"Total Trainable Params: {total_params}")

    # run cycles of optimization
    #optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    #lossfn = torch.nn.CrossEntropyLoss()
    plt.figure(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    for i in range(numberIterations):
        
        # calculate and display loss, then back propagate
        prediction = model(xTrain)
        lossTerms = -yTrain*torch.log(prediction)-(1.0-yTrain)*torch.log(1.0-prediction) 
        loss = lossTerms.sum()
        if i%reportCycle == 0:
            print(f'{loss = }')
            # plt.plot([i],[loss.detach().item()],'.k')
            
        optimizer.zero_grad()
        loss.backward()
        
        ''' use torch optimizer
        '''
        optimizer.step()
        
        
        '''
        do manual gradient descent. n.b. need to turn off grad tracking when
        adjusting parameters, or could screw up calc

        with torch.no_grad():
            for p in model.parameters():
                p -= p.grad*learningRate
                p.grad.zero_()
        '''
       
    # metrics
    
    # must convert probability-logits to one-hots ---
    # convert max logit value to 1, others 0
    yCheck = np.argmax(data.y.detach().numpy(),axis=1).flatten()
    predictionTest = model(data.x)
    pCheck = np.argmax(predictionTest.detach().numpy(),axis=1).flatten()
    
    cm = confusion_matrix( yCheck, pCheck ) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, \
                                  display_labels=['_','H','E','C'])
    

    disp.plot()

    plt.show()
 
    recall = np.diagonal(cm)/cm.sum(axis=1)
    precision = np.diagonal(cm)/cm.sum(axis=0)
    
    print('class\trecall\tprecision')
    for n,r,p in zip(['_','H','E','C'],recall,precision):
        print(f'{n:<5}\t{r:<5.4}\t{p:<5.4}')
        
    
 
    
 
