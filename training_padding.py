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
    3) split test/train                     v/ (pre-calculated)
    4) employ batch optimization            v/
    5) drop out back propagation
    6) use weights to correct imbalance     v/
    7) add embedding                        v/
    8) padding to handle variable length
    9) add attention layer
    
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from ssp_utils import dataReader, seqDataset
from model_20250923_pad import cnnModel

'''
###############################################################################
############################# main ############################################
###############################################################################
'''

# learning parameters
lengthLimits = (100,600)  # screen data for seq lengths in this interval
cropSize = 400  # crop/pad all accepted seqs to this length
numBatches = 0 # if non-zero, ignore batchSize and set to N/numBatches
batchSize = 256  # only use if numBatches = 0
numberEpochs = 50
reportCycle = 29
learningRate = 0.1

refine = True   # creates new model if False

weights = 'calc'    # None: unweighted. 
                    # (WH, WE, WC): use fixed weights
                    # 'calc' : calculated weights to use
                
# file to load and optional file directory---can leave undefined '' or '.'
inputTrain = 'pisces50to600.train.txt'
inputTest = 'pisces50to600.test.txt'
fileDirectory = 'data'

###########################################################################

# load data -------------------------------------------
xTest, yTest = dataReader(os.path.join(
    fileDirectory, inputTest), lengths=lengthLimits, crop=cropSize, swap=True)
xTrain, yTrain = dataReader(os.path.join(
    fileDirectory, inputTrain), lengths=lengthLimits, crop=cropSize, swap=True)
dataTrain = seqDataset(xTrain, yTrain) # needed for batches

# print data/batch stats ------------------------------------
print("DATA SET ")
rows = ['training data', 'training labels', 'test data', 'test labels']
ds = [xTrain, yTrain, xTest, yTest]
print("{:<20} {:<15} {:<15} {:<15}".format('DATA', 'ENTRIES', 'CHANNELS', 'LENGTH'))
for r, d in zip(rows, ds):
    a, b, c = d.shape
    print(f"{r:<20} {a:<15} {b:<15} {c:<15}")
if numBatches > 0:
    batchSize = int(len(xTrain)/numBatches)
else:
    numBatches = int(len(xTrain)/batchSize)
numClasses = yTrain.sum( dim=(0,2) ) 
targetLabels = ['H', 'E', 'C']
# create weights for classes--should broadcast correctly in loss calc
if not weights:
    weights=torch.tensor((1.0,1.0,1.0))
elif weights=='calc':
     # sums over entries and AA positions
    weights = numClasses.sum()/numClasses/3 # dims=(3)
else:
    weights = torch.tensor(weights)
weights.unsqueeze_(1)   # add dim in place to get dims = (3,1) for broadcasting
print('{:<10} {:<10} {:<10} {:<10} {:<10}'.format('index','label','count','fraction','weight') )
for i,tl in enumerate(targetLabels):
    print('{:<10} {:<10} {:<10} {:<10.4} {:<10.4}'.format(
        i,tl,int(numClasses[i]),numClasses[i]/numClasses.sum(), float(weights[i,0]) 
        ) )
print('number of batches:', numBatches)
print('size of batches:', batchSize)
dataloader = DataLoader(dataTrain, batch_size=batchSize, shuffle=True)

# create model ----------------------------------------------------
if not refine:     # if refining pre-existing, don't create new model
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

# run cycles of optimization ----------------------------------------
plt.figure(1)
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
print('\nOPTIMIZATION')
print('{:10} {:10} {:10} {:10}'.format('epoch','batch','loss-train','loss-test') )
stepCount = 0
for i in range(numberEpochs):
    for j, batch in enumerate(dataloader):
        
        # calculate and display loss, then back propagate
        xx, yy = batch[0], batch[1]
        # make the mask, sum along axis=1 (channels) to get 1 in each valid
        # position, 0 in  cropped. then add summed dim back (unsqueeze)
        yymask = ( yy.sum(axis=1) ).unsqueeze_(1)
        prediction = model(xx)
        lossTerms = -yy*torch.log(prediction)*weights*yymask
        loss = lossTerms.sum()/yy.shape.numel() # normalize by num of AAs
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print out if report cycle done
        if stepCount % reportCycle == 0:
            
            # calc test loss
            testPrediction = model( xTest ) 
            testLossTerms = -yTest*torch.log(testPrediction)*weights
            testLoss = testLossTerms.sum()/yTest.shape.numel() # normalize by num of AAs
            print(f"{i:<10} {j:<10} {loss.item():<10.5} {testLoss.item():<10.5}")
            plt.plot([stepCount], [loss.item()], '.k')
            plt.plot([stepCount], [testLoss.item()], '.r')
        
        stepCount += 1

plt.show()

# metrics -------------------------------------------------------
# must convert probability-logits to one-hots ---
# convert max logit value to 1, others 0
print('\nFINAL METRICS')
titles = ['training','test']
xSets = [ xTrain, xTest ]
ySets = [ yTrain, yTest ]
for t,xs,ys in zip(titles,xSets,ySets):

    # problem: whenever zero-padding is encountered, argmax returns class 0!
    # that is checked against some 'random' prediction !!
    # makes it look worse than it is!
    print(t+' set performance')
    mask = (ys.sum(axis=1)).flatten()
    yCheck = np.argmax(ys.detach().numpy(), axis=1).flatten()
    prediction = model(xs)
    pCheck = np.argmax(prediction.detach().numpy(), axis=1).flatten()
    cm = confusion_matrix(yCheck, pCheck, sample_weight=mask)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=targetLabels)
    disp.plot()
    recall = np.diagonal(cm)/cm.sum(axis=1)
    precision = np.diagonal(cm)/cm.sum(axis=0)
    print('{:10} {:10} {:10}'.format('class', 'recall', 'precision'))
    for n, r, p in zip(targetLabels, recall, precision):
        print(f'{n:<10} {r:<10.4} {p:<10.4}')
        

