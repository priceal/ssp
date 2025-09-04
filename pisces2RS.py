#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 10:13:16 2025

@author: allen
"""

import pandas as pd
from sklearn.model_selection import train_test_split

inputFile = 'data/2018-06-06-pdb-intersect-pisces.csv'

minLength = 100
maxLength = 400
testSize = 0.10
valSize = 0.10

outputTrain = 'data/pisces100to400.train.txt'
outputVal = 'data/pisces100to400.val.txt'
outputTest = 'data/pisces100to400.test.txt'

####################################################################
df = pd.read_csv(inputFile)

data=[]
dataDict={}
lengthDict={}
for entry in df.itertuples():

    if entry.seq.count('*')>0:      # reject non-standard AAs
        continue
    if len(entry.seq)<minLength:   # reject less than minLength
        continue
    if len(entry.seq)>maxLength:   # reject greater than maxLength
        continue
    data.append(entry.seq+'+'+entry.sst3)
 #   print(len(entry.seq))
    if len(entry.seq) not in lengthDict.keys():
        lengthDict[len(entry.seq)] = 1
    else:
        lengthDict[len(entry.seq)] += 1
 
train, val_test = train_test_split(data, test_size=testSize+valSize)
val, test = train_test_split( val_test, test_size=(testSize/(testSize+valSize)))

with open(outputTrain,'w') as f:   
    for line in train:
        d, l = line.split('+') 
        f.write( d+'\n' )
        f.write( l+'\n' )
        
with open(outputVal,'w') as f:   
        for line in val:
            d, l = line.split('+') 
            f.write( d+'\n' )
            f.write( l+'\n' )
                
with open(outputTest,'w') as f:   
    for line in test:
        d, l = line.split('+') 
        f.write( d+'\n' )
        f.write( l+'\n' )
        
         