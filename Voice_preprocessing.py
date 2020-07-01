# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:46:36 2020

@author: Alessio Saladino
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd

DATA_DIR = 'Dataset/Voices/'

dataframe = pd.read_csv("vox1_meta.csv",sep="\t")
dataframe = dataframe[["VoxCeleb1 ID","VGGFace1 ID"]]

X_data=[]
y_data=[]
max_length=0
min_length = 5000
lengths = []
for root,dirs,files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith("wav"):
            path=os.path.join(root,file)
            wave, samplerate = librosa.load(path, mono=True)
            #stampa il numero di filtri per finestra e la lunghezza di ogni finestra
            mfcc = librosa.feature.mfcc(wave, samplerate,n_mfcc=12)
            length = len(mfcc[0])
            lengths.append(length)
            print(mfcc.shape)
            if(length>max_length):
                max_length=length
            if(length<min_length):
                min_length=length

lengths = np.array(lengths)
mean = int(lengths.mean())
mean = int(min_length)
print(mean)
for root,dirs,files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith("wav"):
            path=os.path.join(root,file)
            wave, samplerate = librosa.load(path, mono=True)
            mfcc = librosa.feature.mfcc(wave, samplerate,n_mfcc=12)
            mfcc = np.pad(mfcc, ((0,0), (0, max_length-len(mfcc[0]))), mode='constant', constant_values=0) 
            startFrame = 0
            endFrame = int(min_length)
            while True:
                mfccSlice = []
                for i in range(0,12):
                    mfccSlice.append(mfcc[i,startFrame:endFrame])
                mfccSlice = np.array(mfccSlice)
                delta = librosa.feature.delta(mfccSlice,order=1)
                
                id_ = os.path.dirname(os.path.dirname(path))
                id_ = os.path.basename(id_)
                label = dataframe.loc[dataframe['VoxCeleb1 ID'] == id_]["VGGFace1 ID"]
                label = str(np.array(label))
                label = label.replace("[","")
                label = label.replace("]","")
                label = label.replace("'","")
                
                mfccSlice = mfccSlice.flatten()
                delta = delta.flatten()
                features = []
                features.append(mfccSlice)
                features.append(delta)
                features = np.array(features)
                features = features.flatten()
                startFrame = endFrame+1
                endFrame = startFrame+mean
                if(sum(mfccSlice[int(mfccSlice.shape[0]/2):])==0 or endFrame>max_length):
                    break
                X_data.append(features)
                y_data.append(label)
                print(features.shape)
X_data = np.array(X_data)
y_data = np.array(y_data)


X_train,X_test,y_train,y_test = train_test_split(X_data,y_data, test_size=0.2)
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.5)


with open("VoiceData/XTrain.hdf5","wb") as f:
    pickle.dump(X_train,f)

with open("VoiceData/yTrain.hdf5","wb") as f:
    pickle.dump(y_train,f)
    
with open("VoiceData/Xtest.hdf5","wb") as f:
    pickle.dump(X_test,f)

with open("VoiceData/yTest.hdf5","wb") as f:
    pickle.dump(y_test,f)
    
with open("VoiceData/Xval.hdf5","wb") as f:
    pickle.dump(X_val,f)

with open("VoiceData/yVal.hdf5","wb") as f:
    pickle.dump(y_val,f)