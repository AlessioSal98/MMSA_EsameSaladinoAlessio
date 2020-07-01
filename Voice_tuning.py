# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:01:23 2020

@author: Alessio Saladino
"""

from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc
import pandas as pd

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])

with open("VoiceData/Xval.hdf5","rb") as f:
    X_test = pickle.load(f)

with open("VoiceData/yVal.hdf5","rb") as f:
    y_test = pickle.load(f)
    

classes_names = np.unique(y_test)
labelEncoder =  LabelEncoder()
y_encoded = labelEncoder.fit_transform(y_test)
n_classes = np.unique(y_encoded).shape[0]
models = []

for i in range(0,n_classes):
    path = ("Models/"+str(classes_names[i])+"/")
    model_name = path+"VoiceModel.hdf5"
    with open(model_name,"rb") as f:
        model = pickle.load(f)
    models.append(model)


for i in range(0,n_classes):
    FAR = []
    FRR = []
    trs = 0
    distances = []
    y = np.where(y_encoded!=i,0,1)
    probabilities = models[i].predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y, probabilities[:,1])
    roc_auc = auc(fpr,tpr)
    plt.show()
    user_threshold = Find_Optimal_Cutoff(y,probabilities[:,1])
    path = ("Models/"+str(classes_names[i])+"/")
    filename = path+"VoiceThreshold.txt"
    with open(filename,"wb") as f:
        pickle.dump(user_threshold,f)