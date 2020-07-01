# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:20:34 2020

@author: Alessio Saladino
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import numpy as np
import os

with open("FaceData/XTrain.hdf5","rb") as f:
    X_train = pickle.load(f)

with open("FaceData/yTrain.hdf5","rb") as f:
    y_train = pickle.load(f)

    
classes_names = np.unique(y_train)
labelEncoder =  LabelEncoder()
y_encoded = labelEncoder.fit_transform(y_train)
n_classes = np.unique(y_encoded).shape[0]
for i in range(0,n_classes):
    X_resized = []
    y_resized = []
    path = ("Models/"+str(classes_names[i])+"/")
    os.makedirs(path, exist_ok=True)
    model_name = path+"FaceModel.hdf5"
    y = np.where(y_encoded!=i,0,1)
    for j in range(0,y.shape[0]):
        if(y[j]==1):
            X_resized.append(X_train[j])
            y_resized.append(y[j])
    num_positive=len(y_resized)
    j=0
    while(j<num_positive):
        if(y[j]==0):
            X_resized.append(X_train[j])
            y_resized.append(y[j])
        else:
            num_positive+=1
        j+=1
    X_resized = np.array(X_resized)
    y_resized = np.array(y_resized)
    model = SVC(C=10,kernel="rbf",probability=True)
    model.fit(X_train,y)
    with open(model_name,"wb") as f:
        pickle.dump(model,f)