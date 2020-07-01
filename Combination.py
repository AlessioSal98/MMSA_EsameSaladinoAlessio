# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:31:18 2020

@author: aless
"""

import pickle
import numpy as np
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

with open("FaceData/Xtest.hdf5","rb") as f:
    face_X_test = pickle.load(f)

with open("FaceData/yTest.hdf5","rb") as f:
    face_y_test = pickle.load(f)

with open("VoiceData/Xtest.hdf5","rb") as f:
    voice_X_test = pickle.load(f)

with open("VoiceData/yTest.hdf5","rb") as f:
    voice_y_test = pickle.load(f)
    
classes_names = np.unique(face_y_test)
labelEncoder =  LabelEncoder()
face_y_encoded = labelEncoder.fit_transform(face_y_test)
voice_y_encoded = labelEncoder.fit_transform(voice_y_test)
n_classes = len(classes_names)

face_models = []
voice_models = []

for i in range(0,n_classes):
    path = ("Models/"+str(classes_names[i])+"/")
    model_name = path+"FaceModel.hdf5"
    with open(model_name,"rb") as f:
        model = pickle.load(f)
    face_models.append(model)
    
    path = ("Models/"+str(classes_names[i])+"/")
    model_name = path+"VoiceModel.hdf5"
    
    with open(model_name,"rb") as f:
        model = pickle.load(f)
    voice_models.append(model)
    
conf_matrix_results = []
for i in range(0,n_classes):
    face_y = np.where(face_y_encoded!=i,0,1)
    voice_y = np.where(voice_y_encoded!=i,0,1)
    path = ("Models/"+str(classes_names[i])+"/")
    filename = path+"FaceThreshold.txt"
    with open(filename,"rb") as f:
        face_threshold = pickle.load(f)
    filename = path+"VoiceThreshold.txt"
    with open(filename,"rb") as f:
        voice_threshold = pickle.load(f)
    

    y_true = []
    
    for j in range(0,face_y.shape[0]):
        for z in range(0,voice_y.shape[0]):
            if(face_y[j]==1 and voice_y[z]==1):
                y_true.append(1)
            else:
                y_true.append(0)
    y_true = np.array(y_true)
    
    face_probabilities = face_models[i].predict_proba(face_X_test)
    voice_probabilities = voice_models[i].predict_proba(voice_X_test)
    face_prediction = []
    voice_prediction = []
    for j in range(0,face_probabilities.shape[0]):
        if(face_probabilities[j][1]>face_threshold):
            face_prediction.append(1)
        else:
            face_prediction.append(0)
    for j in range(0,voice_probabilities.shape[0]):
        if(voice_probabilities[j][1]>voice_threshold):
            voice_prediction.append(1)
        else:
            voice_prediction.append(0)

    face_prediction = np.array(face_prediction)
    voice_prediction = np.array(voice_prediction)
    
    prediction = []
    for j in range(0,face_prediction.shape[0]):
        for z in range(0,voice_prediction.shape[0]):
            if(face_prediction[j]==1 and voice_prediction[z]==1):
                prediction.append(1)
            else:
                prediction.append(0)
    prediction = np.array(prediction)
    TN, FP, FN, TP = confusion_matrix(y_true,prediction).ravel()
    total = (TP+FP+FN+TN)
    accuracy = (TP+TN)/total
    TN_perc = round(TN/(TN+FP),3)
    FP_perc = round(FP/(FP+TN),3)
    FN_perc = round(FN/(TP+FN),3)
    TP_perc = round(TP/(TP+FN),3)
    conf_matrix_results.append([TN_perc,FP_perc,FN_perc,TP_perc,accuracy])
    title = ("User #"+str(i))
    sn.heatmap(confusion_matrix(y_true,prediction),annot=True,cmap='Blues', fmt='g').set_title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    print("\nValori utente #",i)
    print("TNR: ",round(TN/(TN+FP),3))
    print("FPR: ",round(FP/(FP+TN),3))
    print("FNR: ",round(FN/(TP+FN),3))
    print("TPR: ",round(TP/(TP+FN),3))
    print("Accuracy: ",accuracy)
    
conf_matrix_results = np.array(conf_matrix_results)

print("\nMEDIA DEI VALORI:")
print("TNR: ",conf_matrix_results[:,0].mean())
print("FPR: ",conf_matrix_results[:,1].mean())
print("FNR: ",conf_matrix_results[:,2].mean())
print("TPR: ",conf_matrix_results[:,3].mean())
print("Accuracy: ",conf_matrix_results[:,4].mean())

