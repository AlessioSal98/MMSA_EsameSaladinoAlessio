# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:32:18 2020

@author: Alessio Saladino
"""


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pickle
import numpy as np
from matplotlib import pyplot as plt
    
with open("FaceData/Xtest.hdf5","rb") as f:
    X_test = pickle.load(f)

with open("FaceData/yTest.hdf5","rb") as f:
    y_test = pickle.load(f)
    

classes_names = np.unique(y_test)
labelEncoder =  LabelEncoder()
y_encoded = labelEncoder.fit_transform(y_test)
n_classes = np.unique(y_encoded).shape[0]
models = []

for i in range(0,n_classes):
    path = ("Models/"+str(classes_names[i])+"/")
    model_name = path+"FaceModel.hdf5"
    
    with open(model_name,"rb") as f:
        model = pickle.load(f)
    models.append(model)

conf_matrix_results = []
for i in range(0,n_classes):
    y = np.where(y_encoded!=i,0,1)
    path = ("Models/"+str(classes_names[i])+"/")
    filename = path+"FaceThreshold.txt"
    with open(filename,"rb") as f:
        threshold = pickle.load(f)
    probabilities = []
    probabilities = models[i].predict_proba(X_test)
    prediction = []
    for j in range(0,probabilities.shape[0]):
        if(probabilities[j][1]>threshold):
            prediction.append(1)
        else:
            prediction.append(0)
    prediction = np.array(prediction)
    TN, FP, FN, TP = confusion_matrix(y,prediction).ravel()
    total = (TP+FP+FN+TN)
    accuracy = (TP+TN)/total
    TN_perc = round(TN/(TN+FP),3)
    FP_perc = round(FP/(FP+TN),3)
    FN_perc = round(FN/(TP+FN),3)
    TP_perc = round(TP/(TP+FN),3)
    conf_matrix_results.append([TN_perc,FP_perc,FN_perc,TP_perc,accuracy])
    title = ("User #"+str(i))
    sn.heatmap(confusion_matrix(y,prediction),annot=True,cmap='Blues', fmt='g').set_title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    print("\nValori utente #",i)
    print("TNR: ",round(TN/(TN+FP),3))
    print("FPR: ",round(FP/(FP+TN),3))
    print("FNR: ",round(FN/(TP+FN),3))
    print("TRP: ",round(TP/(TP+FN),3))
    print("Accuracy: ",accuracy)
    
conf_matrix_results = np.array(conf_matrix_results)

print("\nMEDIA DEI VALORI:")
print("TNR: ",conf_matrix_results[:,0].mean())
print("FPR: ",conf_matrix_results[:,1].mean())
print("FNR: ",conf_matrix_results[:,2].mean())
print("TPR: ",conf_matrix_results[:,3].mean())
print("Accuracy: ",conf_matrix_results[:,4].mean())