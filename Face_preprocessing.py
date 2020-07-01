# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:59:18 2020

@author: Alessio Saladino
"""


from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import cv2
import glob
import pickle

img_dir="Dataset/Faces"
filepath="HaarCascades"
cascades = []
for file in glob.glob(filepath+"/*frontalface*"):
    file = cv2.CascadeClassifier(file)
    cascades.append(file)
    
X_data = []
y_data =[]
IMAGE_SIZE=200
for root,dirs,files in os.walk(img_dir):
    j=0
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            image = cv2.imread(path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            for classifier in cascades:
                face = classifier.detectMultiScale(image,1.3,5)
                if face!=():
                    break
            for(x,y,w,h) in face:
                roi=image[y:y+h,x:x+w]
                roi = Image.fromarray(roi)
                roi = roi.resize((IMAGE_SIZE,IMAGE_SIZE),Image.ANTIALIAS)
                roi = np.array(roi)
                M = roi.shape[0]//2
                N = roi.shape[1]//2
                tiles = [roi[x:x+M,y:y+N] for x in range(0,roi.shape[0],M) for y in range(0,roi.shape[1],N)]
                tiles = np.array(tiles)
                histograms = []
                for i in range(0,tiles.shape[0]):
                    hist = local_binary_pattern(tiles[i],8,1)
                    hist = hist.flatten()
                    histograms.append(hist)
                X_data.append(np.array(histograms).flatten())
                path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
                y_data.append(os.path.basename(path))
                print(os.path.basename(path).replace("'",""))
                j+=1
            if(j==5):
                j=0
                break



X_data = np.array(X_data)
y_data = np.array(y_data)

X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2)
X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size=0.5)

with open("FaceData/XTrain.hdf5","wb") as f:
    pickle.dump(X_train,f)

with open("FaceData/yTrain.hdf5","wb") as f:
    pickle.dump(y_train,f)
    
with open("FaceData/Xtest.hdf5","wb") as f:
    pickle.dump(X_test,f)

with open("FaceData/yTest.hdf5","wb") as f:
    pickle.dump(y_test,f)

with open("FaceData/Xval.hdf5","wb") as f:
    pickle.dump(X_val,f)

with open("FaceData/yVal.hdf5","wb") as f:
    pickle.dump(y_val,f)