# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:37:36 2023

@author: Kugavathanan  Yohanathan
         Sayanthana  Yoganathan
         
         COMP-257
"""
#import 
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras 
from sklearn.metrics import classification_report, confusion_matrix

#Data Load and preprocessing
data=scipy.io.loadmat("umist_cropped.mat")
print(dir(data))
print(data.keys())
data=data['facedat']

print(type(data))

#print(data)

print(data[0].shape)
print(data[0][0].shape)

# Restructuring the data
dataX=np.empty((0,))
dataY=[]
total=0

for i,v in enumerate(data[0]):
    print(f'v.transpose((2, 0, 1)).shape: {v.transpose((2, 0, 1)).shape}')
    total+=v.transpose((2, 0, 1)).shape[0]
    dataX=np.append(dataX, v.transpose((2, 0, 1)).ravel(), axis=0)
    for j in range(v.transpose((2, 0, 1)).shape[0]):
        dataY.append(i)

print(total)   

dataX=dataX.reshape(total,112,92)
dataY=np.array(dataY)
print(dataX.shape)
print(dataY.shape)

#1) Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set. 
#Provide your rationale for the split ratio [5 points]

# splitting the data train:validation
sss = StratifiedShuffleSplit(test_size=0.3, random_state=10)

for train_index, test_index in sss.split(dataX, dataY):
    X_train, X_temp = dataX[train_index], dataX[test_index]
    y_train, y_temp = dataY[train_index], dataY[test_index]

sss_validation = StratifiedShuffleSplit (test_size=0.5, random_state=10)

for train_index, validation_index in sss_validation.split(X_temp, y_temp):
    X_validation, X_test = X_temp[train_index], X_temp[validation_index]
    y_validation, y_test = y_temp[train_index], y_temp[validation_index]
    

print(np.unique(y_train,return_counts=True))
print(np.unique(y_validation,return_counts=True))
print(np.unique(y_test,return_counts=True))


print(X_train.shape,X_validation.shape,X_test.shape)
print(y_train.shape,y_validation.shape,y_test.shape)

# reshaping the data as per the input for our models
X_train=X_train.reshape(X_train.shape[0],112*92)
X_validation=X_validation.reshape(X_validation.shape[0],112*92)
X_test=X_test.reshape(X_test.shape[0],112*92)
print(X_train.shape,X_validation.shape,X_test.shape)

#2) Discuss the rationale behind how your team preprocess the data. Include the mathematical equations used and any dimensionality reduction applied 
#to the instanced and discuss its relevance to the problem at hand. Note that your team will receive more points if you perform data preprocessing 
#that help improve the eventual training process. [20 points]
# transformed the data using LinearDiscriminant method
projector = LinearDiscriminantAnalysis(n_components=5)
faces_points = projector.fit_transform(X=X_train,y=y_train)

faces_points_validation=projector.transform(X_validation)
faces_points_test=projector.transform(X_test)

print(faces_points)
print(faces_points.shape)

#plotting the distribution of actual data
plt.figure(figsize=(15,15))
plt.scatter(faces_points[:,0],faces_points[:,1],c=y_train, cmap='viridis')
plt.title('Actual Data Before Clustering')
plt.colorbar()
plt.show()

#3)Select a clustering technique taught in this course and apply it on the training instances. Provide the rationale behind your team’s choice of 
#clustering technique and how your team tuned the parameters for the technique implemented. [30 points]
#clearly discuss - with illustrations - the architecture your team has selected for training and predicting the test instances.

# DBSCAN to cluster the DATA with various hyper parameter
score={}
for distance in np.arange(2,10,1): # 2 to 10
    for samples in range(2,5):    #2 to 5
        model = DBSCAN(eps=distance,min_samples=samples)
        faces_y_pred = model.fit_predict(faces_points)
        print(distance,samples,silhouette_score(faces_points,y_train),silhouette_score(faces_points,faces_y_pred))
        plt.figure(figsize=(15,15))
        plt.scatter(faces_points[:, 0], faces_points[:, 1], c=faces_y_pred, cmap='viridis')
        plt.title('DBSCAN Clustering Result EPS'+str(distance)+" min samples "+str(samples))
        plt.colorbar()
        plt.show()
        score["EPS "+str(distance)+" min samples "+str(samples)]=silhouette_score(faces_points,faces_y_pred)

plt.figure(figsize=(15,15))
plt.bar(score.keys(), score.values())
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
plt.show()

# best DBSCAN model for clustering 
model = DBSCAN(eps=5,min_samples=2)
faces_y_pred = model.fit_predict(faces_points)
print(i,silhouette_score(faces_points,y_train),silhouette_score(faces_points,faces_y_pred))
plt.figure(figsize=(15,15))
plt.scatter(faces_points[:, 0], faces_points[:, 1], c=faces_y_pred, cmap='viridis')
plt.title('DBSCAN Clustering Result EPS'+str(i)+" min samples "+str(j))
plt.colorbar()
plt.show()


print(np.unique(faces_y_pred))

for cluster in np.unique(faces_y_pred):
    if input("next Cluster is " +str(cluster)):
        unique_face=X_train[np.where(faces_y_pred==cluster)]
        for i in range(len(unique_face)):
            face_img = unique_face[i].reshape(112,92)
            plt.figure(figsize=(10,10))
            plt.imshow(face_img, cmap='gray')
            plt.show()

faces_y_pred_validation=model.fit_predict(faces_points_validation)
faces_y_pred_test=model.fit_predict(faces_points_test)


# Dimentionality reduction
X_train=X_train/255
X_validation=X_validation/255
X_test=X_test/255
''' ######################EXPERIMENTAL CODE######################
y_train=faces_y_pred
y_validation=faces_y_pred_validation
y_test=faces_y_pred_test
'''
model_pca=PCA(n_components=0.99)

X_train_pca=model_pca.fit_transform(X_train)
X_validation_pca=model_pca.transform(X_validation)
X_test_pca=model_pca.transform(X_test)
print(X_train_pca.shape,X_validation_pca.shape,X_test.shape)
print(y_train.shape,y_validation.shape,y_test.shape)

#4)Discuss the rationale behind your team’s choice of activation functions, loss function, and how you tuned the hyperparameters of the network model. [30 points]

#5)Discuss the results of the trained system. [5 points]

#6)Present your system. Discuss the decisions your team made, the challenges your team encountered, how your team resolved the problems, and the results. [10 points]
