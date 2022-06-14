# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:32:22 2022

@author: farha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

dataset = pd.read_csv('train.csv')
dataset.drop(dataset.columns[[0,13, 14, 16,17,18,19,10,11,14,15,12]], axis = 1, inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()

#gender
X[:,0]=le.fit_transform(X[:,0])
#s11
X[:,1]=le.fit_transform(X[:,1])
#s12
X[:,2]=le.fit_transform(X[:,2])
#s16
X[:,3]=le.fit_transform(X[:,3])
#s17
X[:,4]=le.fit_transform(X[:,4])
X[:,5]=le.fit_transform(X[:,5])
#s18
X[:,6]=le.fit_transform(X[:,6])


X=pd.DataFrame(X)

#processing s52
X[8].replace(to_replace = 'l', value =float(1),inplace=True)
X[8].replace(to_replace = 'o', value =float(0),inplace=True)
X[8]=X[8].astype(np.float64)
X[8].unique()

#extracting numerical values from s54
#X[10].replace(to_replace = 'b2', value ='0',inplace=True)
#X[10].replace(to_replace = '2b', value ='0',inplace=True)
#X[10].replace(to_replace = 'bb', value ='0',inplace=True)
#X[10].replace(to_replace = 'aa', value ='1',inplace=True)
#X[10].replace(to_replace = 'a2', value ='1',inplace=True)
#X[10].replace(to_replace = '2a', value ='1',inplace=True)
#X[10].replace(to_replace = 'ab', value ='3',inplace=True)
#X[10].replace(to_replace = 'ba', value ='3',inplace=True)
#X[10].replace(to_replace = '22', value ='2',inplace=True)

#X[10]=X[10].fillna(0)  #try another model excluding the missing values  
#X[10] = X[10].astype(np.int32)

#Extracting numerical values from s55
#X[11].replace(to_replace = 'kK', value ='0',inplace=True)
#X[11].replace(to_replace = 'kk', value ='0',inplace=True)
#X[11].replace(to_replace = 'k2', value ='0',inplace=True)
#X[11].replace(to_replace = 'KK', value ='0',inplace=True)
#X[11].replace(to_replace = '22', value ='1',inplace=True)
#X[11].replace(to_replace = 'K2', value ='0',inplace=True)
#X[11].replace(to_replace = '2k', value ='0',inplace=True)
#X[11].replace(to_replace = '2K', value ='0',inplace=True)
#X[11].replace(to_replace = 'Kk', value ='0',inplace=True)

#X[11]=X[11].fillna(0)
#X[11] = X[11].astype(np.int32)
#X[11].unique()




from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'] )

ann.fit(XTrain, yTrain, batch_size=32, epochs=100)

y_pred = ann.predict(XTest)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), yTest.reshape(len(yTest), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(yTest, y_pred)
print(cm)
accuracy_score(yTest, y_pred)*100


###### TEST DATASET WORK

testData=pd.read_csv('test.csv')
testData.drop(testData.columns[[0,13, 14, 16,17,18,19,10,11,14,15,12]], axis = 1, inplace = True)
X = testData.iloc[:, :].values

#gender
X[:,0]=le.fit_transform(X[:,0])
#gender
X[:,1]=le.fit_transform(X[:,1])
#s11
X[:,2]=le.fit_transform(X[:,2])
#s12
X[:,3]=le.fit_transform(X[:,3])
#s16
X[:,4]=le.fit_transform(X[:,4])
#s17
X[:,5]=le.fit_transform(X[:,5])
#s18
X[:,6]=le.fit_transform(X[:,6])

X=pd.DataFrame(X)
X[8].replace(to_replace = 'l', value ='1',inplace=True)
X[8].replace(to_replace = 'o', value ='0',inplace=True)
X[8]=X[8].astype(np.float64)
X[8].unique()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
y_pred = ann.predict(X)

for cell in np.nditer(y_pred, op_flags=['readwrite']):
    if cell[...] > 0.5:
         cell[...] = 1
    else:
         cell[...] = 0

test2_dataset = pd.read_csv('test.csv')
test2_dataset['label'] = y_pred
submission = test2_dataset[["id", "label"]]
submission.to_csv("submission2.csv", index = False)














