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
dataset.drop(dataset.columns[[13, 14, 16,17,18,19,10,14,15,12]], axis = 1, inplace = True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()

#ID
X[:,0]=le.fit_transform(X[:,0])
#gender
X[:,1]=le.fit_transform(X[:,1])
#s11
X[:,2]=le.fit_transform(X[:,2])
#s12
X[:,3]=le.fit_transform(X[:,3])
#s16
X[:,5]=le.fit_transform(X[:,5])
#s17
X[:,6]=le.fit_transform(X[:,6])
#s18
X[:,7]=le.fit_transform(X[:,7])


X=pd.DataFrame(X)

#processing s52
X[9].replace(to_replace = 'l', value =float(1),inplace=True)
X[9].replace(to_replace = 'o', value =float(0),inplace=True)
X[9]=X[9].astype(np.int32)
X[9].unique()

#extracting numerical values from s54
X[10].replace(to_replace = 'b2', value ='0',inplace=True)
X[10].replace(to_replace = '2b', value ='0',inplace=True)
X[10].replace(to_replace = 'bb', value ='0',inplace=True)
X[10].replace(to_replace = 'aa', value ='1',inplace=True)
X[10].replace(to_replace = 'a2', value ='1',inplace=True)
X[10].replace(to_replace = '2a', value ='1',inplace=True)
X[10].replace(to_replace = 'ab', value ='3',inplace=True)
X[10].replace(to_replace = 'ba', value ='3',inplace=True)
X[10].replace(to_replace = '22', value ='2',inplace=True)

X[10]=X[10].fillna(0)  #try another model excluding the missing values  
X[10] = X[10].astype(np.int32)

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

testData=pd.read_csv('test.csv')
testData.drop(testData.columns[[13, 14, 16,17,18,19,10,14,15]], axis = 1, inplace = True)
testDataX = testData.iloc[:, :].values

#gender
testDataX[:, 0] = le.fit_transform(testDataX[:, 2])
#s11
testDataX[:, 1] = le.fit_transform(testDataX[:, 2])
#s12
testDataX[:, 2] = le.fit_transform(testDataX[:, 2])
#s58
testDataX[:, 5] = le.fit_transform(testDataX[:, 2])

testDataX=pd.DataFrame(testDataX)
testDataX[9].replace(to_replace = 'l', value ='1',inplace=True)
testDataX[9].replace(to_replace = 'o', value ='0',inplace=True)

#extracting numerical values from s54
testDataX[10].replace(to_replace = 'b2', value ='0',inplace=True)
testDataX[10].replace(to_replace = '2b', value ='0',inplace=True)
testDataX[10].replace(to_replace = 'bb', value ='0',inplace=True)
testDataX[10].replace(to_replace = 'aa', value ='1',inplace=True)
testDataX[10].replace(to_replace = 'a2', value ='1',inplace=True)
testDataX[10].replace(to_replace = '2a', value ='1',inplace=True)
testDataX[10].replace(to_replace = 'ab', value ='3',inplace=True)
testDataX[10].replace(to_replace = 'ba', value ='3',inplace=True)
testDataX[10].replace(to_replace = '22', value ='2',inplace=True)


testResults=[]
testDataX['label']=testResults
testDataY=testDataX.iloc[:, -1].values
testDataX = sc.fit_transform(testDataX)
testDataY = sc.fit_transform(testDataY)
for column in testDataX:
    y_pred = ann.predict(testDataX)
    y_pred = (y_pred > 0.5)
    np.concatenate((y_pred.reshape(len(y_pred), 1), testDataY.reshape(len(yTest), 1)), 1)
    testResults.append(y_pred)


















