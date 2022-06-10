#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

# 1.Loading the data, check supporting understanding data better
dataset = pd.read_csv('train.csv')
dataset.drop(dataset.columns[[13, 14, 16,17,18,19,10,14,15]], axis = 1, inplace = True)
dataset=dataset.to_numpy()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoderGender=LabelEncoder()
dataset[:,1]=labelEncoderGender.fit_transform(dataset[:,1])
dataset[:,2]=labelEncoderGender.fit_transform(dataset[:,2])
dataset[:,3]=labelEncoderGender.fit_transform(dataset[:,3])
dataset[:,5]=labelEncoderGender.fit_transform(dataset[:,5])
dataset[:,6]=labelEncoderGender.fit_transform(dataset[:,6])
dataset[:,7]=labelEncoderGender.fit_transform(dataset[:,7])
dataset[:,12]=labelEncoderGender.fit_transform(dataset[:,12])
dataframe=pd.DataFrame(dataset)
dataframe[9].replace(to_replace = 'l', value ='1',inplace=True)
dataframe[9].replace(to_replace = 'o', value ='0',inplace=True)

dataframe[10].replace(to_replace = 'b2', value ='0',inplace=True)
dataframe[10].replace(to_replace = '2b', value ='0',inplace=True)
dataframe[10].replace(to_replace = 'bb', value ='0',inplace=True)
dataframe[10].replace(to_replace = 'aa', value ='1',inplace=True)
dataframe[10].replace(to_replace = 'a2', value ='1',inplace=True)
dataframe[10].replace(to_replace = '2a', value ='1',inplace=True)
dataframe[10].replace(to_replace = 'ab', value ='3',inplace=True)
dataframe[10].replace(to_replace = 'ba', value ='3',inplace=True)
dataframe[10].replace(to_replace = '22', value ='2',inplace=True)
dataframe[10].unique()

dataframe[11].replace(to_replace = 'kK', value ='0',inplace=True)
dataframe[11].replace(to_replace = 'kk', value ='0',inplace=True)
dataframe[11].replace(to_replace = '2K', value ='0',inplace=True)
dataframe[11].replace(to_replace = 'k2', value ='0',inplace=True)
dataframe[11].replace(to_replace = 'KK', value ='0',inplace=True)
dataframe[11].replace(to_replace = 'Kk', value ='0',inplace=True)
dataframe[11].replace(to_replace = '2k', value ='0',inplace=True)
dataframe[11].replace(to_replace = 'K2', value ='0',inplace=True)
dataframe[11].replace(to_replace = '22', value ='1',inplace=True)

dataframe[11].unique()