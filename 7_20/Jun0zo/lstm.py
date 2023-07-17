# Preprocessing

import pandas as pd
import numpy as np
import sys

# Visualization

import seaborn as sb
import matplotlib.pyplot as plt

# Modeling

import keras
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import joblib
import warnings
from keras.optimizers import Adam
import silence_tensorflow
warnings.filterwarnings(action='ignore')
silence_tensorflow.silence_tensorflow()

# Data Preprocessing
temp_data = pd.read_csv('/열처리 데이터.csv', index_col=0)
temp_data.describe()

temp_data = temp_data.drop_duplicates(['data_date'])
temp_data.sort_values(['data_date'], inplace=True)
temp_data.reset_index(inplace=True, drop=True)
del temp_data['data_date']

scaler = MinMaxScaler()
temp_data = scaler.fit_transform(temp_data)

joblib.dump(scaler, '/scaler.pkl')
scaler = joblib.load('/scaler.pkl')

Look_Back = 600
for k in range(len(temp_data)-Look_Back-1):
    if k == 0:
        X = temp_data[k:k+Look_Back, :]
        Y = temp_data[k+Look_Back, :]
    else:
        X = np.concatenate((X, temp_data[k:k+Look_Back, :]))
        Y = np.concatenate((Y, temp_data[k+Look_Back, :]))

X = X.reshape(-1, Look_Back, temp_data.shape[1])
Y = Y.reshape(-1, temp_data.shape[1])

print(X, Y)
