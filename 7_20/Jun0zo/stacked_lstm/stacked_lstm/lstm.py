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
from keras.callbacks import Callback

class InputDataTracker(Callback):
    def __init__(self, X):
        self.X = X
        self.input_data_history = []

    def on_epoch_end(self, epoch, logs=None):
        predicted_data = self.model.predict(self.X)
        self.input_data_history.append(predicted_data)

# Data Preprocessing
temp_data = pd.read_csv('./input.csv', index_col=0)
temp_data.describe()

temp_data = temp_data.drop_duplicates(['data_date'])
temp_data.sort_values(['data_date'], inplace=True)
temp_data.reset_index(inplace=True, drop=True)
del temp_data['data_date']

scaler = MinMaxScaler()
temp_data = scaler.fit_transform(temp_data)

joblib.dump(scaler, './scaler.pkl')
scaler = joblib.load('./scaler.pkl')

Look_Back = 600
# print('Loop Start', 0, len(temp_data)-Look_Back-1)
# for k in range(len(temp_data)-Look_Back-1):
for k in range(600):
    if k % 1000 == 0:
        print('Loop', k, len(temp_data)-Look_Back-1)
    if k == 0:
        X = temp_data[k:k+Look_Back, :]
        Y = temp_data[k+Look_Back, :]
    else:
        X = np.concatenate((X, temp_data[k:k+Look_Back, :]))
        Y = np.concatenate((Y, temp_data[k+Look_Back, :]))

X = X.reshape(-1, Look_Back, temp_data.shape[1])
Y = Y.reshape(-1, temp_data.shape[1])

print(X.shape, Y.shape)

from keras.utils import plot_model

def LSTM_model():
    model=Sequential()
    model.add(LSTM(64,input_shape=(600,8),return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(8))
    model.compile(optimizer=Adam(lr=0.001),loss='mean_squared_error',metrics=['mse'])

    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.layers[0].output)
    return model, intermediate_layer_model

model, intermediate_model = LSTM_model()
model.summary()

# plot_model(model, to_file='LSTM_model.png', show_shapes=True) 

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,shuffle=True, 
                                                random_state=1004)

train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,test_size=0.2,shuffle=True, 
                                                random_state=1004)

input_data_tracker = InputDataTracker(train_x)
early_stop=EarlyStopping(monitor='val_loss',patience=2,verbose=1)
history = LSTM_model().fit(train_x,train_x,batch_size=1,epochs=1,verbose=1,validation_data=(valid_x,valid_y),callbacks=[early_stop])


# Print how the data changes through each LSTM layer
for k in range(5):  # Print the first 5 samples
    print(f"Data at the beginning of LSTM layer 1 (Sample {k + 1}):")
    print("input data :\n", train_x[k])
    print("input data shape :", train_x[k].shape)
    intermediate_output1 = intermediate_model.predict(train_x[k:k + 1])
    print(f"Data after passing through LSTM layer 1 (Sample {k + 1}):")
    print("data :\n", intermediate_output1)
    print("data shape :", intermediate_output1.shape)

    intermediate_output2 = model.layers[1](intermediate_output1)  # Output after LSTM layer 2
    print(f"Data after passing through LSTM layer 2 (Sample {k + 1}):")
    print("data :\n", intermediate_output2)
    print("data shape :", intermediate_output2.shape)

    intermediate_output3 = model.layers[2](intermediate_output2)  # Output after LSTM layer 3
    print(f"Data after passing through LSTM layer 3 (Sample {k + 1}):")
    print("data :\n", intermediate_output3)
    print("data shape :", intermediate_output3.shape)

    intermediate_output4 = model.layers[3](intermediate_output3)  # Output after LSTM layer 4
    print(f"Data after passing through LSTM layer 4 (Sample {k + 1}):")
    print("data :\n", intermediate_output4)
    print("data shape :", intermediate_output4.shape)

    output = model.layers[4](intermediate_output4)  # Final output
    print(f"Final output (Sample {k + 1}):")
    print("data :\n", output)
    print("data shape :", output.shape)


    