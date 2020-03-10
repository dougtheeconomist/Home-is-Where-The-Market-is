# Title: Housing Modeling
# Author: Doug Hart
# Date Created: 2/27/2020
# Last Updated: 3/9/2020

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

import tensorflow as tf
keras = tf.keras
from scipy import stats
from sklearn.model_selection import train_test_split

from functions import make_ready, windowize_pan_data, windowize_pan_data_LD, winpan_helper, evaluate_models, evaluate_arima_model, format_list_of_floats, run_arima, windowize_data, split_and_windowize


df = pd.read_pickle('user_ready.pkl',compression='zip')
sdf =  df.loc['16037']  #df[df["city_id"] == 16037]

#Note Jack suggested narrowing to WA cities, this is a reasonable plan to me

#Conducting grid search for optimal model specifications
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = [0,1,2]
q_values = range(0, 6)
evaluate_models(sdf.med_price, sdf.date2, p_values,d_values, q_values)
#Best ARIMA(1, 1, 2) MSE=36473026.456

gs_approved_model = run_arima(sdf.med_price, sdf.date2, 1,1,2)
gs_approved_model.summary()

randsample = [num for num in np.random.randint(1,38006, 10)]
for i in randsample:
    print(condwa.iloc[i])


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~RNN Model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
#Now to build RNN model using LSTM layers

#setting the number of previous values to utilize in predicting next value
n_prev = 12

#rather than traditional test/train split, using data outside WA for train data


#Instantiating Model
Lex = keras.Sequential()
Lex.add(keras.layers.LSTM(32, input_shape=(n_prev, 1), return_sequences=True))
Lex.add(keras.layers.LSTM(32, return_sequences=False))
Lex.add(keras.layers.Dense(1, activation='linear'))
Lex.compile(optimizer='adam',
              loss='mse')


#Optional print model summary
Lex.summary()

#Fitting model with training data, setting epochs to five for now, can always do additional training later
Lex.fit(x_train, y_train, epochs=5)

#To test prior to making actual predictions
n_prev = 12

x_train, x_test, y_train, y_test = split_and_windowize(sdf.logmp, n_prev)
x_train.shape, x_test.shape, y_train.shape, y_test.shaper
y_crit = y_train[-1]


#loading Lex from file
Lex = tf.keras.models.load_model('Lex_prime')