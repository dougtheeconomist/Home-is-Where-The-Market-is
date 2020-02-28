# Title: Housing Modeling
# Author: Doug Hart
# Date Created: 2/27/2020
# Last Updated: 2/27/2020

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
from functions import evaluate_models, evaluate_arima_model, format_list_of_floats, run_arima

import warnings
warnings.filterwarnings("ignore")

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