# Title: functions
# Author: Doug Hart
# Date Created: 2/25/2020
# Last Updated: 3/10/2020

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import tensorflow as tf
keras = tf.keras
from scipy import stats
from sklearn.model_selection import train_test_split

def make_date(val):
    x= datetime.datetime(int(val[0:4]), int(val[5:8]), 1)
    return x

def convert_panel(df,ec):
    '''
    takes csv with a single variabe organized by group with columns for 
    time periods and transposes so that time period can be used as first
    index and group id as second index with single column of data
    
    df: pandas dataframe from original csv

    ec: number of extra columns before actual variable values start
    '''
    spine = np.array(df.iloc[0,ec:-1])
    first = np.array(df.iloc[1,ec:-1])
    final_form = np.hstack([spine.reshape(spine.shape[0],1), np.full((spine.shape[0],1),df.iloc[1,0]), 
    np.full((spine.shape[0],1),df.iloc[1,1]), np.full((spine.shape[0],1),df.iloc[1,2]), 
    first.reshape(first.shape[0],1)])
    for i in range(2, df.shape[0]):
        this_row = np.array(df.iloc[i,ec:-1])
        addition = np.hstack([spine.reshape(spine.shape[0],1), np.full((spine.shape[0],1),df.iloc[i,0]), np.full((spine.shape[0],1),df.iloc[i,1]), np.full((spine.shape[0],1),df.iloc[i,2]), this_row.reshape(first.shape[0],1)])
        final_form = np.vstack([final_form, addition])

    return final_form

def load_and_convert(filepath,ec):
    '''
    For loading csv files and converting them into panel format
    filepath: filepath of csv or other file with variable to convert

    ec: number of extra columns prior to variable of interest to pass to conver_panel
    '''
    df = pd.read_csv(filepath,header=None)
    newsheet = convert_panel(df,ec)
    dfpr = pd.DataFrame(newsheet)
    return dfpr

def int_convert(val):
    crit = type(val)
    if crit == str:
        val = float(val)
    return val

def format_list_of_floats(L):
    return ["{0:2.2f}".format(f) for f in L]

def run_arima(series, date, p,d,q):
    '''should run ARIMA regression on specified series
    need to add returns for fit statistics for comparison
    series: column of df to forecast
    dates: date column, as multi-index doesn't seem compatible here
    
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters to use.
    '''
    model = ARIMA(series, dates = date, order=(p, d, q)).fit()
    
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax.plot(series.index, series)
    fig = model.plot_predict('2020-1-1', '2021', 
                                  dynamic=True, ax=ax, plot_insample=False)
    
    print("ARIMA(1, 1, 5) coefficients from first model:\n  Intercept {0:2.2f}\n  AR {1}".format(
    model.params[0], 
        format_list_of_floats(list(model.params[1:]))
    ))
    
    return model

#Work in progress, will likely have to come back and employ OOP to get right
estimates_stats = dict()
def estimates_store(model):
#     if estimates_stats:
    estimates_stats.update({model: (model.aic, model.bic)})
#     else:
#         estimates_stats = dict()
#         estimates_stats.update({model: (model.aic, model.bic)})
    return estimates_stats


def winpan_helper(data, n_prev):
    '''
    Modified windowizer function specifically to work with the windowizer
    function for panel data. Passes back lists for x,y of individual panel
    to be aggregated together for n_panels in outer function.

    data: series passed in from windowize_pan_data

    n_prev: number of previous values to use as x variables for y
    '''
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    # this might be too clever
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices, None]
    xlist = [i for i in x]
    ylist = [i for i in y]
    return xlist, ylist



def windowize_pan_data(data, id, n_prev):
    '''
    For creating previous time windows to be used as independent variables X
    for y from panel data. 

    data: pandas series to predict

    id_: pandas series used to id panel groups

    n_prev: number of previous values to use as x variables for y
    '''
    gx = []
    gy = []
    for i in id_:
        gslice = []
        for j in range(0, data.shape[0]):
            if id_[j] == i:
                gslice.append(data[j])

        gser = pd.Series(gslice)
    #     print('*')
        glists, gvals = winpan_helper(gser, n_prev)
        gy += gvals
        gx += glists
    return gx, gy
    

def windowize_pan_data_LD(data, id, n_prev):
    '''
    For creating previous time windows to be used as independent variables X
    for y from panel data. This version transforms series in question to be 
    first difference of logarithm base e of original series. 

    data: pandas series to predict

    id_: pandas series used to id panel groups

    n_prev: number of previous values to use as x variables for y
    '''
    gx = []
    gy = []
    for i in id_:
        gslice = []
        for j in range(0, data.shape[0]):
            if id_[j] == i:
                gslice.append(data[j])

        gser = pd.Series(gslice)
        lgser = pd.Series(np.log(gser))
        sdf['fstd'] = None
        for k in range(0, len(lgser)):
            ldgser[k] = (lgser[k]-lgser[k-1])
    
        glists, gvals = winpan_helper(ldgser, n_prev)
        gy += gvals
        gx += glists
    return gx, gy

def reconstruct(series, start):
    '''
    takes predictions from rnn model in the form of differenced log of original 
    variable and un-differences then reverses logarithmic transformation to 
    return interpretable results of rnn forecast. 

    series: series of predictions generated with rnn

    start: last value prior to start of forecast, used to un-first-difference
            the data. Must be in logarithmic form
    '''
    predlist = [start]
    for k in range(0, len(series)):
            predlist.append(predlist[-1]+lgser[k])
    final_predictions = [np.e**val for val in predlist]

    return final_predictions

def get_actual(y_crit, preds):
    log = [y_crit]
    for i in preds:
        log.append(log[-1]+i)

    logs = pd.Series(log)
    newpreds = np.e**(logs)
    return newpreds


def ensemble(m1,m2):
    me = []
    for i in range(0, len(m1)):
        me.append(np.mean(m1[i],m2[i]))
    out = np.array(me)
    return out

def predict_gen(df):
    for i in range(0, df.shape[0]):
        df.city[i]
    return None

def assign_names(df):
    df.rename(columns={0: 'date', 
                   1: 'id', 2: 'city', 3: 'state', 4: 'est_val' }, inplace=True)
    return df

def oprep(df):
    '''
    Takes single city dataframe, resets index and conducts data transformations
    for model input

    Helper function for rollingthunder
    '''
    df.reset_index(drop=True, inplace=True)
    df['logform'] = np.log(df.est_val)
    df['logdiff'] = None
    for i in range(1, len(df.logform)):
        df.logdiff[i] = (df.logform[i]-df.logform[i-1])
    df.logdiff[0] = None
    df.logdiff[0] = df.logdiff.mean()
    df.set_index(df.date, inplace=True)
    return df


def make_ready(df,nd):
    '''
    Takes dataframe for single city and returns dataframe with extended index in which
    to append results of forecast
    
    df: dataframe containing variable to forecast with index set to datetime variable
    
    nd: list of new datetimes to append to original dataframe, should be at least as long as
        number of desired periods to forecast ahead
    '''
    dfi = pd.DataFrame(data =None, index = nd)
    dfnew = pd.DataFrame(data =None, index = df.index.append(dfi.index))
    fr = pd.merge(dfnew,aut3, left_index=True, right_index= True, how = 'left')
    return fr

def add_one(df, nd, new_datum):
    '''
    Appends an additional row to working dataframe with latest predicted value
    
    Helper function for rollingthunder
    df: dataframe upon which to predict
    nd:datetime list to append to index
    new_datum: latest prediction from Lex
    '''
    out = df.append({'logdiff': new_datum,'date': nd}, ignore_index=True)
    out.set_index(out.date, inplace=True)
    return out

def single_window(data, n_prev=12):
    '''
    windowizes a single window to generate one prediction;
    since Lex is trained and generating one prediction ahead at a time,
    only need to pass one prediction's worth of data.
    
    Helper function for rollingthunder
    '''
    x = np.array(data[-n_prev:])
    x.reshape(1,12,1)
    return x

def rollingthunder(df, nd, model, ndlist, num = 18, n_prev=12):
    '''
    Takes dataframe, transforms data, sets and extends time index, and then generates
    multiple forecasts by individaul points and then appending those points into data
    to generate subsequent point.
    
    df: dataframe
    
    nd:datetime list to append to index
    
    model: rnn model used to generate predictions
    
    ndlist: list of strings of dates to be appended to index, 
            used to append new values properly
    
    num: number of future predictions to generate
    
    n_prev: number of past observations to use when calculating next value
    '''
    df = oprep(df)
    y_crit = df['logform'][-1]
    predict_list = []

    for i in range(num):
        xactual = single_window(df.logdiff)
        prediction = model.predict(xactual.reshape(1,12,1))
        predict_list.append(prediction[-1][0])
        df = add_one(df,nd[i],prediction[-1][0])
    return y_crit, predict_list, df

def insample_predict(df):
    '''
    Takes data for city and generates x, y arrays to generate predictions for entire timeframe.
    To be used to find measurement of error of model for city to calculate forecast interval
    Helper function for calc_interval
    
    df: dataframe of city data to forecast
    '''
    df.reset_index(drop=True, inplace=True)
    df.set_index(aut3.date, inplace=True)
    set_x, set_y = windowize_pan_data_LD(df.est_val,df.id,12)
    
    testresults = Lex.predict(apredx)
    tr= [testresults[i][0] for i in range(0,len(testresults))]
    trarray = np.array(tr)
    return trarray, set_y

def calc_interval():
    '''
    Calculates size of interval forecast based on difference of in-sample predictions and
    historic data
    Helper function for interval predict(TBW)
    '''
    predicted, actual = insample_predict(df)
    sq_err= (actual-predicted)**2
    rse = sq_err**.5
    crit = rse*1.65
    return crit




'''~~~~~~~~~~~~~~~~~~~~~~~~Modified From The Ether~~~~~~~~~~~~~~~~~~~~~~~~'''


def evaluate_arima_model(X, timevar, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, dates= timevar, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    bic = model.bic
    aic = modle.aic
    return error, aic, bic

def evaluate_models(dataset, timevar, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        print('*')
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse , aic, bic = evaluate_arima_model(dataset, timevar, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f AIC%s BIC%s' % (order,mse, aic, bic))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# def evaluate_models(dataset, timevar, p_values, d_values, q_values):
#     dataset = dataset.astype('float32')
#     best_score, best_cfg = float("inf"), None
#     for p in p_values:
#         print('*')
#         for d in d_values:
#             for q in q_values:
#                 order = (p,d,q)
#                 try:
#                     mse = evaluate_arima_model(dataset, timevar, order)
#                     if mse < best_score:
#                         best_score, best_cfg = mse, order
#                     print('ARIMA%s MSE=%.3f' % (order,mse))
#                 except:
#                     continue
#     print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


# def batch_producer(raw_data, batch_size, num_steps):
#     # raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

#     data_len = tf.size(raw_data)
#     batch_len = data_len // batch_size
#     data = tf.reshape(raw_data[0: batch_size * batch_len],
#                       [batch_size, batch_len])

#     epoch_size = (batch_len - 1) // num_steps

#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
#     x = data[:, i * num_steps:(i + 1) * num_steps]
#     x.set_shape([batch_size, num_steps])
#     y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
#     y.set_shape([batch_size, num_steps])
#     return x, y

def batch_producer(raw_data, data_len, batch_size, num_steps):
    # raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    # data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    return x, y

'''~~~~~~~~~~~~~~~~~~~~~~~~~~Borrowed from Lecture~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def windowize_data(data, n_prev=12):
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices, None]
    return x, y

def split_and_windowize(data, n_prev, fraction_test=0.3):
    n_predictions = len(data) - 2*n_prev
    
    n_test  = int(fraction_test * n_predictions)
    n_train = n_predictions - n_test   
    
    x_train, y_train = windowize_data(data[:n_train], n_prev)
    x_test, y_test = windowize_data(data[n_train:], n_prev)
    return x_train, x_test, y_train, y_test


