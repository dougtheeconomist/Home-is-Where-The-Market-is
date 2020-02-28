# Title: functions
# Author: Doug Hart
# Date Created: 2/25/2020
# Last Updated: 2/26/2020

import pandas as pd
import numpy as np

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
    n_predictions = len(data) - n_prev
    y = data[n_prev:]
    # this might be too clever
    indices = np.arange(n_prev) + np.arange(n_predictions)[:, None]
    x = data[indices, None]
    xlist = [i for i in x]
    ylist = [i for i in y]
    return xlist, ylist



def windowize_pan_data(data, id, n_prev):
    gx = []
    gy = []
    for i in sdf.cid:
        gslice = []
        for j in range(0, sdf.shape[0]):
            if sdf.cid[j] == i:
                gslice.append(sdf.med_price[j])

        gser = pd.Series(gslice)
    #     print('*')
        glists, gvals = winpan_helper(gser, n_prev)
        gy += gvals
        gx += glists
    return gx, gy





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
	return error



def evaluate_models(dataset, timevar, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        print('*')
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, timevar, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


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