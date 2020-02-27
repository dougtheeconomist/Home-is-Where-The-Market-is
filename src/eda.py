# Title: Housing EDA
# Author: Doug Hart
# Date Created: 2/26/2020
# Last Updated: 2/26/2020

import pandas as pd
import numpy as np
import plotly.express as px
import datetime
import statsmodels.api as sm

#Note: region id for Seattle: 16037
# Bellingham is 50950

df = pd.read_pickle('user_ready.pkl',compression='zip')


'''~~~~~~~~~~~~~~Examining Histograms with Plotly Express~~~~~~~~~~~~~~'''

#first looking at dependent variable
fig = px.histogram(x=df.med_price, nbins = 200, range_x=[0,1500000])
fig.update_layout(
    title="Median Home Prices Across Markets And Time in Months",
    xaxis_title="Median in Dollars",
    yaxis_title="Count",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
fig

#then look at mean of this over time
fig = px.histogram(x=df.med_price.mean(level = 'date'), nbins = 25)
fig.update_layout(
    title="Median Home Prices Averaged Across Time",
    xaxis_title="Median in Dollars",
    yaxis_title="Count",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
fig

#then mean by area, same as groupby(['city','state'])
fig = px.histogram(x=df.med_price.mean(level = 'city_id'), nbins = 200, range_x=[0,1500000])
fig.update_layout(
    title="Median Home Prices Averaged Across Markets",
    xaxis_title="Median in Dollars",
    yaxis_title="Count",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
fig

#then looking at volume sold
fig = px.histogram(x=df.sale_count, nbins = 400, range_x=[0,800] )
fig.update_layout(
    title="Median Homes Sold Across Markets And Time in Months",
    xaxis_title="Median in Dollars",
    yaxis_title="Count",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    ))
fig

run_arima(sdf.med_price, sdf.date2, 1,1,5)



'''~~~~~~~~~~~~~~~~~~~~~~~~~~From The Ither~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error



import warnings
warnings.filterwarnings("ignore")

# evaluate combinations of p, d and q values for an ARIMA model
#this requires lists of potential values for p,d,q

p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 6)
q_values = range(0, 6)
def evaluate_models(dataset, p_values, d_values, q_values):
	# dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))