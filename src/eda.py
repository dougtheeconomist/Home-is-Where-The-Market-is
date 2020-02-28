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



#Note Jack suggested narrowing to WA cities, this is a reasonable plan to me
