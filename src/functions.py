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


def run_arima(series, date, p,d,q):
    '''should run ARIMA regression on specified series
    need to add returns for fit statistics for comparison
    series: column of df to forecast
    dates: date column, as multi-index doesn't seem compatible here
    
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters to use.
    '''
    model = ARIMA(series, dates = date, order=(p, d, q))
    return model.fit()