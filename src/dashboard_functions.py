# Project: For sale home price forcasting
# Author: Doug Hart
# Date Created: 3/1/2020
# Last Updated: 3/1/2020

import pandas as pd
import numpy as np

def find_town(city, df):
    '''
    Used to find specific location input by user, if not found returns message
    to pass back to user

    city: user input 

    df: dataframe, specified by user selected field by house size
    '''
    for i in range(0, len(df)):
        if df.city[i] == city:
            return df.iloc[i]
    Return 'Not Found, not enough data for this location'

def find_check(output):
    check = type(output)
    if check = str:
        return False
    else:
        return True
    
