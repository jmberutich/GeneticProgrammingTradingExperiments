# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 02:09:20 2016

@author: jm
"""

import os
import pandas as pd
import numpy as np
import cPickle



# load the ibex data
ibex = cPickle.load(open("ibex.pickle","rb"))

# Converto to log prices
dataset = ibex.copy()
dataset.Open = np.log(ibex.Open)
dataset.High = np.log(ibex.High)
dataset.Low = np.log(ibex.Low)
dataset.Close = np.log(ibex.Close)

dataset["HLRange"] = dataset.High - dataset-Low
dataset["CORange"] = dataset.Close - dataset.Open

dataset["HH"] = 

# build the features for 1 Min 
for period in range(2,30):
    a    

day = ibex.resample("1d", how={"Open" : "first", "High" : "max", "Low" : "min", "Close" : "last", "Volume" : "sum"}).dropna()
indices = day.index

for i in range(5,201,5)

dataset['LastDayClose'] = np.nan
dataset['LastDayHigh'] = np.nan
dataset['LastDayLow'] = np.nan

for i in range(0,len(indices)-1):
    dataset.loc[indices[i+1].date().strftime('%Y-%m-%d'), 'LastDayClose'] = day.ix[i]['Close']
    dataset.loc[indices[i+1].date().strftime('%Y-%m-%d'), 'LastDayLow'] = day.ix[i]['Low']
    dataset.loc[indices[i+1].date().strftime('%Y-%m-%d'), 'LastDayHigh'] = day.ix[i]['High']
  
  

ibs = (day['Close']-day['Low'])/(day['High']-day['Low'])
ibs1min = ibs.resample("1Min", fill_method="bfill")
ibs1min = ibs1min[ibex.index]
r = close.diff()
r
r = close.diff()/close[0:-1]
r = close.log().diff()
r = np.log(close).diff()
r_close = np.log(close).diff()
r_open = np.log(open).diff()
r_high = np.log(high).diff()
r_low = np.log(low).diff()
mfm = ((close-low)-(high-close) / (high-low))
mvf = mfm * vol
cmf = pd.rolling_sum(mvf,20) / pd.rolling_sum(vol, 20)


indices = day.index

  
    #dataset.loc[indices[i+1].date().strftime('%Y-%m-%d')]['LastDayClose'] = day.loc[indices[i].date().strftime('%Y-%m-%d')]['Close']
    
