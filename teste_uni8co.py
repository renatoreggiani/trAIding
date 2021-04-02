# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:57:53 2021

@author: Pedro
"""
# In[]

import yfinance as yf
import pandas as pd
from darts.models import ExponentialSmoothing
from darts import TimeSeries
import time
import datetime


# In[]

ticker = 'BBAS3.SA'
tkr = yf.Ticker(ticker)
df = tkr.history(period='1y', interval='1d')
df = df.rename(columns={'Close': 'y'}, inplace=False)
series = TimeSeries.from_dataframe(df, time_col=None, value_cols='y', freq='D')
t = time.mktime(datetime.datetime.strptime('2020-05-01', "%Y-%m-%d").timetuple())
train, val = series.split_after(pd.Timestamp(t))  # erro fdp!

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))
