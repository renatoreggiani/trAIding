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
import matplotlib.pyplot as plt


# In[]

ticker = 'BBAS3.SA'
tkr = yf.Ticker(ticker)
df = tkr.history(period='5y', interval='1d')
df = df.rename(columns={'Close': 'y'}, inplace=False)
series = TimeSeries.from_dataframe(df, time_col=None, value_cols='y', freq='D')
train, val = series.split_after(pd.Timestamp('2020-05-02'))

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val))

series.plot(label='actual')
prediction.plot(label='forecast')
plt.legend()
plt.xlabel('Date')
