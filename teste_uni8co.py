# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 11:57:53 2021

@author: Pedro
"""
# In[]

from darts.models import ExponentialSmoothing
from functions import get_finance_data
import matplotlib.pyplot as plt
from darts import TimeSeries


# In[]

ticker = 'BBAS3.SA'

df = get_finance_data(ticker)
# df.index.freq = 'C'
df = df.reset_index()[['Close']]
train, test = df[0:-52], df[-52:]
predictions = []

series = TimeSeries.from_dataframe(df, value_cols='Close', fill_missing_dates=False)

for t in range(len(test)):
    m = ExponentialSmoothing()
    m.fit(train['y'])
    yhat = m.predict(1)[0]
    predictions.append(yhat)
    obs = test[t]
    train.append(obs)
    int('predicted=%f, observed=%f' % (yhat, obs))


df['y'].plot(label='actual')
predictions.plot(label='forecast')
plt.legend()
plt.xlabel('Date')
